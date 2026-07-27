
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal driver for Isaac-robot-US-guidance-cam-v0: drives the guidance task with random
actions (same pattern as scripts/run_robot_US_guidance_env.py). By default (GUI mode
only), this opens a separate Isaac Sim viewport window looking through env 0's D435
camera at startup -- the original scene viewport is left untouched, so both are visible
side by side. This uses Isaac Sim's own viewport rendering (omni.kit.viewport.utility)
rather than a separate OpenCV window, since cv2.imshow needs a GUI-enabled OpenCV build
that most Isaac Lab conda envs don't have (opencv-python-headless is used instead, to
avoid conflicting with Kit's own Qt/GTK) and raises "The function is not implemented".

Pass --no_live_view to skip opening the extra window.

Pass --save_frames to additionally dump periodic RGB/depth PNGs to disk via cv2.imwrite
(this does not need GUI support, so it works regardless; off by default).

.. code-block:: bash

    python scripts/run_robot_US_guidance_cam_env.py --num_envs 4
    python scripts/run_robot_US_guidance_cam_env.py --num_envs 4 --no_live_view --save_frames

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run robot US guidance + D435 camera environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guidance-cam-v0', help="Name of the task")
parser.add_argument(
    "--no_live_view",
    action="store_true",
    default=False,
    help="Don't open a separate D435 viewport window at startup (GUI mode only).",
)
parser.add_argument(
    "--save_frames",
    action="store_true",
    default=False,
    help="Periodically save D435 RGB/depth frames to --save_dir (off by default).",
)
parser.add_argument("--save_every", type=int, default=50, help="Save a camera frame every N steps (if --save_frames).")
parser.add_argument(
    "--save_dir",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "d435_check"),
    help="Directory to save camera frames (absolute by default, so it doesn't depend on cwd).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# this task has a Camera/TiledCamera sensor -- Isaac Sim's render pipeline for camera
# sensors is only initialized when enable_cameras is on, so force it regardless of what
# was passed on the CLI (otherwise TiledCamera._initialize_impl() silently fails to
# fully set up and later crashes with "'TiledCamera' object has no attribute '_timestamp'")
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym

import spinal_surgery  # noqa: F401
import ultrasound_project  # noqa: F401  -- registers Isaac-robot-US-guidance-cam-v0


def _get_camera_frames(env):
    """Fetch env 0's current D435 RGB (BGR for cv2) + normalized depth-visualization frames."""
    d435 = env.unwrapped.d435
    rgb = d435.data.output["rgb"][0].detach().cpu().numpy()  # (H, W, 3) uint8
    depth = d435.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()  # (H, W) float32

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    near, far = env.unwrapped.cfg.d435_cfg.spawn.clipping_range
    depth_vis = np.nan_to_num(depth, posinf=far, neginf=near)
    depth_vis = np.clip((depth_vis - near) / (far - near), 0.0, 1.0)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    return rgb, rgb_bgr, depth, depth_vis


def create_d435_viewport(env):
    """Open a separate Isaac Sim viewport window looking through env 0's D435 camera.

    Uses a dedicated new window (via omni.kit.viewport.utility.create_viewport_window,
    which accepts a camera_path directly) rather than repointing the main viewport, so
    the original free-fly scene view stays visible alongside the camera feed. The
    window only needs to be told once which camera to look through -- the D435 prim
    itself moves with the robot automatically.
    """
    from omni.kit.viewport.utility import create_viewport_window

    # resolve the "{ENV_REGEX_NS}"-style template (env_.*) to the concrete env 0 path
    camera_path = env.unwrapped.cfg.d435_cfg.prim_path.replace("env_.*", "env_0")
    d435_cfg = env.unwrapped.cfg.d435_cfg
    window = create_viewport_window(
        name="D435 Camera", camera_path=camera_path, width=d435_cfg.width, height=d435_cfg.height
    )
    if window is None:
        print("[WARNING]: failed to create a separate D435 viewport window.")
        return
    print(f"[INFO]: opened a separate 'D435 Camera' viewport window looking through {camera_path}")


def save_camera_frame(env, step, save_dir):
    """Save env 0's current D435 RGB + depth output as PNGs under save_dir."""
    rgb, rgb_bgr, depth, depth_vis = _get_camera_frames(env)

    os.makedirs(save_dir, exist_ok=True)
    rgb_path = os.path.join(save_dir, f"rgb_{step:06d}.png")
    depth_path = os.path.join(save_dir, f"depth_{step:06d}.png")
    cv2.imwrite(rgb_path, rgb_bgr)
    cv2.imwrite(depth_path, depth_vis)

    # print basic stats so an all-black/blank frame (camera attached but not actually
    # rendering anything -- e.g. buried inside the robot mesh) is obvious from the logs
    # even before opening the PNGs.
    print(
        f"[INFO]: saved {rgb_path} (rgb mean={rgb.mean():.1f}, min={rgb.min()}, max={rgb.max()}) "
        f"and {depth_path} (depth min={np.nanmin(depth):.3f}, max={np.nanmax(np.where(np.isinf(depth), np.nan, depth)):.3f})"
    )


def main():
    """Main function."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    save_dir = os.path.abspath(args_cli.save_dir)
    print(f"[INFO]: D435 camera prim_path template: {env.unwrapped.cfg.d435_cfg.prim_path}")
    print(f"[INFO]: D435 world position (env 0): {env.unwrapped.d435.data.pos_w[0].tolist()}")
    if args_cli.save_frames:
        print(f"[INFO]: saving D435 frames to {save_dir}")
    if not args_cli.no_live_view and not args_cli.headless:
        create_d435_viewport(env)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            commands = torch.rand((args_cli.num_envs, 3), device=env.unwrapped.device) * 2.0 - 1.0
            commands[:, 2] = commands[:, 2] / 10

            # step the environment
            obs, rew, terminated, truncated, info = env.step(commands)

            # periodically dump the D435 view so the mount offset can be checked/tuned
            if args_cli.save_frames and count % args_cli.save_every == 0:
                save_camera_frame(env, count, save_dir)

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
