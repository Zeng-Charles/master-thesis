# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to scan the body surface along a predefined serpentine (raster) trajectory.

Unlike sonogym-custom/workflows/teleoperation/teleop_se3_agent.py, this script takes no
human input: the probe path is fully determined ahead of time by RasterScanController.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Predefined-trajectory surface scan (no teleop, no RL policy)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=9, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-robot-US-guidance-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_sections",
    type=int,
    default=2,
    help="Number of raster columns of the serpentine scan path.",
)
parser.add_argument(
    "--total_steps",
    type=int,
    default=500,
    help="Number of env steps to complete one full scan pass.",
)
parser.add_argument(
    "--square_size",
    type=float,
    default=25.0,
    help="Half-width (in surface grid units) of the scan rectangle, centered on the target vertebra.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import wandb

from isaaclab.devices import Se3Keyboard

import isaaclab_tasks  # noqa: F401
import spinal_surgery  # noqa: F401
import ultrasound_project  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from ultrasound_project.controllers import RasterScanController


def main():
    """Run a predefined serpentine scan trajectory with no human/RL input."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # the task's internal episode time-out (derived from episode_length_s) auto-resets and
    # randomly relocates the probe every ~episode_length_s seconds; stretch it out so an
    # in-progress scan is never interrupted. NOTE: this must be set on env_cfg *before*
    # gym.make() -- env.max_episode_length is a read-only property computed from
    # env_cfg.episode_length_s, so assigning it on the env instance after creation is a no-op.
    env_cfg.episode_length_s = 1.0e9

    # init wandb
    wandb.init(project=args_cli.task, config=env_cfg)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # scan region: a square around each env's target vertebra, in surface-grid units
    vertebra_2d_pos = env.vertebra_viewer.human_to_ver_per_envs[:, [0, 2]]
    scan_planner = RasterScanController(
        max_action=env.max_action,
        action_scale=env.action_scale,
        human_pos_2d_min=vertebra_2d_pos - args_cli.square_size,
        human_pos_2d_max=vertebra_2d_pos + args_cli.square_size,
        num_sections=args_cli.num_sections,
        total_steps=args_cli.total_steps,
        ratio=[0.05, 0.05, 0.05],
        device=env.sim.device,
        cmd_state_key="cur_cmd_pose",
        action_dim=3,
    )

    # scan state, toggled by the SPACE key callback below
    scan_state = {"active": False, "step": 0}

    def start_scan():
        scan_state["active"] = True
        scan_state["step"] = 0
        print("[scripted_scan] scan started")

    keyboard = Se3Keyboard()
    keyboard.add_callback("SPACE", start_scan)
    print("[scripted_scan] press SPACE to (re)start the scan. Close the window to quit.")

    # reset environment
    obs, info = env.reset()

    idle_action = torch.zeros((args_cli.num_envs, 3), device=env.sim.device)

    # run until the user closes the Isaac Sim window; never closes itself
    while simulation_app.is_running():
        with torch.inference_mode():
            if scan_state["active"]:
                actions = scan_planner.get_action(info, scan_state["step"])
                scan_state["step"] += 1
                if scan_state["step"] >= args_cli.total_steps:
                    scan_state["active"] = False
                    print("[scripted_scan] scan finished, press SPACE to scan again")
            else:
                actions = idle_action
            obs, reward, term, time_out_tensor, info = env.step(actions)

    # close the simulator (only reached after the user closes the window)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
