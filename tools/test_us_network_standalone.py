"""Standalone test of the CT-to-ultrasound network (USSimulatorNetwork), with no Isaac
Lab/Isaac Sim dependency -- lets you sanity-check what the pix2pix/UNet model actually
renders for a given patient/phantom CT slice before wiring up the full simulation.

`import spinal_surgery` (which simulate_US_network.py normally relies on for its
PACKAGE_DIR/PROJECT_DIR constants) pulls in the whole Isaac Lab task registry as a side
effect and needs omni.kit / a full Isaac Sim install. This script avoids that by stubbing
just those two path constants in sys.modules and loading simulate_US_network.py directly
by file path, so it can run in a plain Python + torch + monai environment.

Usage:
    # single slice -> PNG
    python tools/test_us_network_standalone.py --ct path/to/ct.nii.gz --slice_index 210 --output us_test.png

    # every slice along the axis -> animated GIF (or .mp4/.avi) of CT | simulated US side by side
    python tools/test_us_network_standalone.py --ct path/to/ct.nii.gz --sweep --output us_sweep.gif
"""

import argparse
import importlib.util
import os
import sys
import types

import cv2
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# simulate_US_network.py and the pix2pix checkpoints both live in the SonoGym submodule,
# so PROJECT_DIR must point there and not at this repo's root.
SONOGYM_ROOT = os.environ.get("SONOGYM_ROOT", os.path.join(REPO_ROOT, "sonogym-custom"))
PACKAGE_DIR = os.path.join(SONOGYM_ROOT, "source", "spinal_surgery", "spinal_surgery")
PROJECT_DIR = SONOGYM_ROOT


def _load_simulate_us_network_module():
    if "spinal_surgery" not in sys.modules:
        stub = types.ModuleType("spinal_surgery")
        stub.PACKAGE_DIR = PACKAGE_DIR
        stub.PROJECT_DIR = PROJECT_DIR
        sys.modules["spinal_surgery"] = stub

    module_path = os.path.join(PACKAGE_DIR, "lab", "sensors", "ultrasound", "simulate_US_network.py")
    spec = importlib.util.spec_from_file_location("simulate_US_network", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ct_slice(ct_path: str, slice_axis: int, slice_index: int) -> np.ndarray:
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    return np.take(ct_data, slice_index, axis=slice_axis)


def to_u8(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return (x * 255).astype(np.uint8)


def render_frame(simulator, ct_slice: np.ndarray, device: str) -> np.ndarray:
    """Run one CT slice through the network, return a side-by-side (CT | simulated US) uint8 frame."""
    ct_tensor = torch.tensor(ct_slice, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    us_tensor = simulator.simulate_US_image(ct_tensor)

    ct_vis = to_u8(ct_slice)
    us_vis = to_u8(us_tensor[0, 0].detach().cpu().numpy())
    ct_vis_resized = cv2.resize(ct_vis, (us_vis.shape[1], us_vis.shape[0]))
    return np.hstack([ct_vis_resized, us_vis])


def write_gif(frames: list, output_path: str, fps: float) -> None:
    images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def write_video(frames: list, output_path: str, fps: float) -> None:
    h, w = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output_path.endswith(".mp4") else "MJPG"))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    for f in frames:
        writer.write(f)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ct", required=True, help="Path to a ct.nii.gz volume")
    parser.add_argument("--slice_axis", type=int, default=2, choices=[0, 1, 2], help="Array axis to slice along (0=X,1=Y,2=Z). Default 2 (a conventional axial slice).")
    parser.add_argument("--slice_index", type=int, default=None, help="Slice index along slice_axis (single-slice mode -> PNG). Ignored if --sweep is set.")
    parser.add_argument("--sweep", action="store_true", default=False, help="Process every slice along slice_axis (in [--slice_start, --slice_end) step --slice_step) and write a GIF/video instead of one PNG.")
    parser.add_argument("--slice_start", type=int, default=None, help="Sweep mode: first slice index (default 0).")
    parser.add_argument("--slice_end", type=int, default=None, help="Sweep mode: one past the last slice index (default: full extent of slice_axis).")
    parser.add_argument("--slice_step", type=int, default=1, help="Sweep mode: step between slices (default 1 = every slice; use e.g. 4 to sample every 4th slice for a shorter/faster output).")
    parser.add_argument("--fps", type=float, default=10.0, help="Sweep mode: playback speed of the output GIF/video (default 10).")
    parser.add_argument("--output", required=True, help="Output path. Single-slice mode: a .png. Sweep mode: a .gif, or .mp4/.avi for a video.")
    parser.add_argument("--device", default="cpu", help="torch device (default cpu)")
    parser.add_argument(
        "--ct_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Override CT.range (the HU window clamped/normalized before the network sees it) "
        "without editing us_generative_cfg.yaml -- useful for exploring a better window for "
        "phantom material, which doesn't sit in the same HU range as real human tissue.",
    )
    args = parser.parse_args()
    if not args.sweep and args.slice_index is None:
        parser.error("--slice_index is required unless --sweep is set")

    simulate_US_network = _load_simulate_us_network_module()

    cfg_path = os.path.join(PACKAGE_DIR, "lab", "sensors", "cfgs", "us_generative_cfg.yaml")
    us_generative_cfg = YAML().load(open(cfg_path, "r"))
    if args.ct_range is not None:
        us_generative_cfg["CT"]["range"] = list(args.ct_range)
    print(f"[INFO] CT.range (HU window fed to the network): {us_generative_cfg['CT']['range']}")

    simulator = simulate_US_network.USSimulatorNetwork(us_generative_cfg, args.device)

    if not args.sweep:
        ct_slice = load_ct_slice(args.ct, args.slice_axis, args.slice_index)
        print(f"[INFO] CT slice shape={ct_slice.shape}, min={ct_slice.min():.1f}, max={ct_slice.max():.1f} HU")
        frame = render_frame(simulator, ct_slice, args.device)
        cv2.imwrite(args.output, frame)
        print(f"[INFO] wrote {args.output} (left: CT input, right: simulated US)")
        return

    ct_img = nib.load(args.ct)
    axis_size = ct_img.shape[args.slice_axis]
    start = args.slice_start if args.slice_start is not None else 0
    end = args.slice_end if args.slice_end is not None else axis_size
    indices = list(range(start, end, args.slice_step))
    print(f"[INFO] sweeping {len(indices)} slices (axis={args.slice_axis}, range=[{start},{end}), step={args.slice_step})")

    frames = []
    for i, idx in enumerate(indices):
        ct_slice = load_ct_slice(args.ct, args.slice_axis, idx)
        frames.append(render_frame(simulator, ct_slice, args.device))
        if (i + 1) % 20 == 0 or i == len(indices) - 1:
            print(f"[INFO]   {i + 1}/{len(indices)} slices done (slice_index={idx})")

    if args.output.lower().endswith(".gif"):
        write_gif(frames, args.output, args.fps)
    else:
        write_video(frames, args.output, args.fps)
    print(f"[INFO] wrote {args.output} ({len(frames)} frames @ {args.fps} fps, left: CT input, right: simulated US)")


if __name__ == "__main__":
    main()
