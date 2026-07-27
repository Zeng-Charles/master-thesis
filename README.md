# ultrasound-project

Master-thesis work on robotic ultrasound, built on
[SonoGym](https://github.com/SonoGym/SonoGym) / NVIDIA Isaac Lab.

SonoGym is not vendored here — it is the `sonogym-custom/` submodule, tracking
[a fork](https://github.com/Zeng-Charles/SonoGym) that carries only fixes to files which
already exist upstream. Everything new lives in this repo.

## Tasks

All five register when you `import ultrasound_project`.

| Gym id | Defined in | Scene config |
| --- | --- | --- |
| `Isaac-robot-US-guidance-cam-v0` | this repo | `configs/env/robotic_US_guidance_cam.yaml` |
| `Isaac-robot-US-guidance-phantom-v0` | this repo | `configs/env/robotic_US_guidance_phantom.yaml` |
| `Isaac-robot-US-guidance-v0` | submodule | `sonogym-custom/…/robot_US_guidance/cfgs/…yaml` |
| `Isaac-robot-US-reconstruction-v0` | submodule | `sonogym-custom/…/robot_US_reconstruction/cfgs/…yaml` |
| `Isaac-robot-US-guided-surgery-v0` | submodule | `sonogym-custom/…/robot_US_guided_surgery/cfgs/…yaml` |

`-cam-v0` is the guidance task plus a RealSense D435 on the end effector, reachable via
`env.unwrapped.d435.data.output`. `-phantom-v0` is the same task against a CT-derived lung
phantom. Both are copies of SonoGym's `robotic_US_guidance.py` rather than subclasses,
because that task reads its yaml into module-level globals at import time — copying is
what lets them be retuned from `configs/env/` without touching the fork.

Also here: `RasterScanController` (scripted serpentine surface scan, no RL policy) and the
`tools/` pipeline that turns a phantom CT into a simulation body.

## Layout

```
sonogym-custom/          SonoGym fork (submodule)
src/ultrasound_project/  installable package: tasks/ controllers/ envs/ sensors/
                         observations/ rewards/ utils/
assets/patients/         phantom bodies (Git LFS)
configs/                 env / robot / patient / training configs
scripts/                 entry points
tools/                   offline data preparation
data/  outputs/          datasets and run artefacts (not versioned)
tests/  slurm/  docs/
```

## Setup

```bash
git lfs install
git clone --recurse-submodules https://github.com/Zeng-Charles/master-thesis.git
cd master-thesis
```

Then follow [SonoGym's README](sonogym-custom/README.md) for Isaac Sim, Isaac Lab, the
asset/model downloads and the extension install — everything applies unchanged, just
under `sonogym-custom/` instead of the repo root:

```bash
python -m pip install -e sonogym-custom/source/spinal_surgery --no-dependencies
```

Its asset archives unpack to `sonogym-custom/source/spinal_surgery/spinal_surgery/assets`
and `sonogym-custom/models`. Both tasks here use `sim.us: 'net'`, so `models/` is
required, not optional.

Then install this package into the same environment:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt   # optional
```

The expert lerobot dataset (imitation learning only) is SonoGym's too; there is a wrapper
for it that lands in `data/raw/lerobot-dataset/`:

```bash
python scripts/download_lerobot_dataset.py          # what the shipped configs use
python scripts/download_lerobot_dataset.py --all    # all 8 (~34 GB)
```

### Phantom body

`assets/patients/lung_phantom/` ships with the clone via Git LFS. It is derived from
**LUNG-PHANTOM** on The Cancer Imaging Archive
(<https://www.cancerimagingarchive.net/collection/lung-phantom/>), a CT of the FDA
anthropomorphic thorax phantom, CC BY 3.0. TCIA requires this citation:

> Zhao, B. (2015). *Lung Phantom* (Version 2) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/k9/tcia.2015.08a1ixoo>

See [its README](assets/patients/lung_phantom/README.md) for the file list and the
regeneration pipeline.

## Running

```bash
# guidance + D435, random actions, live camera viewport
python scripts/run_robot_US_guidance_cam_env.py --num_envs 4

# scripted serpentine scan -- press SPACE in the viewport to start
python scripts/scan_trajectory_agent.py --task Isaac-robot-US-guidance-v0 --num_envs 4

# training etc. use SonoGym's own workflows, from the submodule
python sonogym-custom/workflows/skrl/train.py --task Isaac-robot-US-guidance-phantom-v0
```

Submodule workflows that need this repo's environments must `import ultrasound_project`;
the scripts here already do.

### Inspecting the ultrasound renderer

`tools/test_us_network_standalone.py` runs the pix2pix CT→US network on its own, without
Isaac Sim, so you can see what it renders before wiring up a simulation. One slice gives a
PNG; `--sweep` walks the volume and gives an animated GIF (or `.mp4`/`.avi`). Output is
always CT input on the left, simulated US on the right.

```bash
CT=assets/patients/lung_phantom/raw/ct.nii.gz

# single axial slice -> PNG
python tools/test_us_network_standalone.py --ct $CT --slice_index 210 --output us.png

# every 4th slice -> GIF at 10 fps
python tools/test_us_network_standalone.py --ct $CT --sweep --slice_step 4 --output us.gif

# a slice range -> mp4
python tools/test_us_network_standalone.py --ct $CT --sweep \
    --slice_start 150 --slice_end 300 --fps 15 --output us.mp4
```

Useful flags: `--slice_axis {0,1,2}` picks the slicing axis (default 2, axial),
`--ct_range LOW HIGH` overrides the HU window fed to the network without editing
`us_generative_cfg.yaml`, and `--device cuda` moves it off the CPU.

## What goes where

- **fork (`sonogym-custom/`)** — only fixes to existing SonoGym files: a batching bug in
  `HumanFrameViewer` breaking `num_envs == 1`, `torch.load` without `map_location`
  breaking CPU-only loading, an import-time side effect in `convert_stl_to_urdf.py`, and
  two run-time defaults.
- **this repo** — every new task, controller, tool, script, config and asset.

Touch the fork only when the change is to a file that already exists upstream.
