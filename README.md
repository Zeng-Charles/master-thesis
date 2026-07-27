# ultrasound-project

Master-thesis work on robotic ultrasound, built on top of
[SonoGym](https://github.com/SonoGym/SonoGym) / NVIDIA Isaac Lab.

SonoGym itself is not vendored into this repo: it is the `sonogym-custom/` submodule,
which tracks [a fork](https://github.com/Zeng-Charles/SonoGym) carrying only a handful of
upstream fixes. Everything that is genuinely new — extra tasks, the scripted-scan
controller, the phantom pipeline, configs and assets — lives here.

## What this repo adds

| Gym id | Description |
| --- | --- |
| `Isaac-robot-US-guidance-cam-v0` | SonoGym's guidance task with a RealSense D435 mounted on the end effector. Behaviour is unchanged; the camera feed is reachable via `env.unwrapped.d435.data.output`. |
| `Isaac-robot-US-guidance-phantom-v0` | The guidance task against a CT-derived physical lung phantom instead of a patient body, with its own config and assets. |

Plus `RasterScanController` (a scripted serpentine surface scan, no RL policy), the
CT → mask → mesh → USD tooling under `tools/` used to turn a phantom scan into a
simulation body, and a standalone tester for the pix2pix ultrasound renderer.

## Layout

```
sonogym-custom/          SonoGym fork (git submodule) -- upstream code + minimal fixes
src/ultrasound_project/  installable package; importing it registers the tasks above
  tasks/                 gym environments
  controllers/           scripted controllers (RasterScanController)
  envs/ sensors/ observations/ rewards/ utils/
assets/                  simulation bodies, robots, probes, scenes (git-ignored)
configs/                 env / robot / patient / training configs
scripts/                 entry points you run
tools/                   offline data-preparation utilities
tests/  slurm/  docs/
data/                    raw + processed datasets (git-ignored)
outputs/                 run artefacts (git-ignored)
```

Neither `assets/` nor `data/` is versioned: individual files run to hundreds of MB, past
GitHub's 100 MB per-file limit, and this repo does not use Git LFS. See
[Getting the assets and data](#getting-the-assets-and-data).

## Setup

```bash
git clone --recurse-submodules https://github.com/Zeng-Charles/master-thesis.git
cd master-thesis
# already cloned without submodules?
git submodule update --init --recursive
```

1. **Isaac Sim + Isaac Lab.** Follow the
   [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   and activate its conda environment. Everything below runs inside that environment.

2. **SonoGym.** Installed from the submodule, without dependencies so it cannot disturb
   the Isaac Lab environment:

   ```bash
   python -m pip install -e sonogym-custom/source/spinal_surgery --no-dependencies
   ```

3. **This package.**

   ```bash
   python -m pip install -e .
   python -m pip install -r requirements-dev.txt   # optional, for linting/tests
   ```

## Getting the assets and data

### SonoGym's own assets and US models

Download from
<https://huggingface.co/datasets/yunkao/SonoGym_assets_models> and unpack into the
submodule, exactly as SonoGym's own README describes — the bed, robots, probes and the
pix2pix checkpoints are resolved relative to the `spinal_surgery` package:

```
assets -> sonogym-custom/source/spinal_surgery/spinal_surgery/assets
models -> sonogym-custom/models
```

The `sim.us: 'net'` renderer used by the phantom task needs `models/`; without it the
task fails at startup.

### Phantom body

`assets/patients/lung_phantom/` — see
[its README](assets/patients/lung_phantom/README.md) for the exact file list and the
`tools/` pipeline that regenerates it from the raw CT in `data/raw/`.

### Expert dataset (optional, imitation learning only)

```bash
python scripts/download_lerobot_dataset.py            # the two datasets the shipped configs use
python scripts/download_lerobot_dataset.py --all      # all 8 (~34 GB)
```

Lands in `data/raw/lerobot-dataset/`. Update `"dataset": {"root": ...}` in
`configs/training/*.json` to match your absolute path.

## Running

```bash
# guidance task with the D435 camera, random actions, live camera viewport
python scripts/run_robot_US_guidance_cam_env.py --num_envs 4

# scripted serpentine surface scan (press SPACE in the viewport to start)
python scripts/scan_trajectory_agent.py --task Isaac-robot-US-guidance-v0 --num_envs 4

# the phantom task, e.g. with SonoGym's own skrl training workflow
python sonogym-custom/workflows/skrl/train.py --task Isaac-robot-US-guidance-phantom-v0
```

Scripts that use SonoGym's workflows (`train.py`, `play.py`, teleoperation, lerobot
recording) live in the submodule and are called from there. Any of them that needs this
repo's environments must `import ultrasound_project` — `scripts/` here already does.

Standalone check of the ultrasound renderer, no Isaac Sim required:

```bash
python tools/test_us_network_standalone.py \
    --ct assets/patients/lung_phantom/raw/ct.nii.gz --slice_index 210 --output us_test.png
```

## What goes where

The split is deliberate, to keep the fork close to upstream and mergeable:

- **fork (`sonogym-custom/`)** — only fixes to existing SonoGym files: a batching bug in
  `HumanFrameViewer` that broke `num_envs == 1`, `torch.load` without `map_location`
  breaking CPU-only model loading, an import-time side effect in `convert_stl_to_urdf.py`,
  and two run-time defaults (trajectory recording off, longer teleop episodes).
- **this repo** — every new task, controller, tool, script, config and asset.

New code should default to this repo; touch the fork only when the change is to a file
that already exists upstream.
