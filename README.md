# ultrasound-project

Master-thesis code for robotic ultrasound, built on [SonoGym](https://github.com/SonoGym/SonoGym)
and NVIDIA Isaac Lab:

- **Guidance tasks**: Isaac Lab environments for probe guidance, with a wrist camera or on a
  CT-derived lung phantom, and a scripted raster-scan controller.
- **Deformation pipeline**: a probe presses into a SonoGym patient in PhysX FEM, the CT is
  warped by the tissue motion, and SonoGym's ultrasound simulators image the deformed CT.

SonoGym is included as the `sonogym-custom/` submodule, a fork with a few bug fixes.

## Requirements

- Linux with an NVIDIA GPU (PhysX soft bodies run only on the GPU)
- Isaac Sim 4.5.0 and Isaac Lab 2.1.0 in a conda environment with Python 3.10
  ([installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html))
- [Git LFS](https://git-lfs.com)

## Setup

```bash
git lfs install
git clone --recurse-submodules https://github.com/Zeng-Charles/master-thesis.git
cd master-thesis

conda activate <your-isaaclab-env>
pip install -e sonogym-custom/source/spinal_surgery --no-dependencies
pip install -e .
pip install -r requirements-dev.txt    # optional: pytest and formatters
```

Download SonoGym's assets and ultrasound models from
<https://huggingface.co/datasets/yunkao/SonoGym_assets_models> and unpack the two folders:

```text
assets/  ->  sonogym-custom/source/spinal_surgery/spinal_surgery/assets/
models/  ->  sonogym-custom/models/
```

## Run

### Guidance tasks

```bash
# guidance task with a wrist camera, random actions
python scripts/run_robot_US_guidance_cam_env.py --num_envs 4

# scripted serpentine scan; press SPACE in the viewport to start
python scripts/scan_trajectory_agent.py --task Isaac-robot-US-guidance-v0 --num_envs 4
```

Task ids and the current state of training: [docs/tasks.md](docs/tasks.md).

### Deformation pipeline

```bash
python -u scripts/deformation/patient_press_record.py --headless --capture   # 1. FEM press (~1 min)
python scripts/deformation/deformed_ct_to_us.py                             # 2. deformed CT + ultrasound (~1 min)
python scripts/deformation/make_report.py                                   # 3. HTML report
```

Only step 1 runs Isaac Sim; drop `--headless` to watch the press in its viewport. Steps 2
and 3 run offline and only write files to `outputs/deformation_us/`. To see the images, open
`outputs/deformation_us/report/index.html`. How it works and what the checks found:
[docs/deformation_pipeline.md](docs/deformation_pipeline.md).

A smaller soft-body check on a box: `python -u scripts/deformation/soft_body_press_demo.py --headless`
([docs/soft_body_press.md](docs/soft_body_press.md)).

## Tests

```bash
python -m pytest tests    # CPU only, no Isaac Sim needed
```

## Troubleshooting

- **pytest fails inside `/opt/ros/...`**: a sourced ROS 2 puts its pytest plugins on
  `PYTHONPATH`. Run `PYTHONPATH= python -m pytest tests`.
- **`Torch not compiled with CUDA enabled`**: a CPU-only torch in `~/.local` shadows the
  conda environment's. Make the environment ignore `~/.local` for good with
  `conda env config vars set PYTHONNOUSERSITE=1`, then re-activate it.
- **An Isaac Sim script prints nothing**: Isaac Sim exits hard on shutdown and drops
  buffered output. Run it with `python -u`.

## Layout

```text
sonogym-custom/           SonoGym fork (submodule)
src/ultrasound_project/   installable package
  tasks/                  Isaac Lab tasks, registered on `import ultrasound_project`
  controllers/            RasterScanController
  deformation/            FEM displacement fields, CT warping
  sensors/                SonoGym's ultrasound simulators, run outside Isaac Sim
  utils/                  patient assets and frames, image metrics
scripts/                  entry points; scripts/deformation/ is the deformation pipeline
tools/                    CT -> simulation body, standalone ultrasound network test
configs/                  environment and training configs
assets/patients/          lung phantom (Git LFS)
tests/                    CPU-only tests
docs/                     technical documentation
data/, outputs/           datasets and run outputs (not versioned)
```

## Documentation

- [docs/deformation_pipeline.md](docs/deformation_pipeline.md): FEM press, CT warp and ultrasound rendering; checks and known limitations
- [docs/soft_body_press.md](docs/soft_body_press.md): the box press demo and displacement-field export
- [docs/tasks.md](docs/tasks.md): task ids, training, what goes in the fork, the lung phantom, the standalone network test
- [assets/patients/lung_phantom/README.md](assets/patients/lung_phantom/README.md): the phantom's files and how to regenerate them

## Data

The lung phantom is derived from **LUNG-PHANTOM** on The Cancer Imaging Archive
(<https://www.cancerimagingarchive.net/collection/lung-phantom/>), CC BY 3.0. TCIA
requires this citation:

> Zhao, B. (2015). *Lung Phantom* (Version 2) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/k9/tcia.2015.08a1ixoo>
