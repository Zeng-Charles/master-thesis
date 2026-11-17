# Isaac Lab tasks

| Gym id | Defined in | Scene config |
| --- | --- | --- |
| `Isaac-robot-US-guidance-cam-v0` | this repo | `configs/env/robotic_US_guidance_cam.yaml` |
| `Isaac-robot-US-guidance-phantom-v0` | this repo | `configs/env/robotic_US_guidance_phantom.yaml` |
| `Isaac-robot-US-guidance-v0` | SonoGym | `sonogym-custom/.../robot_US_guidance/cfgs/...yaml` |
| `Isaac-robot-US-reconstruction-v0` | SonoGym | `sonogym-custom/.../robot_US_reconstruction/cfgs/...yaml` |
| `Isaac-robot-US-guided-surgery-v0` | SonoGym | `sonogym-custom/.../robot_US_guided_surgery/cfgs/...yaml` |

`import ultrasound_project` registers this repo's two tasks and `import spinal_surgery`
registers SonoGym's three. The scripts in `scripts/` import both.

- `-cam-v0` is SonoGym's guidance task plus a RealSense D435 on the end effector. The
  camera is not part of the observation space; read it from
  `env.unwrapped.d435.data.output["rgb"]` or `["distance_to_image_plane"]`.
- `-phantom-v0` is the same guidance task on the CT-derived lung phantom in
  `assets/patients/lung_phantom/`.

Both are near-verbatim copies of SonoGym's `robotic_US_guidance.py`, not subclasses: that
task reads its yaml into module-level globals at import time, so a subclass would be stuck
with the submodule's config. Copying lets them be retuned from `configs/env/` without
touching the fork. The differences are the config and asset paths, the class names and,
for `-cam-v0`, the camera.

## Scripted scan

`RasterScanController` moves the probe along a serpentine (raster) path over the body
surface, with no teleoperation and no policy. `scripts/scan_trajectory_agent.py` runs it on
any guidance task; press SPACE in the viewport to start. `--num_sections` sets the number
of raster columns, `--total_steps` the env steps for one pass, and `--square_size` the
half-width of the scanned square in surface grid units, centred on the target vertebra.

## Training

SonoGym's workflows in `sonogym-custom/workflows/` train SonoGym's own tasks, for example:

```bash
python sonogym-custom/workflows/skrl/train.py --task Isaac-robot-US-guidance-v0
```

They do not work with this repo's tasks yet, for two reasons: the workflows import
`spinal_surgery` but not `ultrasound_project`, so `-cam-v0` and `-phantom-v0` are never
registered there; and the tasks here register only `env_cfg_entry_point`, while
`train.py` also looks up an agent config (`skrl_cfg_entry_point`).

## What goes in the fork

`sonogym-custom/` tracks [a fork](https://github.com/Zeng-Charles/SonoGym) that carries only
fixes to files that already exist upstream:

- a batching bug in `HumanFrameViewer` that broke `num_envs == 1`;
- `torch.load` without `map_location`, which broke CPU-only loading;
- an import-time side effect in `convert_stl_to_urdf.py`;
- two run-time defaults.

Every new task, controller, tool, script, config and asset lives in this repo. Touch the
fork only when the change is to a file that already exists upstream.

## Lung phantom

`assets/patients/lung_phantom/` is a simulation body made from a CT of the FDA thorax
phantom with the scripts in `tools/`. Its [README](../assets/patients/lung_phantom/README.md)
lists the files and the steps to regenerate them.

## Inspecting the ultrasound network

`tools/test_us_network_standalone.py` runs the pix2pix CT -> ultrasound network on its own,
without Isaac Sim, to see what it renders before wiring up a simulation. One slice gives a
PNG; `--sweep` walks the volume and gives an animated GIF (or `.mp4`/`.avi`). The output is
always the CT input on the left and the simulated ultrasound on the right.

```bash
CT=assets/patients/lung_phantom/raw/ct.nii.gz

python tools/test_us_network_standalone.py --ct $CT --slice_index 210 --output us.png        # one slice
python tools/test_us_network_standalone.py --ct $CT --sweep --slice_step 4 --output us.gif   # every 4th slice
python tools/test_us_network_standalone.py --ct $CT --sweep \
    --slice_start 150 --slice_end 300 --fps 15 --output us.mp4                               # a slice range
```

`--slice_axis {0,1,2}` picks the slicing axis (default 2, axial), `--ct_range LOW HIGH`
overrides the HU window fed to the network without editing `us_generative_cfg.yaml`, and
`--device cuda` moves it off the CPU.
