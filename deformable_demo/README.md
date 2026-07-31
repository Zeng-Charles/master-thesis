# deformable_demo

Standalone Isaac Sim soft-body check, independent of SonoGym and of `src/`: a deformable
box on the ground stands in for the patient, a rigid cylinder presses into it with a given
force, and the indentation is measured.

`soft_body_press_demo.py` builds the scene in code. `cube_demo.usd` is an unrelated stage
saved from the GUI; the script does not read it.

## Running

Soft bodies are GPU-only in PhysX (CUDA is the default device). Use the Isaac Lab conda
env, and run unbuffered -- Isaac Sim's `close()` hard-exits and swallows buffered stdout:

```bash
conda activate isaaclab_v2.1

python -u deformable_demo/soft_body_press_demo.py                               # viewport
python -u deformable_demo/soft_body_press_demo.py --headless --force 5 10 20 40 # sweep
python -u deformable_demo/soft_body_press_demo.py --headless --capture          # PNGs + GIF
```

```text
[INFO] settled surface at z = 117.2 mm (indentation zero)
[RESULT]   20.0 N  ->   11.90 mm indentation  (contact stiffness    1681 N/m)
```

## How it works

`--force` is the contact force, not a position target: the script applies `force - m*g` as
an external wrench, so the probe's weight is carried and `--force` is what reaches the
tissue. It ramps up over `--ramp` seconds. Between levels the probe parks 2 mm above the
surface so the tissue springs back. The probe's x/y and orientation are rewritten every
step (a vertical 1-DoF gantry) because a slender cylinder tips over on a compliant
surface; `--free_probe` gives it all 6 DoF.

Indentation is the deepest drop of the top-surface FEM nodes under the probe, measured
against their height *after* settling, so the box's own sag is not counted.


## Knobs

| Flag | Default | Meaning |
| --- | --- | --- |
| `--force` | `20` | Contact force(s) in N, held one after another |
| `--hold` / `--release` / `--settle` / `--ramp` | `3` / `1.5` / `1` / `1` | Phase timing in s |
| `--cycles` | `0` | Repetitions; 0 = forever (GUI) or one pass (headless) |
| `--body_size` | `0.4 0.25 0.12` | Box size in m |
| `--youngs` / `--poissons` / `--density` | `2e4` / `0.45` / `1000` | Tissue material |
| `--hex_res` | `24` | FEM mesh resolution along the longest edge |
| `--surface_res` | `0.02` | Longest render-mesh triangle edge in m |
| `--probe_radius` / `--probe_height` / `--probe_mass` | `0.025` / `0.12` / `0.5` | Cylinder |
| `--no_nodes` / `--free_probe` | off | Drop the heat map / the gantry constraint |
| `--capture` / `--capture_dir` / `--capture_every` | off / `outputs/deformable_demo` / `0.1` | Frames to disk |

`--hex_res` is the simulation mesh and sets cost and accuracy: cells are ~17 mm by
default, so a 25 mm probe spans only a couple of elements and the response is stiffer than
the analytical flat-punch solution. `--surface_res` is only the render mesh and is
subdivided in powers of two (0.02 -> 9216 triangles, 0.01 -> 36864).
