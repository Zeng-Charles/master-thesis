# Soft box press demo

`scripts/deformation/soft_body_press_demo.py` is the smallest PhysX soft-body check in this
repo: a deformable box stands in for the patient, a rigid cylinder presses into it with a
prescribed contact force, and the indentation is measured. It can also export the FEM
displacement as a dense field. The scene is built in code (ground plane, dome light, one
FEM box, one rigid cylinder); nothing is loaded from SonoGym.

## Running

Soft bodies are GPU-only in PhysX (CUDA is the default device). Run unbuffered: Isaac Sim's
`close()` hard-exits and drops buffered stdout.

```bash
python -u scripts/deformation/soft_body_press_demo.py                               # viewport
python -u scripts/deformation/soft_body_press_demo.py --headless --force 5 10 20 40 # force sweep
python -u scripts/deformation/soft_body_press_demo.py --headless --capture          # PNGs + GIF
```

```text
[INFO] settled surface at z = 117.2 mm (indentation zero)
[RESULT]   20.0 N  ->   11.90 mm indentation  (contact stiffness    1681 N/m)
```

`--capture` writes its frames to `outputs/soft_body_press/`.

## How it works

`--force` is the contact force, not a position target: the script applies `force - m*g` as
an external wrench, so the probe's weight is carried and `--force` is what reaches the
tissue. It ramps up over `--ramp` seconds. Between levels the probe parks 2 mm above the
surface so the tissue springs back. The probe's x/y and orientation are rewritten every
step (a vertical 1-DoF gantry) because a slender cylinder tips over on a compliant
surface; `--free_probe` gives it all 6 DoF.

Indentation is the deepest drop of the top-surface FEM nodes under the probe, measured
against their height *after* settling, so the box's own sag is not counted.

The box is spawned from a subdivided mesh: Isaac Lab's `MeshCuboidCfg` has only 8
vertices, and a render mesh with 8 corners cannot show a dent however well the FEM mesh
underneath deforms.

## Exporting the displacement field

`--export_field` writes the FEM displacement to `outputs/soft_body_press/fields/` at the
steady state of each press:

```bash
python -u scripts/deformation/soft_body_press_demo.py --headless --export_field --force 10 20 40
```

```text
[RESULT]   20.0 N  ->   12.15 mm indentation  (contact stiffness    1646 N/m)
[FIELD] outputs/soft_body_press/fields/field_020N.npz
        grid 91x61x33 @ 5 mm, keyed by rest position, zero at the settled shape
        |u|max = 12.06 mm (compare the indentation above), round-trip vs nodes: mean 0.008 mm, max 0.288 mm
```

The simulation nodes are a few thousand scattered points; warping a CT needs a displacement
that can be evaluated anywhere, so `ultrasound_project.deformation.field` scatters them onto
a regular grid (modified Shepard weights, exact at the nodes) that trilinear interpolation
can read:

```python
from ultrasound_project.deformation.field import DeformationField

field = DeformationField.load("outputs/soft_body_press/fields/field_020N.npz")
material = field.material_coords(sample_points)   # deformed position -> where to read the CT
```

The [deformation pipeline](deformation_pipeline.md) maps through the tetrahedral mesh
instead (`TetrahedralInverseMapper`), which is exact inside the mesh and needs no grid.

Two numbers to check every time. `|u|max` should match the indentation printed just above
it; they are measured along completely different paths. The round-trip error is the field
re-evaluated at the nodes it was built from, so it must stay far below the CT voxel size
(0.7 mm for the lung phantom).

Two conventions that are easy to get wrong:

- `--field_reference` (default `settled`) sets the zero of the displacement. A CT is
  acquired with the patient already lying down, so the settled shape is what its voxels
  correspond to; referencing the FEM rest shape instead folds the body's own gravity sag
  into a field that is supposed to describe only what the probe did. It shows up
  immediately in `|u|max`: 14.7 mm against the rest shape versus 12.1 mm of actual press.
- `--field_frame` (default `rest`) sets which configuration indexes the grid. Keyed by rest
  position it is in the CT's own frame and inverts by fixed-point iteration; keyed by
  deformed position it inverts in a single lookup.

## Options

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
| `--capture` / `--capture_dir` / `--capture_every` | off / `outputs/soft_body_press` / `0.1` | Frames to disk |
| `--export_field` / `--field_dir` / `--field_spacing` | off / `outputs/soft_body_press/fields` / `0.005` | Displacement field to disk |
| `--field_reference` / `--field_frame` | `settled` / `rest` | Zero of the displacement / which configuration indexes the grid |

`--hex_res` is the simulation mesh and sets cost and accuracy: cells are ~17 mm by
default, so a 25 mm probe spans only a couple of elements and the response is stiffer than
the analytical flat-punch solution. `--surface_res` is only the render mesh and is
subdivided in powers of two (0.02 -> 9216 triangles, 0.01 -> 36864).
