# Deformation pipeline: soft tissue -> deformed CT -> ultrasound

A probe presses into SonoGym's patient `s0010` in a PhysX FEM simulation, the patient's CT
is warped by the simulated tissue motion, and both of SonoGym's ultrasound simulators image
the deformed CT. No robot is involved. The commands are in the
[README](../README.md#deformation-pipeline); this page covers how each stage works and what
the checks found.

| Stage | Script in `scripts/deformation/` | Isaac Sim | Writes to `outputs/deformation_us/` |
| --- | --- | --- | --- |
| 1. FEM press | `patient_press_record.py` | yes | `fem_press.npz`, `fem_press_render.mp4` |
| 2. Deformed CT and ultrasound | `deformed_ct_to_us.py` | no | `us_sequence.npz`, `metrics.json`, NIfTI volumes |
| 3. Report | `make_report.py` | no | `report/index.html` |

## Stage 1: FEM press

`patient_press_record.py` turns SonoGym's outer surface mesh (`combined_wrapwrap.stl`) into
one PhysX FEM soft body and lays it prone, back up.

- **Material and mesh:** E = 35 kPa, nu = 0.45, `--hex_res 48`.
- **Support:** gravity is off. Simulation nodes inside bone (dilated by 3 mm) and within
  20 mm of the bed are pinned. The FEM reference configuration therefore stays identical
  to the CT, with no gravity sag to subtract, and the vertebrae stay rigid while the tissue
  over them compresses, which a single PhysX material could not do on its own.
- **Probe:** a rigid 58 x 11 x 40 mm box, the footprint of SonoGym's linear transducer,
  placed at the pose SonoGym commands for surface cell (123, 164) with `x_angle = pi/2`, a
  paramedian sagittal view at L3. An external force pushes it along its own axis, on which
  it is held like a 1-DoF gantry, so once it stops moving the contact force equals the
  commanded one.
- **Load:** 5, 10, 15 and 20 N, each ramped over 0.5 s and held for 2 s. All simulation
  nodes are recorded at 20 Hz; the last level is the state stage 2 warps the CT by.
- **Solver** of the recorded run: 5 ms time step, 120 iterations, 40 000 surface faces.

The site was chosen because the skin under the 58 x 11 mm face is flat to within 0.6 mm and
45 mm of erector spinae lie over the laminae. The guidance task's transverse L4 goal is a
poor place to press a flat face: it sits in the midline furrow, the face lands on the two
paraspinal ridges 10-19 mm above the skin point, and the ridges collapse under it.
Collision nodes that end up inside the probe are counted every frame
(`frame_penetration`), so a failed contact shows up in the log.

## Stage 2: deformed CT and ultrasound

`deformed_ct_to_us.py` works from the stage-1 recording and needs no Isaac Sim. All
geometry is computed in CT voxel units (one voxel is 1.5 mm for SonoGym's patients).

1. **Image plane.** Each recorded frame is imaged where the probe actually is: SonoGym's
   `USSlicer` plane, advanced along the probe axis to the recorded face position.
2. **Pull-back.** Each pixel is mapped back through the deformed tetrahedral mesh to the
   material point that now sits there, by barycentric coordinates
   (`TetrahedralInverseMapper`); elements the deformation has turned inside out are
   skipped. Outside the mesh the displacement is extended with a Gaussian fade over 15 mm
   (`--decay_mm`), so the dent reads as air rather than as the tissue that used to be there.
3. **Sampling.** CT, labels and the model-based simulator's speckle maps are read at the
   material point, so the image shows the deformed body and the speckle moves with the
   tissue. Outside the body label the CT is set to air: the patient was scanned supine with
   a table and mattress behind the back, and in the prone scene an indentation must pull
   back onto air.
4. **Rendering.** Each frame goes through SonoGym's pix2pix network and its model-based
   simulator. Two images serve as "no deformation": SonoGym's own image at the same pose,
   and the undeformed CT sliced where the probe face actually is, i.e. a rigid body the
   probe sinks into.
5. **3D volume.** The last frame is also warped in 3D over a box around the probe and
   saved as NIfTI on the CT's grid, with the pull-back displacement and the volume ratio.

### Ultrasound simulators outside Isaac Sim

`ultrasound_project.sensors.sonogym_us` runs SonoGym's own simulator classes, configs and
weights without Isaac Lab. `import spinal_surgery` would register SonoGym's tasks and pull
in all of Isaac Lab, so the simulator modules are loaded by file path with only
`PACKAGE_DIR`/`PROJECT_DIR` stubbed. The module re-implements the part of `USSlicer` that
turns a probe pose into simulator input:

- **Plane:** 150 x 200 px at 0.5 mm, width along the probe's lateral axis, depth along the
  inward normal, starting 3 mm below the skin point.
- **CT preprocessing:** HU clipped to the guidance task's `ct_range` [-300, 3000] and scaled
  to 0..255; the network then clips that to its own `CT.range` [0, 150].
- **Network input:** laid out as (lateral, depth), as the environment feeds it; a
  transposed input produces streaks.
- **Model-based simulator:** run as the guidance task runs it, with `if_use_ct=True`
  (impedance from CT, the other acoustic parameters from the converted label map) and no
  added noise.

One difference is deliberate. SonoGym truncates voxel coordinates to integers when
slicing, which quantises the plane to 1.5 mm and would erase sub-voxel tissue motion. Here
the CT is sampled trilinearly and the labels by nearest neighbour.

## Stage 3: report

`make_report.py` builds one self-contained HTML page (in Chinese) with the animations, the
displacement field, the image comparisons and every check. Figures, both animations and
the Isaac Sim render are embedded, so the page can be shared on its own; the PNG and MP4
files are kept next to it.

### Optional: solver sensitivity sweep

If `outputs/deformation_us/sensitivity/` exists, the report adds a sweep over mesh
resolution and solver iterations. To produce it (nine FEM runs):

```bash
for res in 40 48 56; do for it in 60 120 240; do
  python -u scripts/deformation/patient_press_record.py --headless --hex_res $res --iterations $it \
      --force 10 20 --settle 0.2 --hold 1.5 --record_hz 2 \
      --output outputs/deformation_us/sensitivity/res${res}_it${it}.npz
done; done
```

## Outputs

| File in `outputs/deformation_us/` | Content |
| --- | --- |
| `fem_press.npz` | simulation and collision meshes over time, deformation gradients and stresses, probe force and advance, pinned nodes, run settings |
| `fem_press_render.mp4` | Isaac Sim render of the press (`--capture`) |
| `us_sequence.npz` | per-frame network and model-based images, CT and labels for the deformed body, the rigid body and SonoGym's reference; displacement; speckle tracking |
| `metrics.json` | every check below |
| `deformed_ct.nii.gz`, `deformed_labels.nii.gz` | CT and labels warped by the last frame |
| `ct_body.nii.gz` | undeformed CT, air outside the body |
| `pullback_displacement_mm.nii.gz`, `volume_ratio.nii.gz` | 3D pull-back displacement and volume ratio |
| `report/` | `index.html` and the figures and videos it embeds |
| `sensitivity/` | optional solver sweep |

## Checks

`metrics.json` records:

- **FEM:** the force-indentation curve and its steadiness against a flat-punch estimate;
  element volume ratios; PhysX's deformation gradients against the recorded node positions.
- **Field:** zero load maps to the identity; points pushed forward and pulled back return
  to where they started; the dense field's volume ratio matches the elements'; how much of
  the images and the CT falls where PhysX's voxel mesh is inverted, overlaps itself or has
  holes.
- **Deformed CT:** bone stays put; the skin is where the FEM moved it; each tissue class
  changes volume the way the FEM says it should.
- **Ultrasound:** zero load reproduces SonoGym's image; bone depth moves with the probe;
  speckle tracking between images recovers the FEM displacement.

Results of the current run at 20 N:

| Check | Result |
| --- | --- |
| Skin in the deformed CT vs the FEM-predicted surface | RMSE 0.013 / 0.020 mm (sagittal / transverse) |
| Bone echo shift in the model-based images vs the probe advance | within 0.07 mm |
| Speckle tracking vs FEM motion, model-based simulator | RMSE 0.07-0.10 mm per step, 0.19 mm chained over 0-20 N |
| Zero displacement through the whole chain vs SonoGym's own network image | max pixel difference 0.0013 |

## Known limitations

- **The FEM solver is not converged** at 120 iterations: the 20 N indentation drops 30-42%
  between 60 and 240 iterations and grows with mesh refinement (volumetric locking). What
  is validated is the chain, not the tissue stiffness; the sweep above quantifies this.
- **Surface mesh vs CT skin:** SonoGym's surface mesh lies about 2.7 mm outside the CT's
  -500 HU skin. The probe touches that shell first, so at 20 N the CT skin sits at the top
  of the image (3 mm below the probe face) and the model-based simulator draws a bright
  skin line there.
- **Voxel mesh defects:** PhysX's voxel mesh overlaps itself over ~1% of its volume near
  the skin and has holes at bowel gas, which SonoGym's labels leave as background. Both are
  measured in the report; none of the final image falls in them.
- **Network simulator:** it downsamples its input to 64 x 64, so bone echoes move in
  ~1.6 mm steps, and it regenerates the speckle every frame instead of moving it with the
  tissue.

## Code

| Module | Contents |
| --- | --- |
| `ultrasound_project.deformation.field` | `TetrahedralInverseMapper` (pull-back through the tetrahedral mesh); `DeformationField`, the grid field used by the [box demo](soft_body_press.md); mesh helpers |
| `ultrasound_project.deformation.ct_warp` | pull-back over a voxel box with the outside-mesh extension; CT and label warping |
| `ultrasound_project.sensors.sonogym_us` | SonoGym's two ultrasound simulators outside Isaac Sim; image plane; speckle attached to the material |
| `ultrasound_project.utils.sonogym_patient` | SonoGym patient assets, probe poses, the prone scene frame |
| `ultrasound_project.utils.image_metrics` | image similarity scores; speckle tracking by block matching |

`tests/test_ct_warp.py` and `tests/test_deformation_field.py` check the geometry on small
synthetic meshes, on the CPU.
