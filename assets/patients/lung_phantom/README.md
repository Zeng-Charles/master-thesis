# lung_phantom

Simulation body for `Isaac-robot-US-guidance-phantom-v0`, derived from a CT scan of a
physical lung phantom. **None of the files below are in git** — three of them exceed
GitHub's 100 MB per-file limit and this repo does not use Git LFS. Regenerate them with
the pipeline at the bottom, or copy them from a machine that already has them.

## Layout

The three sub-directories mirror SonoGym's `selected_dataset_body_from_urdf` /
`selected_dataset_stl` / `selected_dataset` split, flattened per patient. The file names
are fixed: `USSlicer`, `SurfaceMotionPlanner` and `VertebraViewer` all build their paths
by appending these exact names to the directories they are handed.

```
raw/
  ct.nii.gz                      113 MB  isotropic CT, what the 'net' US renderer reads
  ct_orig_anisotropic.nii.gz      60 MB  pre-resampling original, kept for reference only
stl/
  combined_label_map.nii.gz      3.3 MB  single-label (120) body-contour mask
  combined_wrapwrap.stl          274 MB  closed outer surface mesh
  combined_wrapwrap.urdf         4.0 KB  URDF wrapper around the STL
  phantom_target.stl              64 KB  placeholder scan target (centroid is the goal)
  standard_right_traj_et.stl     4.0 KB  placeholder trajectory mesh
  body_lowest_y_array.pt         1.7 MB  surface cache, auto-generated on first run
  body_surface_normal_array.pt   2.5 MB  surface cache, auto-generated on first run
usd/
  combined_wrapwrap/             395 MB  Isaac-Sim-ready body, spawned by the task
```

The two `.pt` files are caches: `SurfaceMotionPlanner` writes them the first time the task
runs and reuses them afterwards, so they can be deleted safely.

## Regenerating from the raw scan

Source data is the DICOM series in `data/raw/Lung-Phantom/` (converted to NIfTI in
`data/raw/Lung-Phantom-NIfTI/`). `SONOGYM=sonogym-custom` below.

```bash
# 1. resample onto an isotropic grid -- everything downstream assumes a single scalar
#    label_res, and the raw scan's slice thickness differs from its in-plane spacing
python tools/resample_isotropic.py \
    --ct data/raw/Lung-Phantom-NIfTI/5.1_CHEST_NON_CONTRAST_6.nii.gz \
    --output assets/patients/lung_phantom/raw/ct.nii.gz \
    --target_spacing 0.703125

# 2. threshold into a single-label body-contour mask
python tools/segment_body_contour.py \
    --ct assets/patients/lung_phantom/raw/ct.nii.gz \
    --output assets/patients/lung_phantom/stl/combined_label_map.nii.gz \
    --threshold -400

# 3. marching cubes -> one closed surface mesh
python tools/mask_to_body_stl.py \
    --label_map assets/patients/lung_phantom/stl/combined_label_map.nii.gz \
    --output assets/patients/lung_phantom/stl/combined_wrapwrap.stl

# 4. placeholder target + trajectory meshes for VertebraViewer
python tools/make_placeholder_target_traj.py \
    --center_voxel 256 256 120 \
    --target_out assets/patients/lung_phantom/stl/phantom_target.stl \
    --traj_out assets/patients/lung_phantom/stl/standard_right_traj_et.stl

# 5. STL -> URDF -> USD, using the SonoGym submodule's converters
python $SONOGYM/tools/convert_stl_to_urdf.py    # edit its __main__ paths first
bash   $SONOGYM/tools/convert_body_urdf_usd_folder.sh
```

`label_res` in `configs/env/robotic_US_guidance_phantom.yaml` (0.000703125 m) must match
the `--target_spacing` used in step 1, converted from millimetres to metres.
