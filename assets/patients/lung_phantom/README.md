# lung_phantom

Simulation body for `Isaac-robot-US-guidance-phantom-v0`. Tracked in Git LFS, so it comes
with the clone — no download step.

## Source dataset

Derived from **LUNG-PHANTOM** on The Cancer Imaging Archive:
<https://www.cancerimagingarchive.net/collection/lung-phantom/>

A CT of the FDA anthropomorphic thorax phantom (64-detector GE LightSpeed VCT, 120 kVp /
100 mAs, 1.25 mm slices, 237 images, one subject), released under CC BY 3.0. The
collection's 12 synthetic lesions are irrelevant here: only the phantom's outer body
contour is used, as a surface for the probe to sit against. TCIA's data usage policy
requires this citation:

> Zhao, B. (2015). *Lung Phantom* (Version 2) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/k9/tcia.2015.08a1ixoo>

The original DICOM series is not in git; it sits in `data/raw/Lung-Phantom/` with a NIfTI
conversion alongside, and can be re-downloaded from the collection page.

## Layout

The three sub-directories mirror SonoGym's `selected_dataset_body_from_urdf` /
`selected_dataset_stl` / `selected_dataset` split, flattened per patient. The file names
are fixed: `USSlicer`, `SurfaceMotionPlanner` and `VertebraViewer` all build their paths
by appending these exact names to the directories they are handed.

Every binary below is an LFS object; the `.urdf`, `config.yaml` and this README are
ordinary git objects.

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
