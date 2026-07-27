"""Resample a CT volume to isotropic voxel spacing.

SonoGym's existing patients all use a single scalar `label_res` to convert grid-index
coordinates to real-world meters uniformly across x/y/z (see
scale=(label_res, label_res, label_res) in the task _setup_scene() methods, and
marching_cubes being run without a `spacing` argument elsewhere in this pipeline). That
only works correctly if the underlying volume actually has the same physical size along
every axis. Real DICOM scans frequently don't (e.g. thinner in-plane pixels than slice
thickness), so this step resamples the volume onto a uniform grid before segmentation.

Usage:
    python tools/resample_isotropic.py --ct ct.nii.gz --output ct.nii.gz --target_spacing 0.703125
"""

import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


def resample_isotropic(ct_path: str, output_path: str, target_spacing_mm: float, order: int) -> None:
    img = nib.load(ct_path)
    data = img.get_fdata()
    spacing_mm = np.array(img.header.get_zooms()[:3], dtype=float)

    zoom_factors = spacing_mm / target_spacing_mm
    new_shape = np.round(np.array(data.shape) * zoom_factors).astype(int)

    resampled = ndimage.zoom(data, zoom_factors, order=order)
    # zoom's output shape can be off by a voxel or two from the naive round() above due to
    # rounding inside ndimage.zoom itself; crop/pad to the shape we actually got so the
    # affine below (built from new_shape) matches the array we save.
    new_shape = np.array(resampled.shape)

    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.sign(img.affine[:3, :3]) * target_spacing_mm
    # keep the same physical origin (translation column) as the input volume
    new_affine[:3, 3] = img.affine[:3, 3]

    # CT intensities are integer HU values -- cast back from ndimage.zoom's float64
    # output to the input's own dtype (int16 for every existing ct.nii.gz in this repo)
    # instead of saving a float64 volume, which is ~4x larger on disk for no benefit.
    orig_dtype = img.get_data_dtype()
    resampled = np.round(resampled).astype(orig_dtype)

    out_img = nib.Nifti1Image(resampled, new_affine)
    out_img.set_data_dtype(orig_dtype)
    out_img.header.set_zooms((target_spacing_mm, target_spacing_mm, target_spacing_mm))
    nib.save(out_img, output_path)

    print(f"[INFO] input shape={data.shape}, spacing={tuple(spacing_mm)} mm")
    print(f"[INFO] wrote {output_path}: shape={tuple(new_shape)}, spacing=({target_spacing_mm},)*3 mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ct", required=True, help="Path to the (possibly anisotropic) ct.nii.gz")
    parser.add_argument("--output", required=True, help="Output path for the resampled, isotropic ct.nii.gz")
    parser.add_argument(
        "--target_spacing",
        type=float,
        required=True,
        help="Target isotropic voxel spacing in mm (e.g. the finer of the input's per-axis spacings, "
        "to avoid losing detail by downsampling).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Spline interpolation order for scipy.ndimage.zoom (default 3 = cubic).",
    )
    args = parser.parse_args()
    resample_isotropic(args.ct, args.output, args.target_spacing, args.order)


if __name__ == "__main__":
    main()
