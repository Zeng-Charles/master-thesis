"""Threshold a CT volume into SonoGym's combined_label_map.nii.gz convention.

Produces a single-label segmentation (body_label, default 120, matching
lab/kinematics/surface_motion_planner.py's default and label_conversion.yaml's existing
"120: 12" entry) marking phantom-material voxels vs air/background. This is deliberately
much simpler than the per-organ TotalSegmentator-style segmentation the shipped patient
data uses -- it's sufficient because sim.us='net' (the pix2pix/UNet ultrasound renderer)
works directly off ct.nii.gz intensities and never reads per-organ label values; only the
outer-surface contour (this script's output) feeds the probe-placement/surface-normal
computation.

Run this on an already-isotropic CT (see resample_isotropic.py) so the output's voxel
grid can be used directly by marching cubes without a spacing correction.

Usage:
    python tools/segment_body_contour.py --ct ct.nii.gz --output combined_label_map.nii.gz --threshold -400
"""

import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


def segment_body_contour(ct_path: str, output_path: str, threshold: float, body_label: int) -> None:
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()

    mask = ct_data > threshold

    labeled, num_components = ndimage.label(mask)
    if num_components == 0:
        raise ValueError(f"No voxels above threshold {threshold} in {ct_path} -- check the threshold value.")
    # keep only the largest connected component to drop small noise/artifact blobs,
    # then fill any interior holes so the surface used for probe placement is solid
    component_sizes = ndimage.sum(mask, labeled, index=range(1, num_components + 1))
    largest_component = np.argmax(component_sizes) + 1
    mask = labeled == largest_component
    mask = ndimage.binary_fill_holes(mask)

    label_map = np.zeros(ct_data.shape, dtype=np.int16)
    label_map[mask] = body_label

    out_img = nib.Nifti1Image(label_map, ct_img.affine, ct_img.header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, output_path)

    print(
        f"[INFO] wrote {output_path}: {int(mask.sum())} voxels labeled {body_label} "
        f"({100 * mask.sum() / mask.size:.1f}% of volume)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ct", required=True, help="Path to ct.nii.gz (ideally already isotropic)")
    parser.add_argument("--output", required=True, help="Output path for combined_label_map.nii.gz")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Intensity threshold separating phantom material from air/background "
        "(inspect the CT's intensity histogram, e.g. in 3D Slicer, to pick this).",
    )
    parser.add_argument(
        "--body_label",
        type=int,
        default=120,
        help="Label value for the phantom material, matching SonoGym's body_label convention (default 120).",
    )
    args = parser.parse_args()
    segment_body_contour(args.ct, args.output, args.threshold, args.body_label)


if __name__ == "__main__":
    main()
