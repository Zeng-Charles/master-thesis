"""Extract a single closed outer-surface STL from a body-contour label map.

Uses marching cubes in voxel-index space (no `spacing` argument) so the mesh matches
SonoGym's existing scale=(label_res, label_res, label_res) spawn-time convention -- this
requires the label map to already be isotropic (see resample_isotropic.py), otherwise the
mesh comes out stretched/squashed along whichever axis has different real-world spacing.

Skips the original pipeline's combine_stl.py + simplify_mesh.sh (external `wrapwrap`
binary): that tool exists to shrink-wrap many overlapping per-organ meshes into one closed
shell, which isn't needed here since we only ever segment a single body-contour mask --
marching cubes on one binary mask is already a single closed surface. Minor imperfections
are cleaned up with trimesh instead.

Usage:
    python tools/mask_to_body_stl.py --label_map combined_label_map.nii.gz --output combined_wrapwrap.stl
"""

import argparse

import nibabel as nib
import numpy as np
import trimesh
from skimage import measure


def mask_to_stl(
    label_map_path: str, output_path: str, body_label: int, smooth_iterations: int, target_face_count: int
) -> None:
    label_img = nib.load(label_map_path)
    label_data = label_img.get_fdata()
    mask = (label_data == body_label).astype(np.uint8)

    if mask.sum() == 0:
        raise ValueError(f"No voxels labeled {body_label} in {label_map_path}")

    # if the segmented object touches the volume's edge (e.g. the scan's field of view
    # cuts it off, or it's simply centered near an axis limit), marching cubes has no
    # data beyond the array to close the surface there, leaving an open cap -- pad with
    # a margin of background so every boundary is guaranteed closable, then shift the
    # result back by the same margin so vertices stay in the original voxel-index frame
    pad = 4
    padded_mask = np.pad(mask, pad_width=pad, mode="constant", constant_values=0)

    # no `spacing` kwarg -- vertices stay in voxel-index coordinates, matching how
    # existing patients' meshes are authored (scaled to meters only at USD-spawn time)
    verts, faces, normals, _ = measure.marching_cubes(padded_mask, level=0.5)
    verts -= pad

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    trimesh.repair.fill_holes(mesh)
    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)

    # marching cubes on a full-resolution CT-derived mask produces a mesh with one
    # triangle per voxel-scale surface facet -- multiple millions of faces for a typical
    # scan. That's wildly more detail than a body-contour collision/visual mesh needs
    # (the whole point is just the outer shape for the probe to slide over) and is heavy
    # enough that Isaac Sim's convex-decomposition collider generation + rendering can
    # blow through GPU memory that would otherwise be fine. Decimate down to a sane count.
    if target_face_count is not None and len(mesh.faces) > target_face_count:
        pre_faces = len(mesh.faces)
        mesh = mesh.simplify_quadric_decimation(face_count=target_face_count)
        # decimation can occasionally nick the surface closed; re-run the same repair pass
        trimesh.repair.fill_holes(mesh)
        print(f"[INFO] decimated {pre_faces} -> {len(mesh.faces)} faces")

    mesh.export(output_path)
    print(
        f"[INFO] wrote {output_path}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
        f"watertight={mesh.is_watertight}"
    )
    if not mesh.is_watertight:
        print(
            "[WARNING] mesh is not watertight -- inspect it before converting to URDF/USD "
            "(open holes can break collision generation)."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label_map", required=True, help="Path to combined_label_map.nii.gz")
    parser.add_argument("--output", required=True, help="Output STL path, e.g. combined_wrapwrap.stl")
    parser.add_argument("--body_label", type=int, default=120, help="Label value to extract (default 120).")
    parser.add_argument(
        "--smooth_iterations",
        type=int,
        default=5,
        help="Laplacian smoothing iterations to soften the voxel-stairstep surface (default 5, 0 to disable).",
    )
    parser.add_argument(
        "--target_face_count",
        type=int,
        default=50000,
        help="Decimate the mesh down to about this many faces if marching cubes produces more "
        "(default 50000; a raw full-resolution CT scan can produce millions, which is far more "
        "detail than a collision/visual body-contour mesh needs and can blow through GPU memory "
        "during USD collision generation. Pass 0 to disable decimation.)",
    )
    args = parser.parse_args()
    target_face_count = args.target_face_count if args.target_face_count > 0 else None
    mask_to_stl(args.label_map, args.output, args.body_label, args.smooth_iterations, target_face_count)


if __name__ == "__main__":
    main()
