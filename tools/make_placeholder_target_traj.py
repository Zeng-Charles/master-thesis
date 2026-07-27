"""Generate placeholder target-anatomy + trajectory STLs for VertebraViewer.

VertebraViewer (lab/kinematics/vertebra_viewer.py) unconditionally loads a target-anatomy
mesh and a trajectory mesh for every patient, but only ever uses their geometry: the
target mesh's vertex centroid (the scan-goal position), and a PCA of the trajectory
mesh's vertices (direction/radius, used only by the surgery task, not guidance). Neither
needs real anatomical meaning -- these are simple placeholder primitives positioned
wherever you want the nominal scan target to be, in the same voxel-index coordinates as
the body mesh (mask_to_body_stl.py's output).

Usage:
    python tools/make_placeholder_target_traj.py \\
        --center_voxel 256 256 120 \\
        --target_out phantom_target.stl \\
        --traj_out standard_right_traj_gt.stl
"""

import argparse

import numpy as np
import trimesh


def make_placeholder_meshes(
    center_voxel,
    target_out: str,
    traj_out: str,
    target_radius: float,
    traj_length: float,
    traj_radius: float,
) -> None:
    center = np.asarray(center_voxel, dtype=float)

    target_mesh = trimesh.creation.icosphere(radius=target_radius)
    target_mesh.apply_translation(center)
    target_mesh.export(target_out)
    print(f"[INFO] wrote {target_out} (centroid == {target_mesh.centroid})")

    # any well-elongated shape gives VertebraViewer's PCA a well-defined principal axis;
    # it doesn't matter which axis this cylinder is initially built along -- PCA finds
    # the point cloud's actual spread direction and canonicalizes its sign afterward
    traj_mesh = trimesh.creation.cylinder(radius=traj_radius, height=traj_length, sections=16)
    traj_mesh.apply_translation(center)
    traj_mesh.export(traj_out)
    print(f"[INFO] wrote {traj_out} (centroid == {traj_mesh.centroid})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--center_voxel",
        type=float,
        nargs=3,
        required=True,
        help="Voxel-index (x y z) position inside the body mesh to place the placeholder target/trajectory.",
    )
    parser.add_argument("--target_out", required=True, help="Output STL path for the placeholder target-anatomy mesh.")
    parser.add_argument("--traj_out", required=True, help="Output STL path for the placeholder trajectory mesh.")
    parser.add_argument("--target_radius", type=float, default=5.0, help="Radius (voxels) of the placeholder target sphere.")
    parser.add_argument("--traj_length", type=float, default=20.0, help="Length (voxels) of the placeholder trajectory cylinder.")
    parser.add_argument("--traj_radius", type=float, default=2.0, help="Radius (voxels) of the placeholder trajectory cylinder.")
    args = parser.parse_args()
    make_placeholder_meshes(
        args.center_voxel, args.target_out, args.traj_out, args.target_radius, args.traj_length, args.traj_radius
    )


if __name__ == "__main__":
    main()
