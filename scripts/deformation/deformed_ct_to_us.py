"""Warp SonoGym's CT by a recorded FEM press and image it with SonoGym's ultrasound simulators.

Stage 2 of the soft body -> deformed CT -> ultrasound check. It reads the recording of
``patient_press_record.py`` and runs without Isaac Sim.

Every recorded frame is imaged where the probe actually is: USSlicer's plane, advanced
along the probe axis to the recorded face position. Each pixel is pulled back through the
deformed FEM mesh to the material point that now sits there, and CT, labels and the
model-based simulator's speckle maps are read at that point, so the image shows the
deformed body and its speckle moves with the tissue. Two images serve as "no deformation":
SonoGym's own (same pose, undeformed CT, probe face on the skin point) and the undeformed
CT sliced where the probe face actually is, i.e. a rigid body the probe sinks into.

Outside the body label the CT is set to air. The patient was scanned supine, so the CT
has the table and a mattress a centimetre behind the back; in the prone scene nothing is
there, and a probe indentation must pull back onto air, not onto the mattress.

The last frame is also warped in 3D over a box around the probe and written as NIfTI on
the CT's own grid, with the pull-back displacement and the volume ratio.

Checks, all written to ``metrics.json``:

- FEM: force-indentation curve and steadiness against a flat-punch estimate; element
  volume ratios; PhysX's deformation gradients against the recorded node positions;
- field: zero load maps to identity; points pushed forward and pulled back return to
  where they started; the dense field's volume ratio matches the elements'; how much of
  the images and the CT falls where PhysX's voxel mesh is inverted, overlaps itself or
  has holes;
- deformed CT: bone stays put; the skin is where the FEM moved it; each tissue class
  changes volume the way the FEM says it should;
- ultrasound: zero load reproduces SonoGym's image; bone depth moves with the probe;
  speckle tracking between images recovers the FEM displacement.

``make_report.py`` turns the outputs into the HTML report.

.. code-block:: bash

    python scripts/deformation/deformed_ct_to_us.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from skimage.feature import match_template

from ultrasound_project import PROJECT_DIR
from ultrasound_project.deformation.ct_warp import PullbackField, VoxelBox, pullback, submesh
from ultrasound_project.deformation.field import (
    TetrahedralInverseMapper,
    boundary_faces,
    slice_tetrahedra,
    tetrahedron_volume_ratios,
)
from ultrasound_project.sensors.sonogym_us import (
    MaterialSpeckle,
    SonoGymUltrasound,
    UltrasoundPlane,
    ct_to_sonogym_scale,
    sample_volume,
)
from ultrasound_project.utils.image_metrics import accumulate, block_match, compare, parabolic_offset
from ultrasound_project.utils.sonogym_patient import BONE_LABEL, LABEL_RES, ProbePose, SceneFrame, SonoGymPatient

VOXEL_MM = LABEL_RES * 1e3
SKIN_HU = -500.0
"""Air/skin threshold; outside the body the CT is -1000 HU after masking."""


def parse_args():
    parser = argparse.ArgumentParser(description="Deformed CT -> SonoGym ultrasound, with validation.")
    parser.add_argument("--input", default="outputs/deformation_us/fem_press.npz", help="Stage-1 recording.")
    parser.add_argument("--output_dir", default="outputs/deformation_us", help="Where results go.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decay_mm", type=float, default=15.0, help="Fade length of the displacement outside the deformed mesh."
    )
    parser.add_argument("--nearest_nodes", type=int, default=12, help="Nodes whose tetrahedra are searched per point.")
    parser.add_argument(
        "--box_mm",
        type=float,
        nargs=6,
        default=[-60.0, 60.0, -45.0, 45.0, -15.0, 110.0],
        help="Lateral, elevation and depth extent of the 3D warp box, in mm from the skin point.",
    )
    parser.add_argument("--batch", type=int, default=16, help="Images rendered per network call.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the speckle maps.")
    return parser.parse_args()


class Press:
    """The stage-1 recording, with every position moved into CT voxel coordinates."""

    def __init__(self, path: Path):
        data = np.load(path)
        self.meta = json.loads(str(data["meta"]))
        self.pose = ProbePose.from_dict(self.meta["probe_pose"])
        self.scene = SceneFrame.from_dict(self.meta["scene_frame"])
        self.elements = data["sim_elements"].astype(np.int64)
        self.rest = self.voxels(data["sim_rest"])
        self.frames = np.stack([self.voxels(p) for p in data["sim_frames"]])
        self.collision_elements = data["collision_elements"].astype(np.int64)
        self.collision_rest = self.voxels(data["collision_rest"])
        self.collision_last = self.voxels(data["collision_frames"][-1])
        self.time = data["frame_time"]
        self.force = data["frame_force"]
        self.advance = data["frame_advance"].astype(np.float64)
        self.penetration = data["frame_penetration"][:, 0]
        self.steps = {k: data[f"step_{k}"] for k in ("time", "force", "advance")}
        # PhysX's own per-element quantities after the last step, in the scene frame
        self.physx_gradients = data["deformation_gradients"].astype(np.float64)
        self.physx_stresses = data["stresses"].astype(np.float64)
        self.scene_rest = data["sim_rest"].astype(np.float64)
        self.scene_last = data["sim_frames"][-1].astype(np.float64)

    def voxels(self, scene_points: np.ndarray) -> np.ndarray:
        return (self.scene.to_human(scene_points.astype(np.float64)) / LABEL_RES).astype(np.float32)

    def plateaus(self) -> list[tuple[str, float, int]]:
        """(phase, force, index of its last frame) for the settle phase and every hold."""
        out, end = [], 0.0
        for label, duration, _, force in self.meta["phases"]:
            end += duration
            if label == "settle" or label.startswith("hold"):
                out.append((label, float(force), int(np.nonzero(self.time <= end + 1e-6)[0][-1])))
        return out


def probe_local(pose: ProbePose, voxels):
    """Voxel coordinates -> (lateral, elevation, depth) in mm from the skin point."""
    return (np.asarray(voxels, dtype=np.float64) * LABEL_RES - pose.skin) @ pose.axes * 1e3


def from_probe_local(pose: ProbePose, local_mm: np.ndarray, device) -> torch.Tensor:
    """(lateral, elevation, depth) in mm from the skin point -> voxel coordinates."""
    return torch.as_tensor((pose.skin + local_mm * 1e-3 @ pose.axes.T) / LABEL_RES, device=device, dtype=torch.float32)


def probe_box_corners(pose: ProbePose, box_mm) -> np.ndarray:
    """The 8 corners of a probe-aligned box, in voxel coordinates."""
    ranges = np.asarray(box_mm).reshape(3, 2) * 1e-3
    corners = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    return (pose.skin + corners @ pose.axes.T) / LABEL_RES


def element_gradients(elements, rest, current):
    """Deformation gradient F = dx/dX of every tetrahedron, from node positions."""
    def edges(pos):
        corners = pos[elements]
        return np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))

    return edges(current) @ np.linalg.inv(edges(rest))


def von_mises(stress: np.ndarray) -> np.ndarray:
    deviator = stress - np.trace(stress, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    return np.sqrt(1.5 * np.sum(deviator**2, axis=(1, 2)))


def crossing(profile: np.ndarray, positions: np.ndarray, threshold: float) -> float:
    """First position where a sampled profile rises through ``threshold``, interpolated."""
    above = np.nonzero(profile > threshold)[0]
    if len(above) == 0 or above[0] == 0:
        return float("nan")
    i = above[0]
    t = (threshold - profile[i - 1]) / (profile[i] - profile[i - 1])
    return float(positions[i - 1] + t * (positions[i] - positions[i - 1]))


def bone_rows(labels: np.ndarray) -> np.ndarray:
    """Row of the first bone pixel in every column of a (depth, width) label image, -1 if none."""
    bone = labels == BONE_LABEL
    return np.where(bone.any(axis=0), bone.argmax(axis=0), -1)


def bone_echo_shift(reference: np.ndarray, image: np.ndarray, rows: np.ndarray, half: int, search: int) -> np.ndarray:
    """Per-column depth shift, in rows, of the reference's bone echo in ``image``.

    The template is the reference column within ``half`` rows of the bone surface -- the
    bright echo and the shadow under it -- matched by normalised cross-correlation within
    ``search`` rows. The network repaints the soft tissue above the bone with texture as
    bright as the bone echo, so the brightest pixel near the bone is not a usable marker.
    """
    shifts = np.full(reference.shape[1], np.nan)
    for col in np.nonzero(rows >= 0)[0]:
        r = int(rows[col])
        if r - half - search < 0 or r + half + search >= reference.shape[0]:
            continue
        template = reference[r - half : r + half + 1, col].astype(np.float64)
        window = image[r - half - search : r + half + search + 1, col].astype(np.float64)
        if template.std() < 1e-9 or window.std() < 1e-9:
            continue
        scores = match_template(window[:, None], template[:, None])[:, 0]
        peak = int(np.argmax(scores))
        shifts[col] = peak - search + parabolic_offset(scores, peak)
    return shifts


def pack_polygons(polygons: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.cumsum([0] + [len(p) for p in polygons])
    points = np.concatenate(polygons) if polygons else np.zeros((0, 2))
    return points.astype(np.float32), offsets.astype(np.int64)


def main():
    args = parse_args()
    root = Path(PROJECT_DIR)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()

    press = Press(root / args.input)
    pose = press.pose
    meta = press.meta
    patient = SonoGymPatient(meta["patient"])
    axes_t = torch.as_tensor(pose.axes, device=device, dtype=torch.float32)
    metrics: dict = {
        "input": str(args.input),
        "probe": {"surface_xz": [int(v) for v in pose.surface_xz], "x_angle": float(pose.x_angle)},
    }

    hu, affine = patient.load_ct()
    raw = patient.load_labels(convert=False)
    labels = patient.load_labels()
    hu_body = np.where(raw > 0, hu, -1000.0).astype(np.float32)
    hu_t = torch.as_tensor(hu_body, device=device)
    ct = torch.as_tensor(ct_to_sonogym_scale(hu_body), device=device)
    labels_t = torch.as_tensor(labels.astype(np.float32), device=device)

    # ---------------------------------------------------------------- mesh around the probe
    corners = probe_box_corners(pose, args.box_mm)
    margin = 20.0 / VOXEL_MM
    nodes, local_elements, element_ids = submesh(
        press.elements, press.rest, corners.min(0) - margin, corners.max(0) + margin
    )
    rest = torch.as_tensor(press.rest[nodes], device=device)
    frames = torch.as_tensor(press.frames[:, nodes], device=device)
    elements = torch.as_tensor(local_elements, device=device)
    mapper = TetrahedralInverseMapper(elements, len(nodes), nearest_nodes=args.nearest_nodes)
    last = frames[-1]
    ratios_t = tetrahedron_volume_ratios(elements, rest, last)
    ratios = ratios_t.cpu().numpy()
    print(f"[MESH] {len(nodes)} of {len(press.rest)} nodes, {len(local_elements)} tetrahedra around the probe")

    # ---------------------------------------------------------------- FEM
    plateaus = meta["plateaus"]
    force = np.array([p["force_n"] for p in plateaus])
    indentation = np.array([p["indentation_mm"] for p in plateaus])
    width_m, elevation_m = meta["probe_size_m"][:2]
    radius = np.sqrt(width_m * elevation_m / np.pi)
    youngs, poissons = meta["material"]["youngs_pa"], meta["material"]["poissons"]
    flat_punch_mm = force * (1 - poissons**2) / (2 * radius * youngs) * 1e3
    upper = force >= 10.0
    slope, intercept = np.polyfit(force[upper], indentation[upper], 1)
    residual = indentation[upper] - (slope * force[upper] + intercept)
    r2 = 1 - np.sum(residual**2) / np.sum((indentation[upper] - indentation[upper].mean()) ** 2)

    centroids = probe_local(pose, last[elements].mean(dim=1).cpu().numpy())
    under = (np.abs(centroids[:, 0]) < width_m * 1e3 / 2) & (np.abs(centroids[:, 1]) < 15) & (centroids[:, 2] < 45)
    # PhysX's F after the last step against F from the last recorded node positions
    ours = element_gradients(press.elements[element_ids], press.scene_rest, press.scene_last)
    physx = press.physx_gradients[element_ids]
    layouts = {"F": physx, "F^T": np.transpose(physx, (0, 2, 1))}
    errors = {name: np.linalg.norm(value - ours, axis=(1, 2)) for name, value in layouts.items()}
    layout = min(errors, key=lambda name: np.median(errors[name]))
    metrics["fem"] = {
        "nodes": int(len(press.rest)),
        "tetrahedra": int(len(press.elements)),
        "force_n": force.tolist(),
        "indentation_mm": indentation.tolist(),
        "drift_last_0p5s_mm": [p["drift_last_0p5s_mm"] for p in plateaus],
        "step_jitter_mm": [p["step_jitter_mm"] for p in plateaus],
        "max_penetration_mm": [p["max_penetration_mm"] for p in plateaus],
        "max_pinned_drift_mm": meta["pinning"]["max_pinned_drift_mm"],
        "compliance_mm_per_n": float(slope),
        "stiffness_n_per_mm": float(1.0 / slope),
        "linear_fit_r2_10_to_20n": float(r2),
        "equivalent_halfspace_modulus_pa": float((1 - poissons**2) / (2 * radius * slope * 1e-3)),
        "flat_punch_halfspace_mm": flat_punch_mm.tolist(),
        "contact_offset_mm": meta["contact_advance_m"] * 1e3,
        "final_face_mm": float(press.advance[-1] * 1e3),
        "volume_ratio": {
            "min": float(ratios.min()),
            "p1": float(np.percentile(ratios, 1)),
            "median": float(np.median(ratios)),
            "inverted": int((ratios <= 0).sum()),
            "below_0p5": int((ratios < 0.5).sum()),
            "under_probe_median": float(np.median(ratios[under])),
        },
        "physx_gradient_vs_nodes": {
            "elements": int(len(element_ids)),
            "layout": layout,
            "median_frobenius": float(np.median(errors[layout])),
            "p99_frobenius": float(np.percentile(errors[layout], 99)),
            "median_frobenius_other_layout": float(np.median(errors["F" if layout == "F^T" else "F^T"])),
        },
    }
    print(f"[FEM] plateaus {dict(zip(force.tolist(), np.round(indentation, 2).tolist()))} mm")

    # ---------------------------------------------------------------- field
    rng = np.random.default_rng(args.seed)
    picks = torch.as_tensor(rng.choice(np.nonzero(ratios > 0)[0], 20000), device=device)
    weights = torch.as_tensor(rng.dirichlet(np.ones(4), 20000), device=device, dtype=torch.float32)
    material_true = (weights[..., None] * rest[elements[picks]]).sum(dim=1)
    deformed = (weights[..., None] * last[elements[picks]]).sum(dim=1)
    recovered, found = mapper.material_coords(deformed, rest, last)
    round_trip = (recovered - material_true).norm(dim=-1).cpu().numpy() * VOXEL_MM
    identity, _ = mapper.material_coords(material_true, rest, rest)

    def in_body(voxels: np.ndarray) -> np.ndarray:
        index = np.clip(np.round(voxels).astype(int), 0, np.asarray(raw.shape) - 1)
        return raw[tuple(index.T)] > 0

    # PhysX's voxel mesh is not a partition of space: near the skin some elements overlap
    # already at rest. A point there, or where elements overlap next to an inverted one, has
    # two material origins, and the round trip can come back through the other one.
    missed = round_trip > 0.01
    missed_points = material_true[torch.as_tensor(missed, device=device)].cpu().numpy()
    overlap_rest = (mapper.count_containing(material_true, rest) >= 2).cpu().numpy()
    overlap_now = (mapper.count_containing(deformed, last, ratios_t > 0) >= 2).cpu().numpy()
    near_inverted = torch.isin(elements[picks], elements[ratios_t <= 0].flatten()).any(dim=1).cpu().numpy()
    body_points = in_body(material_true.cpu().numpy())
    distorted = np.nonzero(ratios < 0.5)[0]
    distorted_nodes = rest[elements[distorted]].reshape(-1, 3).cpu().numpy()
    metrics["field"] = {
        "pullback_decay_mm": args.decay_mm,
        "round_trip_points": 20000,
        "round_trip_found": float(found.float().mean()),
        "round_trip_max_mm": float(round_trip.max()),
        "round_trip_p99_mm": float(np.percentile(round_trip, 99)),
        "rest_overlap_fraction": float(overlap_rest.mean()),
        "rest_overlap_fraction_in_body": float(overlap_rest[body_points].mean()),
        "deformed_overlap_fraction": float(overlap_now.mean()),
        "round_trip_over_0p01mm": int(missed.sum()),
        "round_trip_over_0p01mm_in_overlap_or_inverted": float((overlap_rest | overlap_now | near_inverted)[missed].mean())
        if missed.any()
        else 1.0,
        "round_trip_over_0p01mm_rest_depth_mm": [
            float(probe_local(pose, missed_points)[:, 2].min()),
            float(probe_local(pose, missed_points)[:, 2].max()),
        ]
        if missed.any()
        else [],
        "round_trip_over_0p01mm_in_body": float(in_body(missed_points).mean()) if missed.any() else 0.0,
        "zero_load_max_mm": float((identity - material_true).norm(dim=-1).max() * VOXEL_MM),
        "distorted_elements": int(len(distorted)),
        "distorted_rest_depth_mm": [
            float(probe_local(pose, distorted_nodes)[:, 2].min()),
            float(probe_local(pose, distorted_nodes)[:, 2].max()),
        ],
        "distorted_nodes_outside_body": float(1.0 - in_body(distorted_nodes).mean()),
    }

    # ---------------------------------------------------------------- images, frame by frame
    plane = UltrasoundPlane.from_sonogym()
    res_mm = plane.resolution * 1e3
    us = SonoGymUltrasound(device, seed=args.seed)
    speckle = MaterialSpeckle.around_plane(plane.points(pose), 0.02, plane.resolution, device, seed=args.seed)

    def plane_voxels(advance: float) -> torch.Tensor:
        return torch.as_tensor(plane.points(pose, advance) / LABEL_RES, device=device, dtype=torch.float32)

    def inputs(material: torch.Tensor):
        """CT, labels and speckle maps read at material points (W, H, 3); depth-down."""
        t0, t1 = speckle.sample(material * LABEL_RES)
        return (
            sample_volume(ct, material).T,
            sample_volume(labels_t, material, "nearest").round().T,
            t0.T,
            t1.T,
        )

    def render(batch: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
        cts, labs, t0, t1 = (torch.stack(column) for column in zip(*batch))
        return us.render_net(cts).cpu().numpy(), us.render_conv(labs, cts, t0, t1).cpu().numpy()

    reference_voxels = plane_voxels(0.0)
    reference = inputs(reference_voxels)
    ref_net = us.render_net(reference[0][None], calibrate=True)[0].cpu().numpy()
    ref_conv = render([reference])[1][0]
    unmasked = sample_volume(torch.as_tensor(ct_to_sonogym_scale(hu), device=device), reference_voxels).T
    unmasked_net = us.render_net(unmasked[None])[0].cpu().numpy()
    zero_material, _ = pullback(reference_voxels, (res_mm, res_mm), mapper, rest, rest, args.decay_mm)
    zero_net = us.render_net(inputs(zero_material)[0][None])[0].cpu().numpy()

    # every frame twice: the deformed body, and the rigid one the probe sinks into
    # (undeformed CT sliced where the face is -- SonoGym's patient pushed the same way)
    count = len(press.time)
    seq = {k: [] for k in ("ct", "label", "net", "conv", "displacement", "inside")}
    seq.update({f"rigid_{k}": [] for k in ("ct", "label", "net", "conv")})
    for start in range(0, count, args.batch):
        deformed_inputs, rigid_inputs = [], []
        for f in range(start, min(start + args.batch, count)):
            points = plane_voxels(float(press.advance[f]))
            material, inside = pullback(points, (res_mm, res_mm), mapper, rest, frames[f], args.decay_mm)
            deformed_inputs.append(inputs(material))
            rigid_inputs.append(inputs(points))
            seq["displacement"].append((((points - material) * VOXEL_MM) @ axes_t).transpose(0, 1).cpu().numpy())
            seq["inside"].append(inside.T.cpu().numpy())
        net, conv = render(deformed_inputs + rigid_inputs)
        n = len(deformed_inputs)
        for prefix, batch, rows in (("", deformed_inputs, slice(0, n)), ("rigid_", rigid_inputs, slice(n, None))):
            seq[f"{prefix}ct"] += [b[0].cpu().numpy() for b in batch]
            seq[f"{prefix}label"] += [b[1].cpu().numpy() for b in batch]
            seq[f"{prefix}net"] += list(net[rows])
            seq[f"{prefix}conv"] += list(conv[rows])
    seq = {k: np.stack(v) for k, v in seq.items()}
    print(f"[US] {count} frames rendered with SonoGym's network and model-based simulator, deformed and rigid")

    plateau_frames = press.plateaus()
    plateau_index = [f for _, _, f in plateau_frames]
    rigid_net, rigid_conv = seq["rigid_net"][plateau_index], seq["rigid_conv"][plateau_index]
    rigid_label = seq["rigid_label"][plateau_index]

    # ---------------------------------------------------------------- ultrasound checks
    windows = {
        "net": np.percentile(ref_net, [0.5, 99.5]).tolist(),
        "conv": np.percentile(ref_conv, [0.5, 99.5]).tolist(),
    }

    def scaled(kind: str, image: np.ndarray) -> np.ndarray:
        low, high = windows[kind]
        return (image - low) / (high - low)

    ref_rows = bone_rows(reference[1].cpu().numpy())
    ref_images = {"net": ref_net, "conv": ref_conv}
    us_metrics = {
        "display_window": windows,
        "image_top_below_probe_face_mm": float(plane.offset * 1e3),
        "zero_load_net_max_abs": float(np.abs(zero_net - ref_net).max()),
        "masking_net_max_abs": float(np.abs(unmasked_net - ref_net).max()),
        "plateaus": [],
    }
    for k, (label, force_n, f) in enumerate(plateau_frames):
        rows = bone_rows(seq["label"][f])
        both = (rows >= 0) & (ref_rows >= 0)
        shift = (rows[both] - ref_rows[both]) * res_mm
        echo = {}
        for kind in ("net", "conv"):
            columns = bone_echo_shift(ref_images[kind], seq[kind][f], ref_rows, int(5 / res_mm), int(8 / res_mm)) * res_mm
            echo[kind] = {
                "median_mm": float(np.nanmedian(columns)),
                "mad_mm": float(np.nanmedian(np.abs(columns - np.nanmedian(columns)))),
                "columns": int(np.isfinite(columns).sum()),
            }
        tissue = (seq["label"][f] > 0) | (rigid_label[k] > 0)
        us_metrics["plateaus"].append(
            {
                "phase": label,
                "force_n": force_n,
                "frame": f,
                "face_mm": float(press.advance[f] * 1e3),
                "bone_shift_expected_mm": float(-press.advance[f] * 1e3),
                "bone_shift_labels_mm": float(np.mean(shift)),
                "bone_shift_labels_std_mm": float(np.std(shift)),
                "bone_echo_shift_net": echo["net"],
                "bone_echo_shift_conv": echo["conv"],
                "label_changed_vs_rigid": float((seq["label"][f] != rigid_label[k])[tissue].mean()),
                "net_vs_reference": compare(scaled("net", ref_net), scaled("net", seq["net"][f]), 1.0),
                "conv_vs_reference": compare(scaled("conv", ref_conv), scaled("conv", seq["conv"][f]), 1.0),
                "net_vs_rigid": compare(scaled("net", rigid_net[k]), scaled("net", seq["net"][f]), 1.0),
                "conv_vs_rigid": compare(scaled("conv", rigid_conv[k]), scaled("conv", seq["conv"][f]), 1.0),
            }
        )
    final = plateau_frames[-1][2]

    # how much of the final image goes through badly distorted or inverted elements
    points = plane_voxels(float(press.advance[final])).reshape(-1, 3)
    tet_px, _, found_px = mapper.locate(points, last, ratios_t > 0)
    ratio_px = torch.where(found_px, ratios_t[tet_px.clamp_min(0)], torch.nan).reshape(plane.width, plane.depth).T
    affected = (~torch.isfinite(ratio_px)) | (ratio_px < 0.5)
    affected_rows = torch.nonzero(affected.any(dim=1)).flatten()
    us_metrics["final_image_distortion"] = {
        "pixels_unmapped": int((~torch.isfinite(ratio_px)).sum()),
        "pixels_in_elements_below_0p5": int((ratio_px < 0.5).sum()),
        "pixels_in_overlapping_elements": int((mapper.count_containing(points, last, ratios_t > 0) >= 2).sum()),
        "fraction": float(affected.float().mean()),
        "deepest_mm_below_probe_face": float(plane.offset * 1e3 + affected_rows.max() * res_mm)
        if len(affected_rows)
        else 0.0,
    }

    # speckle tracking between loads, against the FEM's motion
    def fem_image_motion(a: int, b: int) -> np.ndarray:
        """Where the tissue under each pixel of frame a appears in frame b: (depth, width, 2) mm."""
        points = plane_voxels(float(press.advance[a]))
        material, _ = pullback(points, (res_mm, res_mm), mapper, rest, frames[a], args.decay_mm)
        moved, _ = mapper.spatial_coords(material.reshape(-1, 3), rest, frames[b])
        origin = pose.skin + (plane.offset + press.advance[b]) * pose.normal
        local = (moved.cpu().numpy().astype(np.float64) * LABEL_RES - origin) @ pose.axes * 1e3
        local = local.reshape(plane.width, plane.depth, 3)
        lateral = (np.arange(plane.width) - plane.width // 2)[:, None] * res_mm
        depth = np.arange(plane.depth)[None, :] * res_mm
        return np.stack([local[..., 2] - depth, local[..., 0] - lateral], axis=-1).transpose(1, 0, 2)

    def tracking_errors(measured: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> dict:
        """Measured against FEM block motion (R, C, 2) mm, (depth, lateral), over ``valid`` blocks."""
        error = measured[valid] - predicted[valid]
        return {
            "valid_blocks": int(valid.sum()),
            "axial_rmse_mm": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "axial_median_abs_mm": float(np.median(np.abs(error[:, 0]))),
            "axial_bias_mm": float(np.mean(error[:, 0])),
            "axial_pearson_r": float(np.corrcoef(measured[valid][:, 0], predicted[valid][:, 0])[0, 1]),
            "lateral_rmse_mm": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "predicted_axial_mm": [float(predicted[valid][:, 0].min()), float(predicted[valid][:, 0].max())],
        }

    pairs = [(plateau_frames[i], plateau_frames[i + 1]) for i in range(len(plateau_frames) - 1)]
    pairs.append((plateau_frames[0], plateau_frames[-1]))
    us_metrics["speckle_tracking"] = []
    matches, motions, valid = {}, {}, {}
    for (_, force_a, a), (_, force_b, b) in pairs:
        motions[a, b] = fem_image_motion(a, b)
        entry = {"from_n": force_a, "to_n": force_b}
        for kind in ("conv", "net"):
            bm = matches[kind, a, b] = block_match(seq[kind][a], seq[kind][b])
            tissue = seq["label"][a][bm.rows][:, bm.cols] > 0
            valid[kind, a, b] = tissue & (bm.correlation > 0.6)
            entry[kind] = {
                "blocks": int(tissue.sum()),
                "mean_correlation": float(np.nanmean(bm.correlation[tissue])),
                **tracking_errors(bm.displacement * res_mm, motions[a, b][bm.rows][:, bm.cols], valid[kind, a, b]),
            }
        us_metrics["speckle_tracking"].append(entry)

    # the whole press once more, chaining the steps instead of matching across it at once
    first = plateau_frames[0][2]
    whole_match = matches["conv", first, final]
    chained = accumulate([matches["conv", a, b] for (_, _, a), (_, _, b) in pairs[:-1]]) * res_mm
    whole_predicted = motions[first, final][whole_match.rows][:, whole_match.cols]
    whole_tissue = seq["label"][first][whole_match.rows][:, whole_match.cols] > 0
    us_metrics["speckle_tracking_chained"] = {
        "from_n": plateau_frames[0][1],
        "to_n": plateau_frames[-1][1],
        "conv": tracking_errors(chained, whole_predicted, whole_tissue & np.isfinite(chained[..., 0])),
    }
    metrics["ultrasound"] = us_metrics
    print(
        "[US] speckle tracking (conv) axial RMSE per step:",
        [round(e["conv"]["axial_rmse_mm"], 3) for e in us_metrics["speckle_tracking"]],
        f"chained {us_metrics['speckle_tracking_chained']['conv']['axial_rmse_mm']:.3f}",
    )

    # ---------------------------------------------------------------- 3D warp of the last frame
    box = VoxelBox.around(corners, 0.0, hu.shape)
    field = PullbackField.build(box, mapper, rest, last, VOXEL_MM, args.decay_mm)
    hu_deformed = field.warp(hu_t)
    raw_deformed = field.warp(torch.as_tensor(raw.astype(np.float32), device=device), mode="nearest").round()
    labels_deformed = field.warp(labels_t, mode="nearest").round()
    jacobian = field.jacobian_determinant()
    volume_ratio = 1.0 / jacobian
    zero_field = PullbackField.build(box, mapper, rest, rest, VOXEL_MM, args.decay_mm)

    box_affine = affine.copy()
    box_affine[:3, 3] = affine[:3, :3] @ box.start + affine[:3, 3]
    nib.save(nib.Nifti1Image(np.round(hu_deformed.cpu().numpy()).astype(np.int16), affine), out / "deformed_ct.nii.gz")
    nib.save(nib.Nifti1Image(raw_deformed.cpu().numpy().astype(np.int16), affine), out / "deformed_labels.nii.gz")
    nib.save(nib.Nifti1Image(hu_body.astype(np.int16), affine), out / "ct_body.nii.gz")
    nib.save(
        nib.Nifti1Image((field.displacement * VOXEL_MM).cpu().numpy().astype(np.float32), box_affine),
        out / "pullback_displacement_mm.nii.gz",
    )
    nib.save(nib.Nifti1Image(volume_ratio.cpu().numpy().astype(np.float32), box_affine), out / "volume_ratio.nii.gz")

    # dense field against the elements it came from, away from the mesh boundary where the
    # finite differences would reach outside
    box_points = box.coords(device).reshape(-1, 3)
    tet_now, _, found_now = mapper.locate(box_points, last, ratios_t > 0)
    interior = ndimage.binary_erosion(field.inside.cpu().numpy(), iterations=2)
    inner = found_now & torch.as_tensor(interior, device=device).reshape(-1)
    dense = volume_ratio.reshape(-1)[inner]
    tissue_after = labels_deformed[box.slices].reshape(-1) > 0
    ratio_now = torch.where(found_now, ratios_t[tet_now.clamp_min(0)], torch.nan)
    # PhysX voxelises the SonoGym body surface, which has inner shells around bowel gas
    # (background in the label map); the mesh has holes there and the field is only the
    # faded extension
    _, _, covered = mapper.locate(box_points, rest)
    holes = (labels_t[box.slices].reshape(-1) > 0) & ~covered
    hole_distance = np.linalg.norm(probe_local(pose, box_points[holes].cpu().numpy()), axis=1)
    metrics["field"].update(
        {
            "box_voxels": [int(n) for n in box.shape],
            "tissue_voxels": int(tissue_after.sum()),
            "tissue_voxels_in_elements_below_0p5": int((tissue_after & (ratio_now < 0.5)).sum()),
            "tissue_voxels_in_mesh_holes": int(holes.sum()),
            "mesh_holes_min_distance_mm": float(hole_distance.min()) if len(hole_distance) else float("nan"),
            "mesh_holes_max_displacement_mm": float(field.displacement.reshape(-1, 3)[holes].norm(dim=-1).max() * VOXEL_MM)
            if bool(holes.any())
            else 0.0,
            "tissue_voxels_folded": int((tissue_after & ~found_now & covered).sum()),
            "tissue_voxels_in_overlapping_elements": int(
                (tissue_after & (mapper.count_containing(box_points, last, ratios_t > 0) >= 2)).sum()
            ),
            "dense_volume_ratio_vs_elements_median_abs": float((dense - ratios_t[tet_now[inner]]).abs().median()),
            "dense_volume_ratio_p0p1": float(torch.quantile(dense, 0.001)),
            "folded_voxels": int((jacobian.reshape(-1)[inner] <= 0).sum()),
            "zero_load_3d_max_mm": float(zero_field.displacement.abs().max() * VOXEL_MM),
            "zero_load_3d_ct_max_hu": float((zero_field.warp(hu_t) - hu_t).abs().max()),
            "max_tissue_displacement_mm": float(field.displacement.reshape(-1, 3)[tissue_after].norm(dim=-1).max() * VOXEL_MM),
        }
    )

    # each class's deformed volume in the box: the material that lands in the box, weighted
    # by its element's volume ratio, against what the warped labels hold
    grown = VoxelBox.around(corners, 10.0, hu.shape)
    material_points = grown.coords(device).reshape(-1, 3)
    moved, _ = mapper.spatial_coords(material_points, rest, last)
    lower = torch.as_tensor(box.start - 0.5, device=device)
    upper_edge = torch.as_tensor(box.start + box.shape - 0.5, device=device)
    lands = ((moved >= lower) & (moved < upper_edge)).all(dim=-1)
    tet_rest, _, found_rest = mapper.locate(material_points, rest)
    material_ratio = torch.where(found_rest, ratios_t[tet_rest.clamp_min(0)], 1.0)
    material_labels = labels_t[grown.slices].reshape(-1)
    labels_before = labels_t[box.slices].reshape(-1)
    labels_after = labels_deformed[box.slices].reshape(-1)
    voxel_ml = VOXEL_MM**3 / 1e3
    classes = []
    for value in torch.unique(labels_before).tolist():
        if value == 0 or int((labels_before == value).sum()) < 500:
            continue
        predicted = float(material_ratio[(material_labels == value) & lands].sum()) * voxel_ml
        measured = float((labels_after == value).sum()) * voxel_ml
        classes.append(
            {
                "class": int(value),
                "before_ml": float((labels_before == value).sum()) * voxel_ml,
                "after_measured_ml": measured,
                "after_predicted_ml": predicted,
                "relative_error": (measured - predicted) / predicted,
            }
        )
    bone_motion = field.displacement.reshape(-1, 3)[labels_after == BONE_LABEL].norm(dim=-1) * VOXEL_MM

    # the probe axis: skin and lamina before and after, read off the NIfTI-gridded volume and
    # by pulling the axis itself back (no resampling onto voxels in between)
    s = np.arange(-15.0, 60.0, 0.1)
    beam = from_probe_local(pose, s[:, None] * np.array([0.0, 0.0, 1.0]), device)
    beam_material, _ = pullback(beam, (0.1,), mapper, rest, last, args.decay_mm)
    hu_beam_before = sample_volume(hu_t, beam).cpu().numpy()
    hu_beam_grid = sample_volume(hu_deformed, beam).cpu().numpy()
    hu_beam_direct = sample_volume(hu_t, beam_material).cpu().numpy()
    bone_before = sample_volume(labels_t, beam, "nearest").cpu().numpy() == BONE_LABEL
    bone_after = sample_volume(labels_t, beam_material, "nearest").cpu().numpy() == BONE_LABEL
    skin_before = crossing(hu_beam_before, s, SKIN_HU)
    skin_grid = crossing(hu_beam_grid, s, SKIN_HU)
    skin_direct = crossing(hu_beam_direct, s, SKIN_HU)
    lamina_before = float(s[np.argmax(bone_before)]) if bone_before.any() else float("nan")
    lamina_after = float(s[np.argmax(bone_after)]) if bone_after.any() else float("nan")
    metrics["deformed_ct"] = {
        "classes": classes,
        "bone_displacement_mm": {
            "median": float(bone_motion.median()),
            "p99": float(torch.quantile(bone_motion, 0.99)),
            "max": float(bone_motion.max()),
        },
        "probe_axis": {
            "face_mm": float(press.advance[-1] * 1e3),
            "skin_before_mm": skin_before,
            "skin_after_direct_mm": skin_direct,
            "skin_after_nifti_grid_mm": skin_grid,
            "lamina_before_mm": lamina_before,
            "lamina_after_mm": lamina_after,
            "tissue_before_mm": lamina_before - skin_before,
            "tissue_after_mm": lamina_after - skin_direct,
            "compression_strain": (lamina_after - skin_direct) / (lamina_before - skin_before) - 1.0,
        },
    }

    # ---------------------------------------------------------------- field on two planes
    tets = local_elements
    rest_local = probe_local(pose, rest.cpu().numpy())
    last_local = probe_local(pose, last.cpu().numpy())
    gradients = element_gradients(tets, rest_local, last_local)  # probe frame: lateral, elevation, depth
    strain = 0.5 * (gradients + np.transpose(gradients, (0, 2, 1))) - np.eye(3)
    stress = von_mises(press.physx_stresses[element_ids])
    visual = {}
    skin_checks = {}
    planes = {
        "sagittal": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (-60.0, 60.0)),
        "transverse": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), (-45.0, 45.0)),
    }
    for name, (u_axis, v_axis, s_range) in planes.items():
        su = np.arange(s_range[0], s_range[1] + 1e-6, 0.5)
        tv = np.arange(-15.0, 100.0 + 1e-6, 0.5)
        shape = (len(su), len(tv))
        grid_local = (su[:, None, None] * u_axis + tv[None, :, None] * v_axis).reshape(-1, 3)
        grid = from_probe_local(pose, grid_local, device)
        # forward field on the material plane
        tet, _, found_grid = mapper.locate(grid, rest)
        moved, _ = mapper.spatial_coords(grid, rest, last)
        forward = probe_local(pose, moved.cpu().numpy()) - grid_local
        tet_np, ok = tet.cpu().numpy(), found_grid.cpu().numpy()

        def per_point(values):
            return np.where(ok, values[np.maximum(tet_np, 0)], np.nan).reshape(shape).T

        # deformed CT on the same spatial plane
        material, _ = pullback(grid.reshape(*shape, 3), (0.5, 0.5), mapper, rest, last, args.decay_mm)
        hu_before_plane = sample_volume(hu_t, grid).reshape(shape).cpu().numpy()
        hu_after_plane = sample_volume(hu_t, material.reshape(-1, 3)).reshape(shape).cpu().numpy()

        # the skin: measured on the deformed CT, against the material skin surface pushed
        # forward and cut with the plane. The surface also moves across the plane, so it is
        # traced on a stack of parallel material planes and cut where it crosses this one.
        skin_material = np.array([crossing(column, tv, SKIN_HU) for column in hu_before_plane])
        skin_measured = np.array([crossing(column, tv, SKIN_HU) for column in hu_after_plane])
        across = np.cross(u_axis, v_axis)
        offsets = np.arange(-4.0, 4.0 + 1e-6, 0.25)
        stack = offsets[:, None, None] * across + grid_local[None]
        hu_stack = sample_volume(hu_t, from_probe_local(pose, stack.reshape(-1, 3), device))
        hu_stack = hu_stack.reshape(len(offsets), *shape).cpu().numpy()
        skin_stack = np.array([[crossing(column, tv, SKIN_HU) for column in layer] for layer in hu_stack])
        surface = offsets[:, None, None] * across + su[None, :, None] * u_axis + np.nan_to_num(skin_stack)[..., None] * v_axis
        moved_surface, _ = mapper.spatial_coords(from_probe_local(pose, surface.reshape(-1, 3), device), rest, last)
        moved_surface = probe_local(pose, moved_surface.cpu().numpy()).reshape(*skin_stack.shape, 3)
        off_plane = np.where(np.isfinite(skin_stack), moved_surface @ across, np.nan)
        cut_s, cut_t = [], []
        for j in range(len(su)):
            e = off_plane[:, j]
            changes = np.nonzero(np.isfinite(e[:-1]) & np.isfinite(e[1:]) & ((e[:-1] <= 0) != (e[1:] <= 0)))[0]
            if len(changes) == 0:
                continue
            i = changes[np.argmin(np.abs(offsets[changes]))]
            w = e[i] / (e[i] - e[i + 1])
            point = moved_surface[i, j] + w * (moved_surface[i + 1, j] - moved_surface[i, j])
            cut_s.append(point @ u_axis)
            cut_t.append(point @ v_axis)
        order = np.argsort(cut_s)
        skin_predicted = np.interp(su, np.asarray(cut_s)[order], np.asarray(cut_t)[order], left=np.nan, right=np.nan)
        half_probe = (width_m if name == "sagittal" else elevation_m) * 1e3 / 2
        error = skin_measured - skin_predicted
        whole = (np.abs(su) <= 50.0) & np.isfinite(error)
        footprint = (np.abs(su) <= half_probe) & np.isfinite(error)
        skin_checks[name] = {
            "rmse_mm": float(np.sqrt(np.mean(error[whole] ** 2))),
            "max_abs_mm": float(np.abs(error[whole]).max()),
            "under_probe_rmse_mm": float(np.sqrt(np.mean(error[footprint] ** 2))),
            "under_probe_half_width_mm": half_probe,
            "skin_under_probe_before_mm": [float(np.nanmin(skin_material[footprint])), float(np.nanmax(skin_material[footprint]))],
            "skin_under_probe_after_mm": [float(np.nanmin(skin_measured[footprint])), float(np.nanmax(skin_measured[footprint]))],
            "skin_off_plane_motion_max_mm": float(np.nanmax(np.abs(off_plane[np.argmin(np.abs(offsets))]))),
        }

        rest_polygons, rest_ids = slice_tetrahedra(tets, rest_local, np.zeros(3), u_axis, v_axis)
        last_polygons, last_ids = slice_tetrahedra(tets, last_local, np.zeros(3), u_axis, v_axis)
        rest_points, rest_offsets = pack_polygons(rest_polygons)
        last_points, last_offsets = pack_polygons(last_polygons)

        def pushed_line(points_mm: np.ndarray) -> np.ndarray:
            """A material line pushed forward, projected back onto the plane."""
            moved_line, _ = mapper.spatial_coords(from_probe_local(pose, points_mm, device), rest, last)
            local = probe_local(pose, moved_line.cpu().numpy())
            return np.stack([local @ u_axis, local @ v_axis], axis=-1)

        visual[name] = {
            "s": su,
            "t": tv,
            "ct_before": sample_volume(ct, grid).reshape(shape).T.cpu().numpy(),
            "ct_after": sample_volume(ct, material.reshape(-1, 3)).reshape(shape).T.cpu().numpy(),
            "forward_mm": forward.reshape(*shape, 3).transpose(1, 0, 2),
            "volume_ratio": per_point(ratios),
            "strain_depth": per_point(strain[:, 2, 2]),
            "von_mises_pa": per_point(stress),
            "skin_material": skin_material,
            "skin_measured": skin_measured,
            "skin_predicted": skin_predicted,
            "rest_polygons": rest_points,
            "rest_offsets": rest_offsets,
            "rest_values": ratios[rest_ids],
            "last_polygons": last_points,
            "last_offsets": last_offsets,
            "last_values": ratios[last_ids],
            "last_strain": strain[last_ids, 2, 2],
            "lines_along_depth": np.stack(
                [
                    pushed_line(value * u_axis + tv[:, None] * v_axis)
                    for value in np.arange(np.ceil(s_range[0] / 5) * 5, s_range[1] + 1e-6, 5.0)
                ]
            ),
            "lines_across": np.stack(
                [pushed_line(su[:, None] * u_axis + value * v_axis) for value in np.arange(0.0, 100.0 + 1e-6, 5.0)]
            ),
        }
    metrics["deformed_ct"]["skin_profile"] = skin_checks
    print(
        f"[CT] probe axis skin {skin_before:+.2f} -> {skin_direct:+.2f} mm (NIfTI grid {skin_grid:+.2f});"
        f" skin profile RMSE {skin_checks['sagittal']['rmse_mm']:.3f} / {skin_checks['transverse']['rmse_mm']:.3f} mm;"
        f" bone moved {float(bone_motion.median()):.3f} mm median"
    )

    # the imaging plane over time: the deformed CT on it, and its cut through the mesh
    # carried along with the nodes
    su, tv = visual["sagittal"]["s"], visual["sagittal"]["t"]
    u_axis, v_axis = planes["sagittal"][:2]
    plane_grid = from_probe_local(pose, (su[:, None, None] * u_axis + tv[None, :, None] * v_axis).reshape(-1, 3), device)
    plane_grid = plane_grid.reshape(len(su), len(tv), 3)
    polygons, cut = slice_tetrahedra(tets, rest_local, np.zeros(3), u_axis, v_axis)
    corner_mm, corner_offsets = pack_polygons(polygons)
    corner_points = from_probe_local(pose, corner_mm[:, :1] * u_axis + corner_mm[:, 1:] * v_axis, device)
    corner_tet, corner_weights, corner_found = mapper.locate(corner_points, rest)
    corner_nodes = elements[corner_tet.clamp_min(0)]
    cut_elements = elements[torch.as_tensor(cut, device=device)]
    animation = {k: [] for k in ("ct", "polygons", "displacement", "ratio")}
    for f in range(count):
        material, _ = pullback(plane_grid, (0.5, 0.5), mapper, rest, frames[f], args.decay_mm)
        animation["ct"].append(sample_volume(ct, material.reshape(-1, 3)).reshape(len(su), len(tv)).T.cpu().numpy())
        moved = (corner_weights[..., None] * frames[f][corner_nodes]).sum(dim=1)
        moved = torch.where(corner_found[:, None], moved, corner_points)
        moved_local = probe_local(pose, moved.cpu().numpy())
        animation["polygons"].append(np.stack([moved_local @ u_axis, moved_local @ v_axis], axis=-1))
        animation["displacement"].append((moved - corner_points).norm(dim=-1).cpu().numpy() * VOXEL_MM)
        animation["ratio"].append(tetrahedron_volume_ratios(cut_elements, rest, frames[f]).cpu().numpy())
    animation = {k: np.stack(v) for k, v in animation.items()}

    # skin patch around the probe, for a 3D view
    faces = boundary_faces(press.collision_elements, press.collision_rest)
    skin_rest = probe_local(pose, press.collision_rest)
    skin_last = probe_local(pose, press.collision_last)
    corner = skin_rest[faces]
    near = np.all((np.abs(corner[..., 0]) < 70) & (np.abs(corner[..., 1]) < 55) & (corner[..., 2] < 25), axis=1)
    used, patch = np.unique(faces[near], return_inverse=True)
    metrics["timing_s"] = time.time() - started

    # ---------------------------------------------------------------- save
    arrays = {
        "frame_time": press.time,
        "frame_force": press.force,
        "frame_advance_mm": press.advance * 1e3,
        "frame_penetration_mm": press.penetration * 1e3,
        "step_time": press.steps["time"],
        "step_force": press.steps["force"],
        "step_advance_mm": press.steps["advance"] * 1e3,
        "contact_mm": np.float32(meta["contact_advance_m"] * 1e3),
        "plateau_frames": np.array([f for _, _, f in plateau_frames]),
        "plateau_forces": np.array([force_n for _, force_n, _ in plateau_frames]),
        "ct": seq["ct"].astype(np.float16),
        "label": seq["label"].astype(np.uint8),
        "net": seq["net"].astype(np.float16),
        "conv": seq["conv"].astype(np.float16),
        "displacement_mm": seq["displacement"].astype(np.float16),
        "inside": seq["inside"],
        "ref_ct": reference[0].cpu().numpy(),
        "ref_label": reference[1].cpu().numpy().astype(np.uint8),
        "ref_net": ref_net,
        "ref_conv": ref_conv,
        "rigid_ct": seq["rigid_ct"].astype(np.float16),
        "rigid_label": seq["rigid_label"].astype(np.uint8),
        "rigid_net": seq["rigid_net"].astype(np.float16),
        "rigid_conv": seq["rigid_conv"].astype(np.float16),
        "final_ratio_px": ratio_px.cpu().numpy(),
        "bm_rows": whole_match.rows,
        "bm_cols": whole_match.cols,
        "bm_conv_mm": whole_match.displacement * res_mm,
        "bm_conv_corr": whole_match.correlation,
        "bm_conv_good": valid["conv", first, final],
        "bm_conv_chained_mm": chained,
        "bm_net_mm": matches["net", first, final].displacement * res_mm,
        "bm_net_corr": matches["net", first, final].correlation,
        "bm_net_good": valid["net", first, final],
        "bm_predicted_mm": whole_predicted,
        "beam_s_mm": s,
        "beam_hu_before": hu_beam_before,
        "beam_hu_direct": hu_beam_direct,
        "beam_hu_grid": hu_beam_grid,
        "skin_vertices_rest": skin_rest[used].astype(np.float32),
        "skin_vertices_last": skin_last[used].astype(np.float32),
        "skin_faces": patch.reshape(-1, 3).astype(np.int32),
        "extent_mm": np.array(plane.extent_mm()),
        "probe_size_mm": np.array(meta["probe_size_m"]) * 1e3,
        "anim_ct": animation["ct"].astype(np.float16),
        "anim_polygons": animation["polygons"].astype(np.float32),
        "anim_offsets": corner_offsets,
        "anim_displacement_mm": animation["displacement"].astype(np.float16),
        "anim_ratio": animation["ratio"].astype(np.float32),
    }
    for name, data in visual.items():
        for key, value in data.items():
            arrays[f"{name}_{key}"] = np.asarray(value)
    np.savez_compressed(out / "us_sequence.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[OUT] {out / 'us_sequence.npz'}, {out / 'metrics.json'}, deformed CT / labels / field as NIfTI")
    print(f"[DONE] {metrics['timing_s']:.1f} s")


if __name__ == "__main__":
    main()
