"""Press SonoGym's patient with a fixed force and record the FEM deformation.

Stage 1 of the soft body -> deformed CT -> ultrasound check. Stage 2,
``deformed_ct_to_us.py``, needs no Isaac Sim and works from the ``.npz`` written here.

The patient is SonoGym's own outer surface (``combined_wrapwrap.stl``), made into one PhysX
FEM soft body and laid prone, back up. Gravity is off and the body is held instead: the
simulation nodes inside bone are pinned, and so are those within ``--support_band`` of the
bed. That keeps the FEM reference configuration identical to the CT -- there is no sag to
subtract -- and keeps the vertebrae rigid while the tissue over them compresses, which a
single PhysX material could not do on its own.

The probe is a rigid box with the footprint of SonoGym's linear transducer, placed at the
pose SonoGym commands for ``--target_xz``/``--x_angle`` and pushed along its own axis by an
external force. It is held on that axis like a 1-DoF gantry, so once it stops moving the
contact force equals the commanded one. The levels in ``--force`` are held one after
another; the last is the state stage 2 warps the CT by.

The default site is a paramedian sagittal view at L3 (cell (123, 164), long axis along the
spine), where the skin under the 58 x 11 mm face is flat to within 0.6 mm and 45 mm of
erector spinae lie over the laminae. The guidance task's transverse L4 goal is a poor place
to press a flat face: it sits in the midline furrow, the face lands on the two paraspinal
ridges 10-19 mm above the skin point, and the narrow ridges collapse under it. Collision
nodes that end up inside the probe are counted every frame (``frame_penetration``), so a
contact that fails this way shows up in the log.

.. code-block:: bash

    python -u scripts/deformation/patient_press_record.py --headless
    python -u scripts/deformation/patient_press_record.py --headless --capture   # + rendered video
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Press SonoGym's patient with a fixed probe force and record the FEM.")
parser.add_argument("--patient", default="s0010", help="SonoGym patient id.")
parser.add_argument("--target_xz", type=int, nargs=2, default=[123, 164], help="SonoGym surface cell (x, z).")
parser.add_argument(
    "--x_angle", type=float, default=math.pi / 2, help="Image-plane angle, SonoGym's third command entry."
)
parser.add_argument(
    "--force", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0], help="Force levels in N, held in turn."
)
parser.add_argument("--settle", type=float, default=0.3, help="Seconds before the first force ramp.")
parser.add_argument("--ramp", type=float, default=0.5, help="Seconds to ramp from one level to the next.")
parser.add_argument("--hold", type=float, default=2.0, help="Seconds each level is held.")
# -- soft body
parser.add_argument("--hex_res", type=int, default=48, help="Voxels of the FEM mesh along the body's longest edge.")
parser.add_argument("--max_faces", type=int, default=40_000, help="Faces of the surface mesh after decimation.")
parser.add_argument("--youngs", type=float, default=3.5e4, help="Young's modulus of the tissue in Pa.")
parser.add_argument("--poissons", type=float, default=0.45, help="Poisson's ratio of the tissue.")
parser.add_argument("--density", type=float, default=1000.0, help="Tissue density in kg/m^3.")
# PhysX's FEM solver is iterative: the response still stiffens between 60 and 120
# iterations, and with 30 the probe sank straight through a hex_res 72 mesh.
parser.add_argument("--iterations", type=int, default=120, help="PhysX solver position iterations.")
parser.add_argument("--damping", type=float, default=0.05, help="Vertex velocity damping.")
parser.add_argument("--elasticity_damping", type=float, default=0.05, help="Material damping.")
parser.add_argument("--no_pin_bones", action="store_true", help="Let the bones deform with the tissue.")
parser.add_argument("--bone_dilate_mm", type=float, default=3.0, help="Grow the bone mask before pinning nodes.")
parser.add_argument("--support_band", type=float, default=0.02, help="Pin nodes this close to the bed, in m.")
# -- probe
parser.add_argument(
    "--probe_size",
    type=float,
    nargs=3,
    default=[0.058, 0.011, 0.04],
    help="Lateral, elevation and axial size of the probe head in m (SonoGym's linear.stl face is 58 x 11 mm).",
)
parser.add_argument("--probe_mass", type=float, default=0.1, help="Probe mass in kg.")
parser.add_argument("--friction", type=float, default=0.1, help="Probe-skin friction (gel-coupled, so low).")
parser.add_argument("--gap", type=float, default=0.001, help="Initial gap between probe and skin in m.")
# -- output
parser.add_argument("--record_hz", type=float, default=20.0, help="Frames per second written for stage 2.")
parser.add_argument("--output", default="outputs/deformation_us/fem_press.npz", help="Where the recording goes.")
parser.add_argument("--capture", action="store_true", help="Also render the press to an mp4 next to --output.")
parser.add_argument("--capture_size", type=int, nargs=2, default=[960, 540], help="Width and height of the render.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.capture:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import ndimage
from scipy.spatial.transform import Rotation

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material

from ultrasound_project import PROJECT_DIR
from ultrasound_project.utils.sonogym_patient import (
    BONE_LABEL,
    LABEL_RES,
    SceneFrame,
    SonoGymPatient,
    sonogym_probe_pose,
)

DT = 1.0 / 200.0


def build_patient(patient: SonoGymPatient) -> tuple[SceneFrame, int]:
    """Spawns the patient's outer surface as an FEM soft body; returns the scene frame."""
    mesh = trimesh.load_mesh(patient.surface_mesh_path, process=True)
    frame = SceneFrame.prone(mesh.bounds)
    if args_cli.max_faces > 0 and len(mesh.faces) > args_cli.max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=args_cli.max_faces)
    vertices = frame.to_scene(np.asarray(mesh.vertices, dtype=np.float64) * LABEL_RES).astype(np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    prim_utils.create_prim(
        "/World/Patient",
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.reshape(-1),
            "faceVertexCounts": np.full(len(faces), 3, dtype=np.int32),
        },
    )
    schemas.define_deformable_body_properties(
        "/World/Patient",
        sim_utils.DeformableBodyPropertiesCfg(
            simulation_hexahedral_resolution=args_cli.hex_res,
            solver_position_iteration_count=args_cli.iterations,
            vertex_velocity_damping=args_cli.damping,
            contact_offset=0.001,
            rest_offset=0.0,
            self_collision=False,
        ),
    )
    material = sim_utils.DeformableBodyMaterialCfg(
        youngs_modulus=args_cli.youngs,
        poissons_ratio=args_cli.poissons,
        density=args_cli.density,
        dynamic_friction=args_cli.friction,
        elasticity_damping=args_cli.elasticity_damping,
    )
    material.func("/World/SoftMaterial", material)
    bind_physics_material("/World/Patient", "/World/SoftMaterial")
    skin = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.50, 0.42), roughness=0.55)
    skin.func("/World/SkinLook", skin)
    bind_visual_material("/World/Patient", "/World/SkinLook")
    return frame, len(faces)


def build_scene(probe_axes: np.ndarray, probe_center: np.ndarray):
    """Lights, a visual-only bed, the probe and optionally a camera."""
    bed = sim_utils.CuboidCfg(
        size=(1.2, 1.2, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.27, 0.3))
    )
    bed.func("/World/Bed", bed, translation=(0.0, 0.0, -0.012))
    dome = sim_utils.DomeLightCfg(intensity=250.0, color=(0.85, 0.87, 0.95))
    dome.func("/World/Light", dome)
    # grazing light across the spine (scene +x, 20 degrees down) to bring out the dent
    sun = sim_utils.DistantLightCfg(intensity=2000.0, angle=1.0, color=(1.0, 0.97, 0.92))
    sun.func("/World/Sun", sun, orientation=(0.8192, 0.0, -0.5736, 0.0))

    quat_xyzw = Rotation.from_matrix(probe_axes).as_quat()
    probe_cfg = RigidObjectCfg(
        prim_path="/World/Probe",
        spawn=sim_utils.CuboidCfg(
            size=tuple(args_cli.probe_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=2.0,
                angular_damping=5.0,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.probe_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.friction, dynamic_friction=args_cli.friction
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.85), metallic=0.3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(v) for v in probe_center),
            rot=(float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
        ),
    )
    probe = RigidObject(cfg=probe_cfg)

    camera = None
    if args_cli.capture:
        width, height = args_cli.capture_size
        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/CaptureCam",
                update_period=0.0,
                width=width,
                height=height,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
            )
        )
    return probe, camera


def pinned_nodes(patient: SonoGymPatient, frame: SceneFrame, rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the simulation nodes held by bone and by the bed."""
    support = rest[:, 2] < args_cli.support_band
    bone_nodes = np.zeros(len(rest), dtype=bool)
    if not args_cli.no_pin_bones:
        bone = patient.load_labels() == BONE_LABEL
        if args_cli.bone_dilate_mm > 0:
            bone = ndimage.distance_transform_edt(~bone) <= args_cli.bone_dilate_mm / (LABEL_RES * 1e3)
        voxel = np.round(frame.to_human(rest.astype(np.float64)) / LABEL_RES).astype(int)
        valid = np.all((voxel >= 0) & (voxel < np.asarray(bone.shape)), axis=1)
        bone_nodes[valid] = bone[voxel[valid, 0], voxel[valid, 1], voxel[valid, 2]]
    return bone_nodes, support


def schedule() -> list[tuple[str, float, float, float]]:
    """(label, duration, force at start, force at end) of every phase."""
    phases = [("settle", args_cli.settle, 0.0, 0.0)]
    previous = 0.0
    for level in args_cli.force:
        phases.append((f"ramp {level:g} N", args_cli.ramp, previous, level))
        phases.append((f"hold {level:g} N", args_cli.hold, level, level))
        previous = level
    return phases


def main():
    if "cuda" not in args_cli.device:
        raise SystemExit("PhysX deformable bodies are GPU-only; run with --device cuda:0")
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = Path(PROJECT_DIR) / output
    output.parent.mkdir(parents=True, exist_ok=True)

    patient = SonoGymPatient(args_cli.patient)
    target_xz = tuple(args_cli.target_xz)
    pose = sonogym_probe_pose(patient, *target_xz, x_angle=args_cli.x_angle)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device, gravity=(0.0, 0.0, 0.0)))
    frame, face_count = build_patient(patient)

    skin = frame.to_scene(pose.skin)
    axes = frame.vector_to_scene(pose.axes.T).T  # columns: lateral, elevation, inward normal
    normal = axes[:, 2]
    half_height = args_cli.probe_size[2] / 2.0
    # placeholder pose; the probe is moved onto the measured skin once the FEM mesh exists
    probe, camera = build_scene(axes, skin - (half_height + 0.05) * normal)

    eye = skin - 0.11 * normal + 0.25 * axes[:, 1] + 0.12 * axes[:, 0]
    target = skin + 0.005 * normal
    sim.set_camera_view(tuple(eye), tuple(target))
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([eye], device=sim.device, dtype=torch.float32),
            torch.tensor([target], device=sim.device, dtype=torch.float32),
        )

    soft = sim.physics_sim_view.create_soft_body_view("/World/Patient")
    if soft._backend is None:
        raise RuntimeError("PhysX failed to create the patient soft body")
    rest = soft.get_sim_nodal_positions()[0].clone()
    elements = soft.get_sim_element_indices()[0].long()
    collision_rest = soft.get_nodal_positions()[0].clone()
    collision_elements = soft.get_element_indices()[0].long()
    device = rest.device
    print(
        f"[FEM] {rest.shape[0]} simulation nodes, {elements.shape[0]} tetrahedra (hex_res {args_cli.hex_res});"
        f" collision mesh {collision_rest.shape[0]} nodes from {face_count} surface faces"
    )

    bone_nodes, support_nodes = pinned_nodes(patient, frame, rest.cpu().numpy())
    pinned = torch.as_tensor(bone_nodes | support_nodes, device=device)
    targets = soft.get_sim_kinematic_targets().clone()
    targets[0, :, :3] = rest
    targets[0, :, 3] = torch.where(pinned, 0.0, 1.0)
    body_index = torch.zeros(1, dtype=torch.long, device=device)
    soft.set_sim_kinematic_targets(targets, indices=body_index)
    print(f"[FEM] pinned {int(bone_nodes.sum())} bone nodes and {int(support_nodes.sum())} bed nodes")

    # first contact: the most outward collision node under the probe's footprint
    skin_t = torch.as_tensor(skin, device=device, dtype=torch.float32)
    axes_t = torch.as_tensor(axes, device=device, dtype=torch.float32)
    normal_t = axes_t[:, 2]
    local = (collision_rest - skin_t) @ axes_t
    lateral_half, elevation_half = args_cli.probe_size[0] / 2.0, args_cli.probe_size[1] / 2.0
    footprint = (local[:, 0].abs() <= lateral_half) & (local[:, 1].abs() <= elevation_half)
    if not bool(footprint.any()):
        raise RuntimeError("no collision node under the probe footprint")
    contact = local[footprint, 2].min().item()

    def penetration(face: float) -> tuple[float, int]:
        """Deepest collision node inside the probe head, and how many are over 0.5 mm in."""
        local = (soft.get_nodal_positions()[0] - skin_t) @ axes_t
        inside = (
            (local[:, 0].abs() <= lateral_half)
            & (local[:, 1].abs() <= elevation_half)
            & (local[:, 2] < face)
            & (local[:, 2] > face - 2.0 * half_height)
        )
        depth = face - local[inside, 2]
        return (depth.max().item() if depth.numel() else 0.0), int((depth > 5e-4).sum())

    print(
        f"[PROBE] surface cell {target_xz}, skin voxel {tuple(np.round(pose.skin / LABEL_RES).astype(int))};"
        f" first contact {contact * 1e3:+.2f} mm along the normal from the CT skin point"
    )

    quat = probe.data.root_state_w[:, 3:7].clone()
    home = probe.data.root_state_w.clone()
    home[:, :3] = skin_t + (contact - args_cli.gap - half_height) * normal_t
    home[:, 7:] = 0.0
    probe.write_root_state_to_sim(home)

    forces = torch.zeros((1, 1, 3), device=device)
    torques = torch.zeros_like(forces)
    phases = schedule()
    record_every = max(int(round(1.0 / (args_cli.record_hz * DT))), 1)
    print("[RUN]", " -> ".join(f"{label} ({duration:g}s)" for label, duration, _, _ in phases))

    times, applied, advances = [], [], []
    frames_t, frames_force, frames_advance, frames_penetration = [], [], [], []
    sim_frames, collision_frames, images = [], [], []
    plateaus = []
    step = 0
    started = time.time()
    for label, duration, force_start, force_end in phases:
        steps = max(int(round(duration / DT)), 1)
        history = []
        for phase_step in range(steps):
            force = force_start + (force_end - force_start) * (phase_step + 1) / steps
            forces[..., 2] = force
            probe.set_external_force_and_torque(forces, torques)

            # 1-DoF gantry along the probe axis: keep the axial motion, drop the rest
            state = probe.data.root_state_w.clone()
            axial = (state[:, :3] - skin_t) @ normal_t
            state[:, :3] = skin_t + axial[:, None] * normal_t
            state[:, 3:7] = quat
            state[:, 7:10] = ((state[:, 7:10] @ normal_t)[:, None]) * normal_t
            state[:, 10:13] = 0.0
            probe.write_root_pose_to_sim(state[:, :7])
            probe.write_root_velocity_to_sim(state[:, 7:])
            probe.write_data_to_sim()

            record = step % record_every == 0
            sim.step(render=record and camera is not None)
            probe.update(DT)
            step += 1

            advance = ((probe.data.root_pos_w[0] - skin_t) @ normal_t).item() + half_height
            times.append(step * DT)
            applied.append(force)
            advances.append(advance)
            history.append(advance)
            if record:
                frames_t.append(step * DT)
                frames_force.append(force)
                frames_advance.append(advance)
                frames_penetration.append(penetration(advance))
                sim_frames.append(soft.get_sim_nodal_positions()[0].cpu().numpy().astype(np.float32))
                collision_frames.append(soft.get_nodal_positions()[0].cpu().numpy().astype(np.float32))
                if camera is not None:
                    camera.update(DT)
                    images.append(camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

        # PhysX's rigid-body velocity is not the probe's motion here (the contact with the
        # soft body corrects positions), so steadiness is judged from positions alone
        window = np.asarray(history[-max(int(0.5 / DT), 1):])
        indentation = window[-10:].mean() - contact
        drift = window[-10:].mean() - window[:10].mean()
        jitter = np.diff(window).std()
        depth, count = penetration(history[-1])
        print(
            f"  [{label:>10s}] indentation {indentation * 1e3:6.2f} mm, drift over last 0.5 s {drift * 1e3:+.3f} mm,"
            f" step jitter {jitter * 1e3:.3f} mm, deepest node inside probe {depth * 1e3:.2f} mm"
            f" ({count} > 0.5 mm)"
        )
        if label.startswith("hold"):
            plateaus.append(
                {
                    "force_n": force_end,
                    "indentation_mm": indentation * 1e3,
                    "drift_last_0p5s_mm": drift * 1e3,
                    "step_jitter_mm": jitter * 1e3,
                    "max_penetration_mm": depth * 1e3,
                    "nodes_penetrating": count,
                }
            )
    wall = time.time() - started

    current = soft.get_sim_nodal_positions()[0]
    pinned_drift = (current - rest)[pinned].norm(dim=-1).max().item() * 1e3 if bool(pinned.any()) else 0.0
    deformation_gradients = soft.get_sim_element_deformation_gradients().view(-1, 3, 3)[: elements.shape[0]]
    stresses = soft.get_sim_element_stresses().view(-1, 3, 3)[: elements.shape[0]]
    print(f"[FEM] pinned nodes moved at most {pinned_drift:.4f} mm; {step} steps in {wall:.1f} s")

    meta = {
        "patient": args_cli.patient,
        "probe_pose": pose.to_dict(),
        "scene_frame": frame.to_dict(),
        "probe_size_m": list(args_cli.probe_size),
        "probe_mass_kg": args_cli.probe_mass,
        "contact_advance_m": contact,
        "gap_m": args_cli.gap,
        "forces_n": list(args_cli.force),
        "phases": [list(p) for p in phases],
        "plateaus": plateaus,
        "material": {"youngs_pa": args_cli.youngs, "poissons": args_cli.poissons, "density": args_cli.density},
        "solver": {
            "dt": DT,
            "hex_res": args_cli.hex_res,
            "iterations": args_cli.iterations,
            "vertex_velocity_damping": args_cli.damping,
            "elasticity_damping": args_cli.elasticity_damping,
            "surface_faces": face_count,
        },
        "pinning": {
            "bones": not args_cli.no_pin_bones,
            "bone_dilate_mm": args_cli.bone_dilate_mm,
            "support_band_m": args_cli.support_band,
            "bone_nodes": int(bone_nodes.sum()),
            "support_nodes": int(support_nodes.sum()),
            "max_pinned_drift_mm": pinned_drift,
        },
        "friction": args_cli.friction,
        "record_hz": 1.0 / (record_every * DT),
        "wall_time_s": wall,
    }
    np.savez_compressed(
        output,
        meta=json.dumps(meta),
        sim_rest=rest.cpu().numpy().astype(np.float32),
        sim_elements=elements.cpu().numpy().astype(np.int32),
        collision_rest=collision_rest.cpu().numpy().astype(np.float32),
        collision_elements=collision_elements.cpu().numpy().astype(np.int32),
        pinned_bone=bone_nodes,
        pinned_support=support_nodes,
        step_time=np.asarray(times, dtype=np.float32),
        step_force=np.asarray(applied, dtype=np.float32),
        step_advance=np.asarray(advances, dtype=np.float32),
        frame_time=np.asarray(frames_t, dtype=np.float32),
        frame_force=np.asarray(frames_force, dtype=np.float32),
        frame_advance=np.asarray(frames_advance, dtype=np.float32),
        frame_penetration=np.asarray(frames_penetration, dtype=np.float32),
        sim_frames=np.stack(sim_frames),
        collision_frames=np.stack(collision_frames),
        deformation_gradients=deformation_gradients.cpu().numpy().astype(np.float32),
        stresses=stresses.cpu().numpy().astype(np.float32),
    )
    print(f"[OUT] {output} ({len(sim_frames)} frames)")

    if images:
        import imageio.v2 as imageio

        video = output.with_name(output.stem + "_render.mp4")
        imageio.mimwrite(video, images, fps=1.0 / (record_every * DT), codec="libx264", quality=8, macro_block_size=8)
        print(f"[OUT] {video}")


if __name__ == "__main__":
    main()
    simulation_app.close()
