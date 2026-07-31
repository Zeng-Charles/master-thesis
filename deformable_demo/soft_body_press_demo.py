# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal soft-body demo: a deformable cuboid on the ground stands in for the patient
phantom, and a rigid cylinder above it is pressed into the surface with a prescribed
force, so you can watch (and measure) how the tissue deforms.

The scene is built from scratch in code -- ground plane, dome light, one FEM deformable
box (:class:`DeformableObject`, PhysX soft body) and one rigid cylinder
(:class:`RigidObject`). The cylinder is driven by an external wrench, not by a position
target: ``--force`` is the *contact* force the probe should push with, and the script
applies ``force - mass * g`` as the external wrench, so the probe's weight is carried and
``--force`` is exactly what reaches the tissue. Each force level is held for ``--hold``
seconds, then released, and the resulting surface indentation is printed.

Because a slender free-floating cylinder tips over as soon as it touches a compliant
surface, the probe is constrained to a vertical 1-DoF "gantry": its x/y position and its
orientation are rewritten every step, while z stays under full physics control. Pass
``--free_probe`` to remove that constraint and let it move in all 6 DoF.

The phantom is spawned from a *subdivided* box (see :func:`spawn_dense_cuboid`): Isaac
Lab's :class:`MeshCuboidCfg` builds one with 8 vertices, and a render mesh with 8 corners
cannot show a dent no matter how well the FEM mesh underneath deforms.

Soft bodies are GPU-only in PhysX, so this needs a CUDA device (the default).

.. code-block:: bash

    # interactive, 20 N probe
    python deformable_demo/soft_body_press_demo.py

    # force-displacement sweep, stiffer tissue, no GUI
    python deformable_demo/soft_body_press_demo.py --headless --force 5 10 20 40 --youngs 3e5

    # render the press to PNGs + a GIF instead of watching the viewport
    python deformable_demo/soft_body_press_demo.py --headless --capture

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Press a rigid cylinder into a deformable phantom body.")
parser.add_argument(
    "--force",
    type=float,
    nargs="+",
    default=[20.0],
    help="Contact force(s) in N to press with. Several values are held one after another as a sweep.",
)
parser.add_argument("--hold", type=float, default=3.0, help="Seconds to hold each force level.")
parser.add_argument(
    "--ramp",
    type=float,
    default=1.0,
    help="Seconds to ramp the force up to its level, so the surface sinks visibly instead of jumping. 0 = step.",
)
parser.add_argument("--settle", type=float, default=1.0, help="Seconds to let the scene settle before pressing.")
parser.add_argument("--release", type=float, default=1.5, help="Seconds without force between two levels.")
parser.add_argument(
    "--cycles",
    type=int,
    default=0,
    help="How often to repeat the whole schedule. 0 = repeat forever (GUI) / once (headless).",
)
# -- phantom body
parser.add_argument(
    "--body_size",
    type=float,
    nargs=3,
    default=[0.4, 0.25, 0.12],
    help="Size (x, y, z) of the deformable box in m.",
)
parser.add_argument("--youngs", type=float, default=2.0e4, help="Young's modulus of the tissue in Pa.")
parser.add_argument("--poissons", type=float, default=0.45, help="Poisson's ratio of the tissue.")
parser.add_argument("--density", type=float, default=1000.0, help="Tissue density in kg/m^3.")
parser.add_argument(
    "--hex_res",
    type=int,
    default=24,
    help="Target hexahedral resolution of the FEM simulation mesh (higher = finer but slower).",
)
parser.add_argument(
    "--surface_res",
    type=float,
    default=0.02,
    help="Longest triangle edge of the phantom's render mesh in m. This is what lets the dent be rendered at all.",
)
parser.add_argument(
    "--no_nodes",
    action="store_true",
    default=False,
    help="Drop the surface-node overlay and show the bare deformed mesh.",
)
# -- probe
parser.add_argument("--probe_radius", type=float, default=0.025, help="Cylinder radius in m.")
parser.add_argument("--probe_height", type=float, default=0.12, help="Cylinder height in m.")
parser.add_argument("--probe_mass", type=float, default=0.5, help="Cylinder mass in kg.")
parser.add_argument(
    "--free_probe",
    action="store_true",
    default=False,
    help="Let the probe move in 6 DoF instead of constraining it to the vertical axis.",
)
# -- capture
parser.add_argument(
    "--capture",
    action="store_true",
    default=False,
    help="Render a close-up of the contact to PNGs (and a GIF) under --capture_dir. Works headless.",
)
parser.add_argument(
    "--capture_dir",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "deformable_demo"),
    help="Where --capture writes its frames.",
)
parser.add_argument("--capture_every", type=float, default=0.1, help="Seconds of sim time between captured frames.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# a camera sensor needs the render pipeline, which is off by default when headless
if args_cli.capture:
    args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from PIL import Image
from typing import Callable

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material
from isaaclab.utils import configclass

GRAVITY = 9.81
# physics step: soft-body contact needs a smaller step than the 1/60 s default
SIM_DT = 1.0 / 200.0
# render every n-th physics step, i.e. at 50 Hz -- rendering at 200 Hz only slows the viewport down
RENDER_EVERY = 4
# heat map for the surface-node overlay: how far a node has sunk (in m) -> colour
NODE_BANDS = (0.001, 0.004, 0.008)
NODE_COLORS = ((0.85, 0.87, 0.9), (1.0, 0.83, 0.3), (1.0, 0.5, 0.1), (0.85, 0.1, 0.1))


def spawn_dense_cuboid(
    prim_path: str,
    cfg: "DenseMeshCuboidCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawns a deformable box whose render mesh is fine enough to show a dent.

    This is the one reason not to use :class:`~isaaclab.sim.MeshCuboidCfg` here. That
    spawner builds the box with ``trimesh.creation.box``, i.e. 8 vertices and 12
    triangles. PhysX skins the render mesh to the tetrahedral simulation mesh, so with 8
    corner vertices the surface stays a flat-faced box in the viewport however deep the
    FEM mesh underneath is actually pushed in -- the deformation is simulated but
    invisible. Subdividing the box first gives the surface vertices to move.

    Everything else mirrors what Isaac Lab's mesh spawners do: the deformable schema and
    the materials go on the mesh prim, one level below the asset's Xform.
    """
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    box = trimesh.creation.box(extents=cfg.size)
    vertices, faces = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, max_edge=cfg.max_edge)
    print(f"[INFO] phantom render mesh: {len(vertices)} vertices, {len(faces)} triangles")

    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    prim_utils.create_prim(
        mesh_prim_path,
        prim_type="Mesh",
        attributes={
            "points": vertices,
            "faceVertexIndices": faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces)),
            "subdivisionScheme": "bilinear",
        },
    )

    if cfg.mass_props is not None:
        schemas.define_mass_properties(mesh_prim_path, cfg.mass_props)
    schemas.define_deformable_body_properties(mesh_prim_path, cfg.deformable_props)

    # note: visual and physics material share one prim path by default, as they do upstream
    for material_cfg, material_rel_path, bind in (
        (cfg.visual_material, cfg.visual_material_path, bind_visual_material),
        (cfg.physics_material, cfg.physics_material_path, bind_physics_material),
    ):
        if material_cfg is None:
            continue
        material_path = (
            material_rel_path if material_rel_path.startswith("/") else f"{geom_prim_path}/{material_rel_path}"
        )
        material_cfg.func(material_path, material_cfg)
        bind(mesh_prim_path, material_path)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class DenseMeshCuboidCfg(sim_utils.MeshCuboidCfg):
    """A :class:`MeshCuboidCfg` that subdivides the box before spawning it."""

    func: Callable = spawn_dense_cuboid

    max_edge: float = 0.012
    """Longest triangle edge of the render mesh (in m)."""


def design_scene() -> tuple[dict, dict]:
    """Spawns ground, light, the deformable phantom body and the rigid probe.

    Returns:
        The scene entities, and a dict of scene dimensions the run loop needs.
    """
    body_size = tuple(args_cli.body_size)
    # the box is spawned around its own centre, so lift it by half its height to rest on the floor
    body_center_z = body_size[2] / 2.0
    surface_z = body_size[2]
    # start the probe just above the surface, so the first contact is soft
    probe_center_z = surface_z + args_cli.probe_height / 2.0 + 0.002

    # ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # lighting: a dim dome for fill and a low sun for the rest. Under flat ambient light a
    # shallow dent is nearly invisible -- the shading gradient across it and the shadow the
    # probe casts into it are what make it read as a dent at all
    cfg_light = sim_utils.DomeLightCfg(intensity=600.0, color=(0.8, 0.82, 0.9))
    cfg_light.func("/World/light", cfg_light)
    cfg_sun = sim_utils.DistantLightCfg(intensity=3000.0, angle=1.0, color=(1.0, 0.97, 0.92))
    # tilt the default straight-down light to ~40 deg elevation, coming across the scene
    cfg_sun.func("/World/sun", cfg_sun, orientation=(0.8644, -0.1271, 0.4031, 0.2725))

    # -- deformable phantom body: a soft box standing in for the patient
    phantom_cfg = DeformableObjectCfg(
        prim_path="/World/Phantom",
        spawn=DenseMeshCuboidCfg(
            size=body_size,
            max_edge=args_cli.surface_res,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                rest_offset=0.0,
                contact_offset=0.001,
                simulation_hexahedral_resolution=args_cli.hex_res,
                solver_position_iteration_count=20,
                vertex_velocity_damping=0.005,
                self_collision=False,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=args_cli.youngs,
                poissons_ratio=args_cli.poissons,
                density=args_cli.density,
                dynamic_friction=0.5,
                elasticity_damping=0.01,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.6, 0.55), roughness=0.35),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, body_center_z)),
        debug_vis=False,
    )
    phantom = DeformableObject(cfg=phantom_cfg)

    # -- rigid probe: the cylinder pressed into the surface
    probe_cfg = RigidObjectCfg(
        prim_path="/World/Probe",
        spawn=sim_utils.MeshCylinderCfg(
            radius=args_cli.probe_radius,
            height=args_cli.probe_height,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # damping keeps the probe from ringing on the compliant surface, so the
                # steady-state indentation is reached within the hold window
                linear_damping=2.0,
                angular_damping=5.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                # a light probe would fall asleep between force levels
                sleep_threshold=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.probe_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(rest_offset=0.0, contact_offset=0.002),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.85), metallic=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, probe_center_z)),
    )
    probe = RigidObject(cfg=probe_cfg)

    entities = {"phantom": phantom, "probe": probe}
    dims = {"body_size": body_size, "surface_z": surface_z, "probe_center_z": probe_center_z}

    # -- capture camera: same close-up as the viewport, but written to disk
    if args_cli.capture:
        camera_cfg = CameraCfg(
            prim_path="/World/CaptureCam",
            update_period=0.0,
            height=540,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
        )
        entities["camera"] = Camera(cfg=camera_cfg)

    return entities, dims


def make_node_markers() -> VisualizationMarkers:
    """Dots on the phantom's top surface, coloured by how far they have sunk.

    Shading alone barely shows a centimetre-deep dent on a half-metre body. A grid of
    markers riding the FEM surface nodes does: the dots dip with the surface and turn from
    grey through orange to red as they are pushed down.
    """
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SurfaceNodes",
        markers={
            f"band_{idx}": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color, roughness=0.4),
            )
            for idx, color in enumerate(NODE_COLORS)
        },
    )
    return VisualizationMarkers(cfg)


def contact_view(dims: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Eye and target of a close-up on the contact point.

    A 15 mm dent under a 50 mm probe is a few pixels wide from across the room, which is
    the other half of why the deformation is easy to miss.
    """
    surface_z = dims["surface_z"]
    return (0.26, 0.21, surface_z + 0.25), (0.0, 0.0, surface_z - 0.01)


def build_schedule() -> list[tuple[str, float, float]]:
    """Builds the (label, duration, contact force) phases of one pass."""
    schedule = [("settle", args_cli.settle, 0.0)]
    for force in args_cli.force:
        schedule.append((f"press {force:g} N", args_cli.hold, force))
        schedule.append(("release", args_cli.release, 0.0))
    return schedule


def surface_node_masks(phantom: DeformableObject, dims: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Selects the FEM nodes on the top surface, and the subset of them under the probe.

    Indentation is measured on the subset only -- averaging over the whole top face would
    mix the local dent with the global sag of the box under its own weight. The full top
    face is what the marker overlay rides on.
    """
    rest_pos = phantom.data.default_nodal_state_w[0, :, :3]
    # top layer only: the hexahedral resolution is spread over the longest edge, so half
    # a cell is a band that catches the top nodes and nothing below them
    cell = max(dims["body_size"]) / max(args_cli.hex_res, 1)
    is_top = rest_pos[:, 2] > dims["surface_z"] - 0.5 * cell
    # under the probe: within twice the probe radius of the axis
    radial = torch.linalg.norm(rest_pos[:, :2], dim=-1)
    under_probe = is_top & (radial < 2.0 * args_cli.probe_radius)
    if not bool(under_probe.any()):
        # coarse mesh: fall back to the whole top layer
        under_probe = is_top
    return is_top, under_probe


def save_frames(frames: list):
    """Writes the captured frames as PNGs plus an animation of the whole press."""
    if not frames:
        return
    out_dir = os.path.abspath(args_cli.capture_dir)
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{idx:04d}.png"))
    gif_path = os.path.join(out_dir, "press.gif")
    imageio.mimsave(gif_path, frames, duration=args_cli.capture_every, loop=0)
    print(f"[INFO] wrote {len(frames)} frames and {gif_path}")


def run_simulator(sim: sim_utils.SimulationContext, entities: dict, dims: dict):
    """Runs the press schedule and reports the resulting deformation."""
    phantom: DeformableObject = entities["phantom"]
    probe: RigidObject = entities["probe"]
    device = sim.device

    camera: Camera | None = entities.get("camera")
    schedule = build_schedule()
    top_mask, mask = surface_node_masks(phantom, dims)
    probe_weight = args_cli.probe_mass * GRAVITY

    # marker overlay on the whole top face. Like the indentation, it is coloured against
    # the settled surface, not the rest shape, so the box's own sag does not tint it
    node_markers = None if args_cli.no_nodes else make_node_markers()
    node_bands = torch.tensor(NODE_BANDS, device=device)
    node_baseline_z = None

    # capture only on steps that were actually rendered
    capture_every = max(round(args_cli.capture_every / SIM_DT / RENDER_EVERY), 1) * RENDER_EVERY
    frames: list = []

    # buffers for the external wrench on the probe (num_instances, num_bodies, 3)
    forces = torch.zeros(1, 1, 3, device=device)
    torques = torch.zeros_like(forces)
    # the probe's home pose, restored every step unless --free_probe
    home_pose = torch.tensor([[0.0, 0.0, dims["probe_center_z"], 1.0, 0.0, 0.0, 0.0]], device=device)
    home_state = torch.cat([home_pose, torch.zeros(1, 6, device=device)], dim=-1)

    print("[INFO] press schedule:", " -> ".join(f"{label} ({dur:g}s)" for label, dur, _ in schedule))
    print(f"[INFO] probe weight ({probe_weight:.2f} N) is compensated, so --force is the net press force")

    # per-node height of the top surface after settling: the zero of the indentation.
    # taking it after settling rather than from the rest shape keeps the body's own sag
    # under gravity out of the measurement
    baseline_z = None
    phase_idx = 0
    phase_step = 0
    step_count = 0
    cycle = 0
    while simulation_app.is_running():
        label, duration, target_force = schedule[phase_idx]
        phase_len = max(int(duration / SIM_DT), 1)

        pressing = target_force > 0.0

        # -- the applied wrench carries the probe's own weight on top of the requested
        # force, so --force is exactly what reaches the tissue. Ramping it up over --ramp
        # seconds makes the surface sink visibly instead of snapping to its final dent
        ramp = min(phase_step * SIM_DT / args_cli.ramp, 1.0) if args_cli.ramp > 0.0 else 1.0
        forces[..., 2] = -(target_force * ramp - probe_weight) if pressing else 0.0
        probe.set_external_force_and_torque(forces, torques)

        # -- while released, park the probe at its home pose, clear of the surface, so the
        # tissue springs back on its own and every press starts from the same geometry
        if not pressing:
            probe.write_root_state_to_sim(home_state)
        # -- 1-DoF gantry: keep the probe upright and on the z axis, leave z to physics
        elif not args_cli.free_probe:
            pose = probe.data.root_state_w[:, :7].clone()
            pose[:, :2] = home_pose[:, :2]
            pose[:, 3:7] = home_pose[:, 3:7]
            probe.write_root_pose_to_sim(pose)
            vel = probe.data.root_state_w[:, 7:].clone()
            vel[:, :2] = 0.0  # no lateral drift
            vel[:, 3:] = 0.0  # no tipping
            probe.write_root_velocity_to_sim(vel)

        # -- overlay: dots riding the surface nodes, coloured by how far they have sunk.
        # updated before the step so they are in place for the frame it renders
        render_now = step_count % RENDER_EVERY == 0
        if node_markers is not None and render_now:
            node_pos = phantom.data.nodal_pos_w[0, top_mask].clone()
            drop = (node_baseline_z - node_pos[:, 2]) if node_baseline_z is not None else node_pos[:, 2] * 0.0
            node_pos[:, 2] += 0.004  # sit the dots on the surface rather than half inside it
            node_markers.visualize(translations=node_pos, marker_indices=torch.bucketize(drop, node_bands))

        # -- step
        probe.write_data_to_sim()
        sim.step(render=render_now)
        phantom.update(SIM_DT)
        probe.update(SIM_DT)
        phase_step += 1
        step_count += 1

        # -- capture
        if camera is not None and step_count % capture_every == 0:
            camera.update(SIM_DT)
            frames.append(camera.data.output["rgb"][0, ..., :3].clone().cpu().numpy().astype(np.uint8))

        # -- measure: deepest point of the dent, over the nodes under the probe
        surface_z = phantom.data.nodal_pos_w[0, mask, 2]
        if baseline_z is None and label == "settle" and phase_step == phase_len:
            baseline_z = surface_z.clone()
            node_baseline_z = phantom.data.nodal_pos_w[0, top_mask, 2].clone()
            print(f"[INFO] settled surface at z = {baseline_z.mean().item() * 1e3:.1f} mm (indentation zero)")
        indent = (baseline_z - surface_z).max().item() * 1e3 if baseline_z is not None else 0.0
        if baseline_z is not None and phase_step % int(0.5 / SIM_DT) == 0:
            probe_bottom = (probe.data.root_pos_w[0, 2].item() - args_cli.probe_height / 2.0) * 1e3
            print(
                f"  [{label:>12s}] t={phase_step * SIM_DT:4.1f}s  indentation={indent:6.2f} mm"
                f"  probe tip z={probe_bottom:6.1f} mm"
            )

        # -- advance / wrap the schedule
        if phase_step >= phase_len:
            if baseline_z is not None and target_force > 0.0:
                stiffness = target_force / (indent * 1e-3) if indent > 1e-6 else float("nan")
                print(
                    f"[RESULT] {target_force:6.1f} N  ->  {indent:6.2f} mm indentation"
                    f"  (contact stiffness {stiffness:7.0f} N/m)"
                )
            phase_step = 0
            phase_idx += 1
            if phase_idx >= len(schedule):
                phase_idx = 0
                cycle += 1
                max_cycles = args_cli.cycles if args_cli.cycles > 0 else (1 if args_cli.headless else 0)
                if max_cycles and cycle >= max_cycles:
                    print("[INFO] schedule finished")
                    save_frames(frames)
                    return
                # reset the phantom so every cycle starts from the same rest shape
                phantom.write_nodal_state_to_sim(phantom.data.default_nodal_state_w.clone())
                phantom.reset()
                probe.write_root_state_to_sim(home_state)
                probe.reset()
                baseline_z = None
                node_baseline_z = None


def main():
    """Main function."""
    if "cuda" not in args_cli.device:
        raise SystemExit(
            f"Deformable bodies are simulated on the GPU only, but --device is '{args_cli.device}'."
            " Re-run with --device cuda:0."
        )

    # initialise the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # design the scene
    scene_entities, dims = design_scene()

    # look at the contact point, in the viewport and through the capture camera alike
    eye, target = contact_view(dims)
    sim.set_camera_view(eye, target)

    # play the simulator
    sim.reset()
    if "camera" in scene_entities:
        scene_entities["camera"].set_world_poses_from_view(
            torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device)
        )
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities, dims)


if __name__ == "__main__":
    main()
    simulation_app.close()
