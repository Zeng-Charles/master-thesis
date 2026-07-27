# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-robot-US-guidance-v0 with a RealSense D435 camera mounted on the end effector.

Action/observation/reward/reset logic is entirely inherited, unmodified, from
roboticUSEnv -- this task only adds a camera sensor to the scene. The camera is not
wired into the observation space; its output is only reachable via
`env.unwrapped.d435.data.output["rgb"]` / `["distance_to_image_plane"]`, e.g. for a
future perception pipeline or for offline inspection (see
scripts/run_robot_US_guidance_cam_env.py).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass

from spinal_surgery import ASSETS_DATA_DIR
from spinal_surgery.tasks.robot_US_guidance.robotic_US_guidance import (
    INIT_STATE_BED,
    INIT_STATE_HUMAN,
    label_res,
    roboticUSEnv,
    roboticUSEnvCfg,
    scale_bed,
    scene_cfg,
    usd_file_list,
)

# end-effector link to mount the camera on, matching the same robot-type branch
# robotic_US_guidance.py itself uses for its SceneEntityCfg body_names
if scene_cfg["robot"]["type"] == "kuka":
    _EE_LINK_NAME = "lbr_link_ee"
else:
    _EE_LINK_NAME = "fr3_link8"

# RealSense D435 intrinsics: Isaac Lab's own tested RealSense intrinsic matrix
# (isaaclab/test/sensors/test_camera.py, at 320x240), scaled 2x to 640x480. A single
# simplified pinhole model is shared by both the "rgb" and "distance_to_image_plane"
# outputs -- the D435's separate RGB/IR-depth modules and their physical offset from
# each other are not modeled individually.
_D435_WIDTH, _D435_HEIGHT = 640, 480
_D435_SCALE = _D435_WIDTH / 320.0  # == _D435_HEIGHT / 240.0
_D435_FX_320, _D435_CX_320 = 229.31640625, 164.810546875
_D435_FY_240, _D435_CY_240 = 229.826171875, 122.1650390625
_D435_INTRINSIC_MATRIX = [
    _D435_FX_320 * _D435_SCALE, 0.0, _D435_CX_320 * _D435_SCALE,
    0.0, _D435_FY_240 * _D435_SCALE, _D435_CY_240 * _D435_SCALE,
    0.0, 0.0, 1.0,
]


@configclass
class roboticUSGuidanceCamEnvCfg(roboticUSEnvCfg):
    d435_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Robot_US/{_EE_LINK_NAME}/d435_cam",
        update_period=0.0,
        width=_D435_WIDTH,
        height=_D435_HEIGHT,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=_D435_INTRINSIC_MATRIX,
            width=_D435_WIDTH,
            height=_D435_HEIGHT,
            clipping_range=(0.2, 10.0),  # approx. D435 depth sensing range
        ),
        # PLACEHOLDER mount offset relative to the ee link frame -- tune pos/rot after
        # visually checking the rendered feed (scripts/run_robot_US_guidance_cam_env.py).
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


class roboticUSGuidanceCamEnv(roboticUSEnv):
    cfg: roboticUSGuidanceCamEnvCfg

    def _setup_scene(self):
        """Same scene as roboticUSEnv._setup_scene, plus a D435 camera on the end effector.

        The camera prim must exist before `self.scene.clone_environments(...)` runs so
        it gets replicated into every parallel env along with the robot/bed/human -- so
        the base class's scene-setup body is duplicated here (with the camera inserted)
        rather than calling super()._setup_scene() and adding the camera afterward, which
        would leave every env but the first without a working camera.
        """
        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # lights
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        dome_light_cfg.func("/World/Light", dome_light_cfg)

        # articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # D435 camera -- constructed here (before clone_environments below) so its
        # child prim under the ee link is present when the scene gets cloned.
        self.d435 = TiledCamera(self.cfg.d435_cfg)

        if scene_cfg["sim"]["vis_us"]:
            usd_folder = "usd_colored"
        else:
            usd_folder = "usd_no_contact"
        medical_bed_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bed",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ASSETS_DATA_DIR}/MedicalBed/"
                + usd_folder
                + "/hospital_bed.usd",
                scale=(scale_bed, scale_bed, scale_bed),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=INIT_STATE_BED,
        )
        medical_bed = RigidObject(medical_bed_cfg)

        # human:
        human_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Human",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_file_list,
                random_choice=False,
                scale=(label_res, label_res, label_res),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=0.000001,
                    max_angular_velocity=0.000001,
                    max_depenetration_velocity=0.00001,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=INIT_STATE_HUMAN,
        )
        self.human = RigidObject(human_cfg)
        # assign members
        self.scene.clone_environments(copy_from_source=False)
        # add articulation/sensors to scene
        self.scene.articulations["robot_US"] = self.robot
        self.scene.rigid_objects["human"] = self.human
        self.scene.sensors["d435"] = self.d435
