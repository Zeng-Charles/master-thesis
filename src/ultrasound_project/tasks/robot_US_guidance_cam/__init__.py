# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-robot-US-guidance-cam-v0: SonoGym's guidance task plus a RealSense D435 on the
end effector, with its own config (configs/env/robotic_US_guidance_cam.yaml).

robotic_US_guidance_cam.py is a near-verbatim copy of SonoGym's robotic_US_guidance.py
rather than a subclass of it: the task reads its scene settings into module-level globals
at import time, so a subclass would have been stuck with the submodule's yaml. Copying is
what lets the camera scene be retuned from this repo. The differences are the config and
asset paths, the class names, and the camera itself.

Action, observation, reward and reset behaviour are unchanged from the original guidance
task. The camera is not part of the observation space; its output is reachable via
`env.unwrapped.d435.data.output["rgb"]` / `["distance_to_image_plane"]`, e.g. for a future
perception pipeline or for offline inspection (see scripts/run_robot_US_guidance_cam_env.py).
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-robot-US-guidance-cam-v0",
    entry_point=f"{__name__}.robotic_US_guidance_cam:roboticUSGuidanceCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotic_US_guidance_cam:roboticUSGuidanceCamEnvCfg",
    },
)
