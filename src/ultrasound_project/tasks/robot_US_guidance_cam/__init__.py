# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
