# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-robot-US-guidance-phantom-v0: a sibling of SonoGym's Isaac-robot-US-guidance-v0
with its own config (configs/env/robotic_US_guidance_phantom.yaml) pointed at a physical
phantom (lung_phantom) instead of a real patient, so the original guidance task/config is
never touched. See robotic_US_guidance_phantom.py -- it's a near-verbatim copy of
SonoGym's robotic_US_guidance.py; the only differences are the config/asset paths and the
class names.
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-robot-US-guidance-phantom-v0",
    entry_point=f"{__name__}.robotic_US_guidance_phantom:roboticUSPhantomEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robotic_US_guidance_phantom:roboticUSPhantomEnvCfg",
    },
)
