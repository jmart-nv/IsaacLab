# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoketest Camera environment — lightweight camera pipeline verification task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Smoketest-Camera-Direct-v0",
    entry_point=f"{__name__}.smoketest_camera_env:SmoketestCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smoketest_camera_env_cfg:SmoketestCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SmoketestCameraPPORunnerCfg",
    },
)
