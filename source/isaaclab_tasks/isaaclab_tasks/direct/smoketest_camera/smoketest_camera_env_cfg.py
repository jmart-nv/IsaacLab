# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Smoketest Camera environment.

A lightweight camera-based RL task designed for fast (~1 min) verification of the
TiledCamera rendering pipeline across physics/renderer presets. Uses a small MLP
policy on flattened 32x32 RGB images instead of a CNN, eliminating backprop as the
bottleneck.

Scene: sphere agent on a ground plane, chasing a colored target marker, viewed by
an overhead tiled camera.
"""

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg()
    physx = PhysxCfg()
    newton = NewtonCfg()


@configclass
class SmoketestCameraEnvCfg(DirectRLEnvCfg):
    # env — short episodes so reaching the target quickly dominates wandering
    decimation = 2
    episode_length_s = 3.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysicsCfg())

    # sphere agent — saturated bright blue, ~3 pixels across at 32x32
    sphere_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=2.0,
                angular_damping=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0), emissive_color=(0.0, 0.0, 0.4), roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.15)),
    )

    # target marker — saturated bright green, no-collision ghost body
    # Non-kinematic with mass (Newton needs valid inertia), gravity disabled,
    # no collision_props so the sphere passes through.
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), emissive_color=(0.0, 0.4, 0.0), roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.12)),
    )

    # overhead tiled camera — 32x32 RGB, looking straight down
    # At z=1.5 with focal_length=10, FOV ≈ 93°, ground ≈ 3.2m wide, pixel ≈ 0.1m
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.5),
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=32,
        height=32,
        renderer_cfg=MultiBackendRendererCfg(),
    )

    write_image_to_file = False

    # spaces — 32x32x3 = 3072 flattened
    action_space = 2
    state_space = 0
    observation_space = 3072

    # viewer
    viewer = ViewerCfg(eye=(10.0, 10.0, 10.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # task parameters
    action_scale = 5.0
    arena_radius = 0.8
    target_spawn_radius = 0.3
    min_spawn_separation = 0.3

    # reward — progress-based (CatBot/Humanoid/Locomotion pattern)
    rew_scale_progress = 10.0
    rew_scale_reached = 10.0
    rew_scale_speed_bonus = 5.0
    rew_scale_oob = -5.0
    reach_threshold = 0.2
