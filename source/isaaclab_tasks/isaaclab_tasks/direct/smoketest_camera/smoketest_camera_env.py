# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoketest Camera Environment.

A minimal camera-based RL task for fast verification of the TiledCamera rendering
pipeline. A sphere agent navigates toward a colored target on a ground plane, observed
by an overhead 32x32 RGB camera. The image is flattened and fed to an MLP policy.

This task is designed to converge in ~20-30 iterations (~1 min), testing:
  - TiledCamera initialization and rendering
  - GPU image capture → tensor pipeline
  - Renderer preset switching (RTX / Warp / OvRTX)
  - Image data flowing into observations and policy
  - Basic RL convergence with visual input
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, save_images_to_file
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from .smoketest_camera_env_cfg import SmoketestCameraEnvCfg


class SmoketestCameraEnv(DirectRLEnv):
    """Smoketest Camera Environment — sphere chases target under overhead camera."""

    cfg: SmoketestCameraEnvCfg

    def __init__(self, cfg: SmoketestCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale

        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros_like(self._forces)
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)

    def close(self):
        super().close()

    def _setup_scene(self):
        self._sphere = RigidObject(self.cfg.sphere_cfg)
        self._target = RigidObject(self.cfg.target_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        # Flat solid ground — no grid texture, clean at low resolution
        ground_cfg = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), roughness=1.0),
        )
        ground_cfg.func("/World/Ground", ground_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/Ground"])

        self.scene.rigid_objects["sphere"] = self._sphere
        self.scene.rigid_objects["target"] = self._target
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._forces[:] = 0.0
        self._forces[:, 0, :2] = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self._sphere.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
        )

    def _get_observations(self) -> dict:
        camera_data = self._tiled_camera.data.output["rgb"] / 255.0

        if self.cfg.write_image_to_file:
            save_images_to_file(camera_data, "smoketest_camera_rgb.png")

        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data = camera_data - mean_tensor

        obs = camera_data.reshape(self.num_envs, -1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        sphere_pos = wp.to_torch(self._sphere.data.root_pos_w)[:, :2]
        target_pos = wp.to_torch(self._target.data.root_pos_w)[:, :2]

        distance = torch.norm(sphere_pos - target_pos, dim=-1)
        reached = (distance < self.cfg.reach_threshold).float()

        # Progress reward: positive when approaching, negative when retreating
        # (potential-based shaping — same pattern as CatBot, Humanoid, Locomotion)
        progress = self._prev_distance - distance
        self._prev_distance = distance.clone()

        # Speed bonus: extra reward for reaching quickly (more time remaining = bigger bonus)
        time_remaining = 1.0 - (self.episode_length_buf.float() / self.max_episode_length)
        speed_bonus = reached * time_remaining

        reward = (
            self.cfg.rew_scale_progress * progress
            + self.cfg.rew_scale_reached * reached
            + self.cfg.rew_scale_speed_bonus * speed_bonus
            + self.cfg.rew_scale_oob * self.reset_terminated.float() * (1.0 - reached)
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        sphere_pos = wp.to_torch(self._sphere.data.root_pos_w)[:, :2]
        target_pos = wp.to_torch(self._target.data.root_pos_w)[:, :2]
        env_origins_xy = self.scene.env_origins[:, :2]

        distance = torch.norm(sphere_pos - target_pos, dim=-1)
        reached = distance < self.cfg.reach_threshold

        dist_from_center = torch.norm(sphere_pos - env_origins_xy, dim=-1)
        out_of_bounds = dist_from_center > self.cfg.arena_radius

        terminated = out_of_bounds | reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._sphere._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        TWO_PI = 2.0 * 3.14159
        origins = self.scene.env_origins[env_ids]

        # Randomize target position within inner zone
        tgt_angle = sample_uniform(0.0, TWO_PI, (num_resets,), self.device)
        tgt_radius = sample_uniform(0.0, self.cfg.target_spawn_radius, (num_resets,), self.device)

        target_pose = wp.to_torch(self._target.data.default_root_pose)[env_ids].clone()
        target_pose[:, 0] = origins[:, 0] + tgt_radius * torch.cos(tgt_angle)
        target_pose[:, 1] = origins[:, 1] + tgt_radius * torch.sin(tgt_angle)
        target_pose[:, 2] = origins[:, 2] + 0.12

        # Randomize sphere position, guaranteeing minimum separation from target
        sph_angle = sample_uniform(0.0, TWO_PI, (num_resets,), self.device)
        sph_radius = sample_uniform(
            self.cfg.min_spawn_separation, self.cfg.arena_radius * 0.8, (num_resets,), self.device
        )
        # Offset from target position, not from center, so separation is guaranteed
        sphere_pose = wp.to_torch(self._sphere.data.default_root_pose)[env_ids].clone()
        sphere_pose[:, 0] = target_pose[:, 0] + sph_radius * torch.cos(sph_angle)
        sphere_pose[:, 1] = target_pose[:, 1] + sph_radius * torch.sin(sph_angle)
        sphere_pose[:, 2] = origins[:, 2] + 0.15

        # Clamp sphere inside arena
        sphere_offset_x = sphere_pose[:, 0] - origins[:, 0]
        sphere_offset_y = sphere_pose[:, 1] - origins[:, 1]
        dist_from_center = torch.norm(torch.stack([sphere_offset_x, sphere_offset_y], dim=-1), dim=-1)
        clamp_scale = torch.clamp(self.cfg.arena_radius * 0.8 / (dist_from_center + 1e-6), max=1.0)
        sphere_pose[:, 0] = origins[:, 0] + sphere_offset_x * clamp_scale
        sphere_pose[:, 1] = origins[:, 1] + sphere_offset_y * clamp_scale

        self._target.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        self._sphere.write_root_pose_to_sim_index(root_pose=sphere_pose, env_ids=env_ids)
        self._sphere.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(num_resets, 6, device=self.device), env_ids=env_ids
        )

        # Initialize prev_distance for progress reward
        sphere_xy = sphere_pose[:, :2]
        target_xy = target_pose[:, :2]
        self._prev_distance[env_ids] = torch.norm(sphere_xy - target_xy, dim=-1)
