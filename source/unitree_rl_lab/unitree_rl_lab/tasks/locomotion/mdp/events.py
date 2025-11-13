# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event terms for unitree tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_payload_relative_to_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.23),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    # Extract the assets
    robot: Articulation = env.scene[robot_cfg.name]
    payload: RigidObject = env.scene[payload_cfg.name]

    # Get robot base positions for the specified environments
    robot_base_pos = robot.data.root_pos_w[env_ids]  # Shape: (num_envs, 3)
    robot_base_quat = robot.data.root_quat_w[env_ids]  # Shape: (num_envs, 4) [w, x, y, z]

    # Convert position offset to tensor
    offset_tensor = torch.tensor(position_offset, device=env.device, dtype=torch.float32)
    offset_tensor = offset_tensor.unsqueeze(0).expand(len(env_ids), -1)  # Shape: (num_envs, 3)

    # For simplicity, we assume the robot is mostly upright and use world-frame offset
    # If the robot tilts significantly, we would need to rotate the offset by robot_base_quat
    # But for a quadruped on flat terrain, this approximation is sufficient
    payload_pos = robot_base_pos + offset_tensor

    # Set orientation (constant across all environments)
    payload_quat = torch.tensor(orientation, device=env.device, dtype=torch.float32)
    payload_quat = payload_quat.unsqueeze(0).expand(len(env_ids), -1)  # Shape: (num_envs, 4)

    # Combine position and orientation into pose
    payload_pose = torch.cat([payload_pos, payload_quat], dim=-1)  # Shape: (num_envs, 7)

    # Set zero velocity
    payload_velocity = torch.zeros((len(env_ids), 6), device=env.device, dtype=torch.float32)

    # Write the root state to simulation
    payload.write_root_pose_to_sim(payload_pose, env_ids=env_ids)
    payload.write_root_velocity_to_sim(payload_velocity, env_ids=env_ids)
