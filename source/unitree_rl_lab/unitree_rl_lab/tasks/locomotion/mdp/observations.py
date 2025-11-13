from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def payload_relative_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    payload_cfg: SceneEntityCfg = SceneEntityCfg("payload"),
) -> torch.Tensor:
    """Compute the relative XY position and velocity of the payload in the robot's base frame.

    This observation provides the critic with privileged information about where the payload
    is located and how fast it's moving relative to the robot's base, expressed in the robot's
    local frame (forward/left).

    Args:
        env: The environment instance.
        robot_cfg: The robot asset configuration. Default is "robot".
        payload_cfg: The payload asset configuration. Default is "payload".

    Returns:
        Combined tensor (num_envs, 4) in robot's base frame containing:
            - [0] is the forward/backward position offset (robot's local x-axis)
            - [1] is the left/right position offset (robot's local y-axis)
            - [2] is the forward/backward velocity (robot's local x-axis)
            - [3] is the left/right velocity (robot's local y-axis)
    """
    # Extract the assets
    robot: Articulation = env.scene[robot_cfg.name]
    payload: RigidObject = env.scene[payload_cfg.name]

    # Get positions, velocities, and orientation in world frame
    robot_pos = robot.data.root_pos_w  # Shape: (num_envs, 3)
    robot_quat = robot.data.root_quat_w  # Shape: (num_envs, 4) [w, x, y, z]
    payload_pos = payload.data.root_pos_w  # Shape: (num_envs, 3)
    payload_lin_vel = payload.data.root_lin_vel_w  # Shape: (num_envs, 3)

    # Compute relative position in world frame
    relative_pos_world = payload_pos - robot_pos  # Shape: (num_envs, 3)

    # Extract quaternion components [w, x, y, z]
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]

    # Compute rotation matrix from quaternion (transpose for inverse rotation)
    # We only need first two rows since we're returning XY
    # R^T (world to body) first row - for x component in body frame
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)

    # R^T second row - for y component in body frame
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)

    # Apply rotation to get position in robot's base frame
    relative_pos_base_x = r00 * relative_pos_world[:, 0] + r01 * relative_pos_world[:, 1] + r02 * relative_pos_world[:, 2]
    relative_pos_base_y = r10 * relative_pos_world[:, 0] + r11 * relative_pos_world[:, 1] + r12 * relative_pos_world[:, 2]

    # Apply same rotation to get velocity in robot's base frame
    payload_vel_base_x = r00 * payload_lin_vel[:, 0] + r01 * payload_lin_vel[:, 1] + r02 * payload_lin_vel[:, 2]
    payload_vel_base_y = r10 * payload_lin_vel[:, 0] + r11 * payload_lin_vel[:, 1] + r12 * payload_lin_vel[:, 2]

    # Stack position and velocity: [pos_x, pos_y, vel_x, vel_y]
    return torch.stack([relative_pos_base_x, relative_pos_base_y, payload_vel_base_x, payload_vel_base_y], dim=-1)  # Shape: (num_envs, 4)
