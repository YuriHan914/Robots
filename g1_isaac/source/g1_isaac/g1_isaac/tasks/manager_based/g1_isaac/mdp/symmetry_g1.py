# Copyright (c) 2025, Unitree Robotics Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 휴머노이드 로봇을 위한 좌우 대칭 변환 기능.

이 모듈은 G1 로봇의 좌우 이동 및 yaw 회전 정책 학습을 위한 대칭 변환 기능을 제공합니다.
한쪽 방향(좌측 이동, 좌회전)만 학습하고 반대 방향은 미러링으로 처리하여 
학습 효율성과 수렴 속도를 향상시킵니다.

주요 기능:
- 관측 데이터의 좌우 대칭 변환
- 액션의 좌우 대칭 변환  
- 속도 커맨드의 좌우 대칭 변환
- G1 29DOF 조인트의 정확한 좌우 매핑 및 부호 반전

G1 29DOF 조인트 순서 (joint_sdk_names 기준):
[0-5]: 왼쪽 다리 (left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll)
[6-11]: 오른쪽 다리 (right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll)  
[12-14]: 허리 (waist_yaw, waist_roll, waist_pitch)
[15-21]: 왼쪽 팔 (left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw)
[22-28]: 오른쪽 팔 (right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw)
"""

import torch
from tensordict import TensorDict
from typing import Tuple, Optional
from isaaclab.envs import ManagerBasedRLEnv


def compute_symmetric_states_g1(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> Tuple[TensorDict | None, torch.Tensor | None]:
    """G1 로봇을 위한 좌우 대칭 변환.
    
    한쪽 방향 (좌측 이동, 좌회전)의 학습 데이터를 미러링하여
    반대 방향 (우측 이동, 우회전) 데이터로 증강합니다.
    
    Args:
        env: G1 환경 인스턴스
        obs: 관측 데이터 (TensorDict)
        actions: 액션 데이터 (torch.Tensor)
        
    Returns:
        미러링된 (관측, 액션) 튜플
    """
    obs_aug = None
    actions_aug = None
    
    if obs is not None:
        obs_aug = obs.clone()
        if "policy" in obs_aug:
            obs_aug["policy"] = _transform_policy_obs_left_right_g1(env, obs_aug["policy"])
        if "critic" in obs_aug:
            obs_aug["critic"] = _transform_critic_obs_left_right_g1(env, obs_aug["critic"])
    
    if actions is not None:
        actions_aug = _transform_actions_left_right_g1(actions)
    
    return obs_aug, actions_aug


def _transform_policy_obs_left_right_g1(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """G1 정책 관측의 좌우 대칭 변환.
    
    velocity_env_cfg.py의 PolicyCfg 관측 순서 기준:
    - base_ang_vel (3): [wx, wy, wz] -> [wx, -wy, -wz]
    - projected_gravity (3): [gx, gy, gz] -> [gx, -gy, gz]  
    - velocity_commands (3): [vx, vy, wz] -> [vx, -vy, -wz]
    - joint_pos_rel (29): G1 29DOF 조인트 위치 미러링
    - joint_vel_rel (29): G1 29DOF 조인트 속도 미러링
    - last_action (29): 이전 액션 미러링
    
    총 관측 크기: 3+3+3+29+29+29 = 96 (history_length=5이면 96*5=480)
    """
    obs = obs.clone()
    device = obs.device
    
    # 히스토리가 있는 경우를 고려하여 마지막 차원에서 96개씩 처리
    obs_dim = 96  # 단일 타임스텝 관측 크기
    
    if obs.shape[-1] % obs_dim == 0:
        # 히스토리가 있는 경우 각 타임스텝별로 변환
        history_length = obs.shape[-1] // obs_dim
        obs_reshaped = obs.view(*obs.shape[:-1], history_length, obs_dim)
        
        for t in range(history_length):
            obs_t = obs_reshaped[..., t, :]
            
            # base_ang_vel [0:3]: [wx, wy, wz] -> [wx, -wy, -wz]
            obs_t[..., 0:3] = obs_t[..., 0:3] * torch.tensor([1, -1, -1], device=device)
            
            # projected_gravity [3:6]: [gx, gy, gz] -> [gx, -gy, gz] 
            obs_t[..., 3:6] = obs_t[..., 3:6] * torch.tensor([1, -1, 1], device=device)
            
            # velocity_commands [6:9]: [vx, vy, wz] -> [vx, -vy, -wz]
            obs_t[..., 6:9] = obs_t[..., 6:9] * torch.tensor([1, -1, -1], device=device)
            
            # joint_pos_rel [9:38]: G1 29DOF 조인트 미러링
            obs_t[..., 9:38] = _switch_g1_joints_left_right(obs_t[..., 9:38])
            
            # joint_vel_rel [38:67]: G1 29DOF 조인트 속도 미러링  
            obs_t[..., 38:67] = _switch_g1_joints_left_right(obs_t[..., 38:67])
            
            # last_action [67:96]: 이전 액션 미러링
            obs_t[..., 67:96] = _switch_g1_joints_left_right(obs_t[..., 67:96])
            
            obs_reshaped[..., t, :] = obs_t
        
        obs = obs_reshaped.view(*obs.shape[:-1], -1)
    else:
        # 히스토리가 없는 경우 직접 변환
        # base_ang_vel [0:3]
        obs[..., 0:3] = obs[..., 0:3] * torch.tensor([1, -1, -1], device=device)
        # projected_gravity [3:6]
        obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1, -1, 1], device=device)
        # velocity_commands [6:9]  
        obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1, -1, -1], device=device)
        # joint_pos_rel [9:38]
        obs[..., 9:38] = _switch_g1_joints_left_right(obs[..., 9:38])
        # joint_vel_rel [38:67]
        obs[..., 38:67] = _switch_g1_joints_left_right(obs[..., 38:67])
        # last_action [67:96]
        obs[..., 67:96] = _switch_g1_joints_left_right(obs[..., 67:96])
    
    return obs


def _transform_critic_obs_left_right_g1(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """G1 크리틱 관측의 좌우 대칭 변환.
    
    CriticCfg 관측 순서:
    - base_lin_vel (3): [vx, vy, vz] -> [vx, -vy, vz]
    - base_ang_vel (3): [wx, wy, wz] -> [wx, -wy, -wz]
    - projected_gravity (3): [gx, gy, gz] -> [gx, -gy, gz]
    - velocity_commands (3): [vx, vy, wz] -> [vx, -vy, -wz]
    - joint_pos_rel (29): G1 29DOF 조인트 미러링
    - joint_vel_rel (29): G1 29DOF 조인트 속도 미러링
    - last_action (29): 이전 액션 미러링
    
    총 관측 크기: 3+3+3+3+29+29+29 = 99 (history_length=5이면 99*5=495)
    """
    obs = obs.clone()
    device = obs.device
    
    obs_dim = 99  # 단일 타임스텝 크리틱 관측 크기
    
    if obs.shape[-1] % obs_dim == 0:
        # 히스토리가 있는 경우
        history_length = obs.shape[-1] // obs_dim
        obs_reshaped = obs.view(*obs.shape[:-1], history_length, obs_dim)
        
        for t in range(history_length):
            obs_t = obs_reshaped[..., t, :]
            
            # base_lin_vel [0:3]: [vx, vy, vz] -> [vx, -vy, vz]
            obs_t[..., 0:3] = obs_t[..., 0:3] * torch.tensor([1, -1, 1], device=device)
            
            # base_ang_vel [3:6]: [wx, wy, wz] -> [wx, -wy, -wz]
            obs_t[..., 3:6] = obs_t[..., 3:6] * torch.tensor([1, -1, -1], device=device)
            
            # projected_gravity [6:9]: [gx, gy, gz] -> [gx, -gy, gz]
            obs_t[..., 6:9] = obs_t[..., 6:9] * torch.tensor([1, -1, 1], device=device)
            
            # velocity_commands [9:12]: [vx, vy, wz] -> [vx, -vy, -wz]
            obs_t[..., 9:12] = obs_t[..., 9:12] * torch.tensor([1, -1, -1], device=device)
            
            # joint_pos_rel [12:41]: G1 29DOF 조인트 미러링
            obs_t[..., 12:41] = _switch_g1_joints_left_right(obs_t[..., 12:41])
            
            # joint_vel_rel [41:70]: G1 29DOF 조인트 속도 미러링
            obs_t[..., 41:70] = _switch_g1_joints_left_right(obs_t[..., 41:70])
            
            # last_action [70:99]: 이전 액션 미러링
            obs_t[..., 70:99] = _switch_g1_joints_left_right(obs_t[..., 70:99])
            
            obs_reshaped[..., t, :] = obs_t
        
        obs = obs_reshaped.view(*obs.shape[:-1], -1)
    else:
        # 히스토리가 없는 경우
        obs[..., 0:3] = obs[..., 0:3] * torch.tensor([1, -1, 1], device=device)
        obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1, -1, -1], device=device)
        obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1, -1, 1], device=device)
        obs[..., 9:12] = obs[..., 9:12] * torch.tensor([1, -1, -1], device=device)
        obs[..., 12:41] = _switch_g1_joints_left_right(obs[..., 12:41])
        obs[..., 41:70] = _switch_g1_joints_left_right(obs[..., 41:70])
        obs[..., 70:99] = _switch_g1_joints_left_right(obs[..., 70:99])
    
    return obs


def _transform_actions_left_right_g1(actions: torch.Tensor) -> torch.Tensor:
    """G1 액션의 좌우 대칭 변환."""
    return _switch_g1_joints_left_right(actions)


def _switch_g1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """G1 29DOF 조인트의 좌우 스왑 + 부호 반전.
    
    joint_sdk_names 순서:
    [0-5]: 왼쪽 다리 - left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
    [6-11]: 오른쪽 다리 - right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll  
    [12-14]: 허리 - waist_yaw, waist_roll, waist_pitch
    [15-21]: 왼쪽 팔 - left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
    [22-28]: 오른쪽 팔 - right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
    
    대칭평면 XZ 기준으로 Y축에 대한 회전(roll)과 Z축에 대한 회전(yaw)은 부호 반전이 필요합니다.
    """
    joint_data_switched = torch.zeros_like(joint_data)
    
    # 다리: 좌우 스왑 (left [0-5] <-> right [6-11])
    joint_data_switched[..., [0,1,2,3,4,5]] = joint_data[..., [6,7,8,9,10,11]]  # left <- right
    joint_data_switched[..., [6,7,8,9,10,11]] = joint_data[..., [0,1,2,3,4,5]]  # right <- left
    
    # 허리: 그대로 유지하되 yaw와 roll은 부호 반전
    joint_data_switched[..., [12,13,14]] = joint_data[..., [12,13,14]]
    
    # 팔: 좌우 스왑 (left [15-21] <-> right [22-28])
    joint_data_switched[..., [15,16,17,18,19,20,21]] = joint_data[..., [22,23,24,25,26,27,28]]  # left <- right  
    joint_data_switched[..., [22,23,24,25,26,27,28]] = joint_data[..., [15,16,17,18,19,20,21]]  # right <- left
    
    # roll 및 yaw 축 조인트 부호 반전
    # 인덱스 매핑:
    # hip_roll: [1(left), 7(right)] 
    # hip_yaw: [2(left), 8(right)]
    # ankle_roll: [5(left), 11(right)]
    # waist_yaw: [12], waist_roll: [13]  
    # shoulder_roll: [16(left), 23(right)]
    # shoulder_yaw: [17(left), 24(right)]
    # wrist_roll: [19(left), 26(right)]
    # wrist_yaw: [21(left), 28(right)]
    roll_yaw_indices = [1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28]
    joint_data_switched[..., roll_yaw_indices] *= -1.0
    
    return joint_data_switched


def create_mirrored_command_ranges(original_ranges) -> dict:
    """한쪽 방향만 학습하도록 커맨드 범위를 제한.
    
    Args:
        original_ranges: 원본 커맨드 범위 설정 (UniformLevelVelocityCommandCfg.Ranges)
        
    Returns:
        좌측 이동과 좌회전만 허용하는 제한된 범위 딕셔너리
    """
    return {
        "lin_vel_x": original_ranges.lin_vel_x,  # 전후진은 그대로
        "lin_vel_y": (0.0, original_ranges.lin_vel_y[1]) if original_ranges.lin_vel_y[1] > 0 else (0.0, 0.1),  # 좌측 이동만 (양수)
        "ang_vel_z": (0.0, original_ranges.ang_vel_z[1]) if original_ranges.ang_vel_z[1] > 0 else (0.0, 0.1),  # 좌회전만 (양수)
        "heading": getattr(original_ranges, 'heading', (-3.14, 3.14))  # 헤딩은 그대로 (있는 경우)
    }


def should_apply_mirror_for_deployment(lin_vel_y: float, ang_vel_z: float) -> bool:
    """배포 시 우측 이동이나 우회전 커맨드인지 판단.
    
    Args:
        lin_vel_y: Y축 선속도 (좌우 이동)
        ang_vel_z: Z축 각속도 (yaw 회전)
        
    Returns:
        미러링 적용 여부 (True: 우측 이동 또는 우회전)
    """
    return lin_vel_y < -0.01 or ang_vel_z < -0.01  # 작은 노이즈 고려
