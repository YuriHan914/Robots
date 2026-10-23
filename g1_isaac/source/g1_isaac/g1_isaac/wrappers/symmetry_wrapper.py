# Copyright (c) 2025, Unitree Robotics Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 로봇을 위한 대칭 데이터 증강 래퍼.

이 래퍼는 RSL-RL 학습 시 실시간으로 좌우 대칭 변환을 적용하여
한쪽 방향 데이터를 양방향 데이터로 증강합니다.

사용법:
1. 기존 RslRlVecEnvWrapper 대신 G1SymmetryVecEnvWrapper 사용
2. mirror_prob로 미러링 확률 조절 (기본 50%)
3. 학습 중 자동으로 관측과 액션이 대칭 변환됨

주의사항:
- G1 29DOF 전용으로 설계됨
- 관측/액션 순서가 velocity_env_cfg.py와 일치해야 함
- 커맨드는 제한된 범위(한쪽 방향)로 설정되어야 함
"""

import torch
from typing import Tuple, Dict
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from ..tasks.locomotion.mdp.symmetry_g1 import _transform_actions_left_right_g1, compute_symmetric_states_g1


class G1SymmetryVecEnvWrapper(RslRlVecEnvWrapper):
    """G1 로봇을 위한 대칭 변환 환경 래퍼.
    
    RSL-RL 학습 중 확률적으로 관측과 액션을 좌우 대칭 변환하여
    데이터 증강 효과를 제공합니다.
    """

    def __init__(self, env: ManagerBasedRLEnv, mirror_prob: float = 0.5, **kwargs):
        """초기화.
        
        Args:
            env: G1 환경 인스턴스
            mirror_prob: 미러링 적용 확률 (0.0-1.0)
            **kwargs: RslRlVecEnvWrapper의 추가 인자들
        """
        super().__init__(env, **kwargs)
        self.mirror_prob = mirror_prob
        self._should_mirror = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        print(f"[G1SymmetryWrapper] Initialized with mirror_prob={mirror_prob}")
        print(f"[G1SymmetryWrapper] Environment: {env.__class__.__name__}")
        print(f"[G1SymmetryWrapper] Num envs: {self.num_envs}")

    def reset(self) -> Tuple[TensorDict, Dict]:
        """환경 리셋."""
        obs_dict, extras = super().reset()
        
        # 리셋 시 미러링 상태 초기화
        self._should_mirror.fill_(False)
        
        return obs_dict, extras

    def _mirror_obs(self, obs_dict: TensorDict) -> TensorDict:
        """미러링 대상 환경들의 관측에 좌우 대칭 변환 적용."""
        mirror_indices = self._should_mirror.nonzero(as_tuple=True)[0]
        if len(mirror_indices) == 0:
            return obs_dict
        
        obs_dict = obs_dict.clone()
        # 미러링할 환경들의 관측만 추출해서 대칭 변환
        obs_subset_mirrored, _ = compute_symmetric_states_g1(self.unwrapped, obs_dict[mirror_indices], None)
        # 변환된 관측을 원래 위치에 복사
        obs_dict[mirror_indices] = obs_subset_mirrored
        return obs_dict

    def get_observations(self) -> TensorDict:
        """관측을 가져오고 확률적으로 미러링 적용."""
        # 각 환경별로 미러링 여부 결정
        self._should_mirror = torch.rand(self.num_envs, device=self.device) < self.mirror_prob
        
        return self._mirror_obs(super().get_observations())

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict]:
        """환경 스텝 실행.
        
        미러링된 관측에 대응하는 액션도 역변환하여 환경에 적용합니다.
        """
        # 미러링된 환경들의 액션을 역변환 (관측과 반대 방향)
        if self._should_mirror.any():
            actions = actions.clone()
            actions[self._should_mirror] = _transform_actions_left_right_g1(actions[self._should_mirror])
        
        # 환경에 (역변환된) 액션 적용 (clip, dones, time_outs 처리는 부모 래퍼가 담당)
        obs_dict, rewards, dones, extras = super().step(actions)
        
        # 다음 관측도 동일한 미러링 패턴 적용
        return self._mirror_obs(obs_dict), rewards, dones, extras


class G1SymmetryDataAugmentation:
    """학습 데이터를 위한 정적 대칭 변환 유틸리티.
    
    오프라인 데이터나 배치 처리에서 사용할 수 있는 
    대칭 변환 기능을 제공합니다.
    """
    
    @staticmethod
    def augment_batch(
        observations: torch.Tensor, 
        actions: torch.Tensor, 
        env: ManagerBasedRLEnv,
        augment_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 데이터에 대칭 변환 적용.
        
        Args:
            observations: 관측 배치 [batch_size, obs_dim]
            actions: 액션 배치 [batch_size, action_dim]  
            env: 환경 인스턴스
            augment_ratio: 증강할 데이터 비율
            
        Returns:
            증강된 (관측, 액션) 배치
        """
        batch_size = observations.shape[0]
        augment_size = int(batch_size * augment_ratio)
        
        if augment_size == 0:
            return observations, actions
        
        # 무작위로 증강할 샘플 선택
        augment_indices = torch.randperm(batch_size)[:augment_size]
        
        # 선택된 샘플들의 관측과 액션
        obs_to_augment = observations[augment_indices]
        actions_to_augment = actions[augment_indices]
        
        # TensorDict 형식으로 변환 (간단화를 위해 policy만 가정)
        obs_dict = TensorDict({"policy": obs_to_augment}, batch_size=augment_size)
        
        # 대칭 변환 적용
        obs_dict_mirrored, actions_mirrored = compute_symmetric_states_g1(
            env, obs_dict, actions_to_augment
        )
        
        # 원본 데이터와 증강 데이터 결합
        observations_augmented = torch.cat([observations, obs_dict_mirrored["policy"]], dim=0)
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0)
        
        return observations_augmented, actions_augmented
    
    @staticmethod
    def create_symmetric_pair(
        observation: torch.Tensor,
        action: torch.Tensor, 
        env: ManagerBasedRLEnv
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """단일 샘플의 대칭 쌍 생성.
        
        Args:
            observation: 단일 관측 [obs_dim]
            action: 단일 액션 [action_dim]
            env: 환경 인스턴스
            
        Returns:
            대칭 변환된 (관측, 액션)
        """
        obs_dict = TensorDict({"policy": observation.unsqueeze(0)}, batch_size=1)
        action_batch = action.unsqueeze(0)
        
        obs_dict_mirrored, action_mirrored = compute_symmetric_states_g1(
            env, obs_dict, action_batch
        )
        
        return obs_dict_mirrored["policy"].squeeze(0), action_mirrored.squeeze(0)
