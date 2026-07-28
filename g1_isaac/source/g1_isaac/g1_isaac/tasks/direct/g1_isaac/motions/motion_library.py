# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-clip reference-motion library for the ASAP-style motion-tracking task.

This is a trimmed, IsaacLab-native stand-in for ASAP's ``MotionLibBase``/``MotionLibRobot``
(``humanoidverse/utils/motion_lib``): instead of porting ASAP's own MJCF/skeleton-based motion file
format and batched-FK engine (``Humanoid_Batch``), it wraps one :class:`MotionLoader` per CSV clip
(the format already used by the AMP task and ``scripts/play_motion.py``) and reproduces the parts of
ASAP's motion-library API the env needs: per-env clip assignment, duration-weighted sampling,
periodic resampling, and time-indexed state queries batched across (possibly different) clips.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .motion_loader import MotionLoader


def resolve_motion_files(motion_path: str) -> list[str]:
    """Resolve a motion-file config value to a sorted list of CSV paths.

    ``motion_path`` may be a single CSV file or a directory containing one or more ``*.csv`` clips
    (enabling multi-clip training without changing the config schema).
    """
    if os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.csv")))
        if not files:
            raise ValueError(f"No .csv motion clips found in directory: {motion_path}")
        return files
    if not os.path.isfile(motion_path):
        raise ValueError(f"Invalid motion file/directory path: {motion_path}")
    return [motion_path]


class MotionLibrary:
    """Owns one :class:`MotionLoader` per clip and assigns/samples them per environment."""

    def __init__(self, motion_files: list[str], fps: float, num_envs: int, device: torch.device | str) -> None:
        assert len(motion_files) > 0, "MotionLibrary requires at least one motion clip."
        self.device = device
        self.num_envs = num_envs
        self._loaders = [MotionLoader(motion_file=f, fps=fps, device=device) for f in motion_files]

        dof_names = self._loaders[0].dof_names
        for f, loader in zip(motion_files[1:], self._loaders[1:]):
            if loader.dof_names != dof_names:
                raise ValueError(
                    "All motion clips in a MotionLibrary must share the same DOF names/order "
                    f"(clip '{f}' does not match '{motion_files[0]}')."
                )
        self._dof_names = dof_names

        durations = torch.tensor([loader.duration for loader in self._loaders], dtype=torch.float32, device=device)
        self._durations = durations
        # duration-weighted sampling probability (more frames -> sampled more often), matching ASAP's
        # MotionLibBase default (`_sampling_prob` initialized uniform-by-frame-count over motions)
        self._sampling_prob = durations / durations.sum()

        # per-env assigned clip index, populated by load_motions()
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)

        print(f"Motion library loaded: {len(self._loaders)} clip(s) from {os.path.dirname(motion_files[0]) or '.'}")

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def num_dofs(self) -> int:
        return len(self._dof_names)

    @property
    def num_motions(self) -> int:
        return len(self._loaders)

    def load_motions(self, env_ids: torch.Tensor | None = None, random_sample: bool = True) -> None:
        """(Re)assign a reference clip to the given envs (all envs if ``env_ids`` is None).

        Args:
            env_ids: Environments to (re)assign. Defaults to all environments.
            random_sample: If True, sample clips with probability proportional to their duration
                (matches ASAP's training-time behavior). If False, assign clips round-robin in order
                (matches ASAP's deterministic eval-time behavior).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_samples = len(env_ids)
        if num_samples == 0:
            return
        if random_sample:
            sampled = torch.multinomial(self._sampling_prob, num_samples, replacement=True)
        else:
            sampled = torch.arange(num_samples, device=self.device) % self.num_motions
        self.motion_ids[env_ids] = sampled

    def get_motion_length(self, motion_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Per-env clip duration [s]."""
        motion_ids = self.motion_ids if motion_ids is None else motion_ids
        return self._durations[motion_ids]

    def sample_times(self, motion_ids: torch.Tensor) -> np.ndarray:
        """Uniform-random time in ``[0, duration]`` for each given (possibly per-env-different) clip."""
        durations = self._durations[motion_ids].cpu().numpy()
        return durations * np.random.uniform(low=0.0, high=1.0, size=len(motion_ids))

    def sample(
        self, motion_ids: torch.Tensor, times: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample interpolated motion state at ``times``, dispatching each env to its assigned clip.

        Args:
            motion_ids: Clip index per env, shape ``(N,)``.
            times: Query time per env (seconds, clip-local), shape ``(N,)``.

        Returns:
            Same 6-tuple as :meth:`MotionLoader.sample`, batched over ``N``.
        """
        motion_ids_np = motion_ids.cpu().numpy() if torch.is_tensor(motion_ids) else np.asarray(motion_ids)
        num_samples = len(motion_ids_np)
        out: list[torch.Tensor] | None = None
        for clip_id in np.unique(motion_ids_np):
            mask = motion_ids_np == clip_id
            loader = self._loaders[int(clip_id)]
            sampled = loader.sample(num_samples=int(mask.sum()), times=times[mask])
            if out is None:
                out = [torch.zeros((num_samples, *t.shape[1:]), dtype=t.dtype, device=self.device) for t in sampled]
            idx = torch.from_numpy(np.nonzero(mask)[0]).to(self.device)
            for buf, t in zip(out, sampled):
                buf[idx] = t
        assert out is not None
        return tuple(out)  # type: ignore[return-value]
