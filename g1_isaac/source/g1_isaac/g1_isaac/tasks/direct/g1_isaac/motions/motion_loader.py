# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loader for retargeted G1 motion clips (CSV) used as AMP reference motions.

The CSV format is the one produced by the G1 motion retargeting pipeline and consumed by
``scripts/play_motion.py``: a ``Frame`` column, ``root_translateX/Y/Z`` (cm), ``root_rotateX/Y/Z``
(deg, extrinsic XYZ euler) and one ``<joint_name>_dof`` column (deg) per articulated G1 joint. Unlike
the ``.npz`` motion files used by Isaac Lab's humanoid AMP example, the clip has no recorded
velocities or per-body poses, so DOF and root velocities are estimated here via finite differences at
load time and the AMP observation is built from root + DOF state only (no key-body tracking).
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch

from isaaclab.utils.math import quat_box_minus, quat_from_euler_xyz


class MotionLoader:
    """Loads a retargeted G1 motion clip and samples it (with interpolation) at arbitrary times."""

    def __init__(self, motion_file: str, fps: float, device: torch.device | str) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Path to the retargeted motion CSV file.
            fps: Playback frame rate of the motion clip (the CSV itself has no timestamps).
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = device
        self.dt = 1.0 / fps

        with open(motion_file) as f:
            rows = list(csv.reader(f))
        header, data_rows = rows[0], rows[1:]
        data = np.array(data_rows, dtype=np.float64)
        self.num_frames = data.shape[0]
        self.duration = self.dt * (self.num_frames - 1)

        def col(name: str) -> np.ndarray:
            return data[:, header.index(name)]

        # root position: cm -> m
        root_pos = (
            np.stack([col("root_translateX"), col("root_translateY"), col("root_translateZ")], axis=-1) / 100.0
        )
        # root orientation: extrinsic XYZ euler, deg -> rad -> quaternion (w, x, y, z)
        roll = np.deg2rad(col("root_rotateX"))
        pitch = np.deg2rad(col("root_rotateY"))
        yaw = np.deg2rad(col("root_rotateZ"))
        root_quat = quat_from_euler_xyz(torch.from_numpy(roll), torch.from_numpy(pitch), torch.from_numpy(yaw)).numpy()

        # DOF columns, kept in CSV order
        self._dof_names = [c[: -len("_dof")] for c in header if c.endswith("_dof")]
        dof_pos = np.stack([np.deg2rad(col(f"{name}_dof")) for name in self._dof_names], axis=-1)

        self.dof_positions = torch.tensor(dof_pos, dtype=torch.float32, device=self.device)
        self.root_positions = torch.tensor(root_pos, dtype=torch.float32, device=self.device)
        self.root_rotations = torch.tensor(root_quat, dtype=torch.float32, device=self.device)

        # the clip has no recorded velocities: estimate them via finite differences
        self.dof_velocities = torch.tensor(
            np.gradient(dof_pos, self.dt, axis=0), dtype=torch.float32, device=self.device
        )
        self.root_linear_velocities = torch.tensor(
            np.gradient(root_pos, self.dt, axis=0), dtype=torch.float32, device=self.device
        )
        self.root_angular_velocities = self._finite_difference_angular_velocity(self.root_rotations)

        print(f"Motion loaded ({motion_file}): duration: {self.duration:.2f} sec, frames: {self.num_frames}")

    def _finite_difference_angular_velocity(self, quat: torch.Tensor) -> torch.Tensor:
        """Forward-difference angular velocity (world frame) estimated from consecutive quaternions."""
        q_next = torch.cat([quat[1:], quat[-1:]], dim=0)
        ang_vel = quat_box_minus(q_next, quat) / self.dt
        ang_vel[-1] = ang_vel[-2]  # last frame has no "next" sample, hold the previous estimate
        return ang_vel

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names, in the order used by the internal tensors."""
        return self._dof_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton DOFs."""
        return len(self._dof_names)

    def _interpolate(self, a: torch.Tensor, *, blend: torch.Tensor, start: np.ndarray, end: np.ndarray) -> torch.Tensor:
        """Linear interpolation between consecutive values. ``a`` has shape (num_frames, X)."""
        a0, a1 = a[start], a[end]
        if a0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a0 + blend * a1

    def _slerp(self, q: torch.Tensor, *, blend: torch.Tensor, start: np.ndarray, end: np.ndarray) -> torch.Tensor:
        """Batched spherical linear interpolation between consecutive quaternions (w, x, y, z)."""
        q0, q1 = q[start], q[end]
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        sin_angle = torch.sin(angle)
        near_zero = sin_angle < 1.0e-6
        safe_sin_angle = torch.where(near_zero, torch.ones_like(sin_angle), sin_angle)
        blend = blend.unsqueeze(-1)
        ratio_a = torch.where(near_zero, 1.0 - blend, torch.sin((1.0 - blend) * angle) / safe_sin_angle)
        ratio_b = torch.where(near_zero, blend, torch.sin(blend * angle) / safe_sin_angle)
        return ratio_a * q0 + ratio_b * q1

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Compute the surrounding frame indexes and blend factor for the given times.

        Times outside ``[0, duration]`` are clamped to the clip boundaries.
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round().astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = np.clip((times - index_0 * self.dt) / self.dt, 0.0, 1.0)
        return index_0, index_1, torch.tensor(blend, dtype=torch.float32, device=self.device)

    def sample_times(self, num_samples: int) -> np.ndarray:
        """Sample random motion times uniformly in ``[0, duration]``."""
        return self.duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (interpolated) motion data at the given times (or random times if not provided).

        Returns:
            A tuple ``(dof_positions, dof_velocities, root_positions, root_rotations,
            root_linear_velocities, root_angular_velocities)``, each batched over ``len(times)``.
        """
        times = self.sample_times(num_samples) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.root_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_angular_velocities, blend=blend, start=index_0, end=index_1),
        )
