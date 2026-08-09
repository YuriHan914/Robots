# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared raw-DDS message types + topic names for the G1 web controller <-> Isaac Sim bridge.

Both ``unitree_g1_web_controller_complete.py`` (web relay) and ``g1_isaac_sim_bridge.py`` (Isaac Sim
process) import this module so their DDS topic/type definitions can never drift apart - DDS matches
publishers/subscribers by (topic name, type name), so both ends must agree exactly.

This is *raw* DDS via Eclipse CycloneDDS' Python binding (``pip install cyclonedds``) - no ROS 2 in
the loop - the same transport family Unitree's own ``unitree_sdk2py`` SDK uses to talk to a real G1
(its ``rt/lowcmd``/``rt/lowstate`` style topics), just with message types specific to this web
controller instead of the real robot's low-level command/state structs.

Deliberately not ROS 2: ROS 2's own Python bindings (rclpy) ship as a C extension compiled against
one specific system Python (e.g. apt's ROS 2 Jazzy on Ubuntu 24.04 targets Python 3.12), which breaks
under a conda env pinned to a different Python (this project's ``isaac`` env, e.g. 3.11 for Isaac Lab
compatibility) - the two ABIs simply aren't compatible. ``cyclonedds``'s PyPI wheels are built
per-Python-version already, so ``pip install cyclonedds`` inside whatever interpreter you use just
works, independent of any system ROS 2 install.

Both processes must run on the same DDS domain id (``--dds_domain_id``, default 0) and be reachable
over UDP multicast (same host/LAN by default; see ``CYCLONEDDS_URI`` for custom network config).
"""

from dataclasses import dataclass, field

from cyclonedds.core import Listener
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence

DDS_DOMAIN_ID_DEFAULT = 0


def make_participant(domain_id: int = DDS_DOMAIN_ID_DEFAULT) -> DomainParticipant:
    return DomainParticipant(domain_id)


def dds_listener(handler):
    """Wrap a plain ``handler(sample)`` callable into a CycloneDDS :class:`Listener` that drains
    every sample available on ``on_data_available`` (fires on CycloneDDS' own thread - callers must
    make ``handler`` thread-safe, e.g. via a lock around any shared state it touches).

    Exceptions raised inside ``handler`` are caught and printed here instead of being silently
    swallowed at the C callback boundary (a Python exception crossing back into CycloneDDS' native
    dispatch thread would otherwise just disappear, with no message on stdout/stderr at all) -
    without this, a bug in a handler looks identical to the DDS message never arriving.
    """

    def _on_data_available(reader) -> None:
        for sample in reader.take(N=32):
            if sample is not None:
                try:
                    handler(sample)
                except Exception:
                    import traceback

                    traceback.print_exc()

    return Listener(on_data_available=_on_data_available)


# ---------------------------------------------------------------------------
# Commands: web controller -> sim bridge
# ---------------------------------------------------------------------------


@dataclass
class CmdVel(IdlStruct, typename="g1.CmdVel"):
    """Manual joystick command. ``linear_y`` is accepted for forward-compat but ignored by the
    sim bridge's gait generator (no dedicated strafing gait, see ``CpgGait``)."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class GotoCommand(IdlStruct, typename="g1.GotoCommand"):
    """Walk/run to a world-frame ``(x, y)`` point (meters, robot-spawn-relative)."""

    x: float = 0.0
    y: float = 0.0
    gait: str = "walk"


@dataclass
class PolicyCommand(IdlStruct, typename="g1.PolicyCommand"):
    """``"play"`` starts the skrl AMP dance policy, ``"stop"`` returns to stand."""

    command: str = "stop"


@dataclass
class Trigger(IdlStruct, typename="g1.Trigger"):
    """Fire-and-forget signal (goto_cancel / home_position / emergency_stop). DDS structs need at
    least one field, so this carries a timestamp, mainly useful for logging/debugging."""

    stamp: float = 0.0


# ---------------------------------------------------------------------------
# Telemetry: sim bridge -> web controller
# ---------------------------------------------------------------------------


@dataclass
class RobotPose(IdlStruct, typename="g1.RobotPose"):
    """World pose of the robot base. Quaternion is (qw, qx, qy, qz)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0


@dataclass
class JointState(IdlStruct, typename="g1.JointState"):
    name: sequence[str] = field(default_factory=list)
    position: sequence[float] = field(default_factory=list)
    velocity: sequence[float] = field(default_factory=list)


@dataclass
class Status(IdlStruct, typename="g1.Status"):
    """JSON payload: ``{"mode", "gait", "target", "distance_to_target", "policy_loaded"}``."""

    payload_json: str = "{}"


@dataclass
class CameraFrame(IdlStruct, typename="g1.CameraFrame"):
    """One JPEG-encoded frame from the robot's head camera, base64-encoded so it can be forwarded
    to the browser as-is (as a data: URL) with no re-encoding on the web controller side."""

    width: int = 0
    height: int = 0
    jpeg_base64: str = ""


TOPIC_CMD_VEL = "g1/cmd_vel"
TOPIC_GOTO_COMMAND = "g1/goto_command"
TOPIC_GOTO_CANCEL = "g1/goto_cancel"
TOPIC_POLICY_COMMAND = "g1/policy_command"
TOPIC_HOME_POSITION = "g1/home_position"
TOPIC_RESET_SIM = "g1/reset_sim"
TOPIC_EMERGENCY_STOP = "g1/emergency_stop"
TOPIC_ROBOT_POSE = "g1/robot_pose"
TOPIC_JOINT_STATES = "g1/joint_states"
TOPIC_STATUS = "g1/status"
TOPIC_CAMERA_FRAME = "g1/camera_frame"
