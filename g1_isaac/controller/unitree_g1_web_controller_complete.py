# unitree_g1_web_controller_complete.py
"""Unitree G1 웹 기반 컨트롤러 (raw DDS 릴레이)

이 프로세스는 물리 시뮬레이션이나 정책 추론을 직접 수행하지 않는다 - 그 역할은 전부
``g1_isaac_sim_bridge.py``(Isaac Sim 프로세스, isaac conda 환경에서 python으로 직접 실행 -
Isaac Lab이 그 환경에 pip 설치되어 있어 isaaclab.sh가 필요 없다)가 담당한다. 여기서는
브라우저(WebSocket/SocketIO) <-> DDS(Eclipse CycloneDDS, ROS 2 아님) 사이의 얇은 중계만
수행한다:

* 웹 UI에서 온 조이스틱/목표지점/정책 명령을 ``g1/*`` DDS 토픽으로 publish
* ``g1_isaac_sim_bridge.py``가 publish하는 로봇 pose/joint_states/status를 구독해
  연결된 모든 브라우저 클라이언트로 다시 emit

토픽/메시지 타입은 ``g1_dds_types.py``에 정의되어 있고, 두 프로세스가 이 모듈을 함께
import해서 절대 어긋나지 않는다. ROS 2가 아니라 raw DDS(``pip install cyclonedds``)를 쓰는
이유는 ``g1_isaac_sim_bridge.py`` 모듈 docstring 참고 (요약: ROS 2의 rclpy C 확장은 특정
시스템 Python에 고정 컴파일되어 있어, Isaac Lab이 설치된 conda 환경의 Python 버전과 안 맞으면
import 자체가 깨진다 - cyclonedds는 pip 휠이 Python 버전별로 미리 빌드되어 있어 이런 문제가
없다).

실행:
    conda activate isaac
    pip install flask flask-cors flask-socketio cyclonedds   # 최초 1회

    # 1) Isaac Sim + 로봇 로드 + 컨트롤러 (다른 터미널)
    python controller/g1_isaac_sim_bridge.py

    # 2) 이 웹 서버
    python controller/unitree_g1_web_controller_complete.py
    http://localhost:5000        (컴퓨터)
    http://<your-ip>:5000        (모바일, 같은 네트워크)
"""

import argparse
import json
import logging
import math
import threading
import time

from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from g1_dds_types import (
    DDS_DOMAIN_ID_DEFAULT,
    TOPIC_CAMERA_FRAME,
    TOPIC_CMD_VEL,
    TOPIC_EMERGENCY_STOP,
    TOPIC_GOTO_CANCEL,
    TOPIC_GOTO_COMMAND,
    TOPIC_HOME_POSITION,
    TOPIC_JOINT_STATES,
    TOPIC_POLICY_COMMAND,
    TOPIC_RESET_SIM,
    TOPIC_ROBOT_POSE,
    TOPIC_STATUS,
    CameraFrame,
    CmdVel,
    GotoCommand,
    JointState,
    PolicyCommand,
    RobotPose,
    Status,
    Trigger,
    dds_listener,
    make_participant,
)
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML UI lives directly under controller/ (no templates/ subfolder); no separate static
# assets (the page is a single self-contained HTML file), so static serving is disabled to avoid
# exposing the rest of controller/ (source, docs) over HTTP.
app = Flask(__name__, template_folder=".", static_folder=None)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

_state_lock = threading.Lock()
latest_state = {
    "connected_to_sim": False,
    "mode": "stand",
    "gait": "stand",
    "target": None,
    "distance_to_target": None,
    "policy_loaded": False,
    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "heading": 0.0},
}

# DDS has no "disconnect" event for a publisher that just stops (UDP is fire-and-forget) - the only
# way to notice g1_isaac_sim_bridge.py went away is a watchdog: if no telemetry (pose/joint/status,
# published every sim step) has arrived recently, treat the sim as disconnected. See
# _connection_watchdog(), started from __main__.
_last_telemetry_time = 0.0
_TELEMETRY_TIMEOUT_S = 2.0  # generous vs. the sim's ~20 Hz telemetry rate (a message every ~0.05s)


def _yaw_deg_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    """Yaw (degrees) from an Isaac Lab (w, x, y, z) quaternion - same formula as
    ``g1_isaac_sim_bridge.py``'s ``_yaw_from_quat_wxyz``, duplicated here since these two scripts
    are independently deployable and don't share a math-utils module."""
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


class G1DdsBridge:
    """얇은 raw-DDS 릴레이: 웹 명령 -> DDS publish, DDS 텔레메트리 -> 웹(SocketIO)."""

    def __init__(self, domain_id: int):
        self.participant = make_participant(domain_id)

        self.cmd_vel_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_CMD_VEL, CmdVel))
        self.goto_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_GOTO_COMMAND, GotoCommand))
        self.goto_cancel_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_GOTO_CANCEL, Trigger))
        self.policy_writer = DataWriter(
            self.participant, Topic(self.participant, TOPIC_POLICY_COMMAND, PolicyCommand)
        )
        self.home_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_HOME_POSITION, Trigger))
        self.reset_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_RESET_SIM, Trigger))
        self.estop_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_EMERGENCY_STOP, Trigger))

        self._pose_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_ROBOT_POSE, RobotPose), listener=dds_listener(self._on_pose)
        )
        self._joint_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_JOINT_STATES, JointState),
            listener=dds_listener(self._on_joint_state),
        )
        self._status_reader = DataReader(
            self.participant, Topic(self.participant, TOPIC_STATUS, Status), listener=dds_listener(self._on_status)
        )
        self._camera_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_CAMERA_FRAME, CameraFrame),
            listener=dds_listener(self._on_camera_frame),
        )

        logger.info(f"✅ DDS(CycloneDDS) bridge ready on domain {domain_id}")

    def _on_pose(self, msg: RobotPose) -> None:
        heading = _yaw_deg_from_quat_wxyz(msg.qw, msg.qx, msg.qy, msg.qz)
        pose = {"x": msg.x, "y": msg.y, "z": msg.z, "heading": heading}
        global _last_telemetry_time
        with _state_lock:
            _last_telemetry_time = time.time()
            latest_state["connected_to_sim"] = True
            latest_state["pose"] = pose
        socketio.emit("robot_pose", pose)

    def _on_joint_state(self, msg: JointState) -> None:
        global _last_telemetry_time
        with _state_lock:
            _last_telemetry_time = time.time()
        socketio.emit(
            "joint_state",
            {"name": list(msg.name), "position": list(msg.position), "velocity": list(msg.velocity)},
        )

    def _on_status(self, msg: Status) -> None:
        try:
            status = json.loads(msg.payload_json)
        except ValueError:
            return
        global _last_telemetry_time
        with _state_lock:
            _last_telemetry_time = time.time()
            latest_state["connected_to_sim"] = True
            latest_state.update(status)
            snapshot = dict(latest_state)
        socketio.emit("status", snapshot)

    def _on_camera_frame(self, msg: CameraFrame) -> None:
        # forwarded as-is (still base64) - the browser renders it directly as a data: URL, no
        # re-encoding needed on this hop.
        socketio.emit("camera_frame", {"width": msg.width, "height": msg.height, "jpeg_base64": msg.jpeg_base64})


bridge: G1DdsBridge | None = None


def init_dds(domain_id: int) -> None:
    global bridge
    bridge = G1DdsBridge(domain_id)


def _connection_watchdog() -> None:
    """Background task: flip ``connected_to_sim`` back to False (and tell connected browsers) once
    telemetry from g1_isaac_sim_bridge.py has been silent for ``_TELEMETRY_TIMEOUT_S`` - the only way
    to notice the sim process died/was killed, since DDS never tells us a publisher went away."""
    while True:
        socketio.sleep(1.0)
        with _state_lock:
            timed_out = latest_state["connected_to_sim"] and (time.time() - _last_telemetry_time) > _TELEMETRY_TIMEOUT_S
            if timed_out:
                latest_state["connected_to_sim"] = False
            snapshot = dict(latest_state)
        if timed_out:
            logger.warning(f"No telemetry from g1_isaac_sim_bridge.py for {_TELEMETRY_TIMEOUT_S:.0f}s - marking disconnected")
            socketio.emit("status", snapshot)


# ==================== Flask 라우트 ====================


@app.route("/")
def index():
    return render_template("g1_web_ui_mobile_first.html")


@app.route("/api/status")
def get_status():
    with _state_lock:
        return jsonify(dict(latest_state))


@socketio.on("connect")
def handle_connect():
    with _state_lock:
        snapshot = dict(latest_state)
    emit("status", snapshot)


@socketio.on("cmd_vel")
def handle_cmd_vel(data):
    if bridge is None:
        return
    bridge.cmd_vel_writer.write(
        CmdVel(
            linear_x=float(data.get("linear_x", 0.0)),
            linear_y=float(data.get("linear_y", 0.0)),
            angular_z=float(data.get("angular_z", 0.0)),
        )
    )


@socketio.on("goto_target")
def handle_goto_target(data):
    """목표 지점 (x, y)으로 걷기/뛰기 모드 시작."""
    if bridge is None:
        return
    bridge.goto_writer.write(
        GotoCommand(x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)), gait=data.get("gait", "walk"))
    )


@socketio.on("cancel_goto")
def handle_cancel_goto(_data=None):
    if bridge is not None:
        bridge.goto_cancel_writer.write(Trigger(stamp=time.time()))


@socketio.on("play_policy")
def handle_play_policy(_data=None):
    """skrl로 학습된 AMP 댄스 정책 재생 시작 (sim 쪽에서 체크포인트 지연 로드)."""
    if bridge is not None:
        bridge.policy_writer.write(PolicyCommand(command="play"))


@socketio.on("stop_policy")
def handle_stop_policy(_data=None):
    if bridge is not None:
        bridge.policy_writer.write(PolicyCommand(command="stop"))


@socketio.on("home_position")
def handle_home(_data=None):
    if bridge is None:
        logger.warning("home_position: DDS bridge not initialized, command dropped")
        return
    logger.info("home_position: publishing g1/home_position")
    bridge.home_writer.write(Trigger(stamp=time.time()))
    emit("response", {"status": "home"})


@socketio.on("reset_sim")
def handle_reset_sim(_data=None):
    """'초기화' 버튼: 넘어졌든 아니든 시뮬레이션을 명시적으로 리셋한다 (자동 리셋은 없음,
    see g1_isaac_sim_bridge.py)."""
    if bridge is None:
        logger.warning("reset_sim: DDS bridge not initialized, command dropped")
        return
    logger.info("reset_sim: publishing g1/reset_sim")
    bridge.reset_writer.write(Trigger(stamp=time.time()))
    emit("response", {"status": "reset"})


@socketio.on("emergency_stop")
def handle_emergency_stop(_data=None):
    if bridge is not None:
        bridge.estop_writer.write(Trigger(stamp=time.time()))
    emit("response", {"status": "emergency_stop"}, broadcast=True)


# ==================== 메인 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web UI <-> DDS relay for the G1 sim bridge.")
    parser.add_argument("--port", type=int, default=5000, help="Web server port.")
    parser.add_argument(
        "--dds_domain_id",
        type=int,
        default=DDS_DOMAIN_ID_DEFAULT,
        help="CycloneDDS domain id (must match g1_isaac_sim_bridge.py).",
    )
    args = parser.parse_args()

    init_dds(args.dds_domain_id)
    socketio.start_background_task(_connection_watchdog)
    logger.info(f"🚀 웹 서버 시작: http://0.0.0.0:{args.port}")
    socketio.run(app, host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
