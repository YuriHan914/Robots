# Web Controller

*[English](web-controller.en.md)*

`controller/` 아래에 있는, 브라우저에서 시뮬레이션 상의 Unitree G1을 조작하는 웹 기반 컨트롤러.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `g1_isaac_sim_bridge.py` | Isaac Sim 프로세스. G1 로드, 물리 루프, 걷기 제어(CPG), skrl/PPO 정책 추론, 헤드 카메라·lidar 스트리밍을 전부 담당. `isaac` conda 환경에서 직접 실행. |
| `unitree_g1_web_controller_complete.py` | 웹 서버. Flask + Flask-SocketIO로 HTTP/WebSocket을 받아 raw DDS(Eclipse CycloneDDS)로 중계만 한다 - 물리/추론 로직 없음. |
| `g1_web_ui_mobile_first.html` | 웹 UI 본체 (LOCATION MAP, CAM FEED, 조이스틱, 액션 버튼). |
| `g1_dds_types.py` | 웹 서버 ↔ Isaac Sim 프로세스가 주고받는 DDS 메시지 타입/토픽 이름 정의. |
| `ros2_slam_bridge.py` | raw DDS(`g1/lidar_scan`, `g1/robot_pose`) ↔ ROS 2(`/scan`, `/tf`) 변환 브리지. `slam_toolbox`를 위해 별도의 `ros2_slam` conda 환경(Python 3.12)에서 실행. |
| `slam_toolbox_params.yaml` | `slam_toolbox online_async_launch.py`용 파라미터 (스캔 토픽, 프레임, lidar 레인지 등). |
| `QUICK_START.md` | 5분 안에 실행하는 최소 가이드. |
| `WEB_CONTROLLER_GUIDE.md` | 아키텍처, systemd 배포, 커스터마이징, 보안, 트러블슈팅까지 다루는 상세 가이드. |

아키텍처:

```
브라우저 (데스크톱/모바일)
    ↓ HTTP/WebSocket (:5000)
unitree_g1_web_controller_complete.py  (Flask + Flask-SocketIO, 얇은 릴레이)
    ↓ raw DDS (g1/cmd_vel, g1/goto_command, g1/policy_command, ...)
g1_isaac_sim_bridge.py  (Isaac Sim 프로세스)
    ├─ CPG 걷기 컨트롤러 (manual/goto)
    ├─ PPO Stand 정책 / skrl AMP 댄스 정책 추론
    ├─ 헤드 카메라 (RGB) 스트리밍
    └─ 헤드 lidar 스캔 → ros2_slam_bridge.py → slam_toolbox → g1.OccupancyMap
        ↓
Isaac Sim
```

> 참고: 웹 서버는 **Flask 기반**(FastAPI 아님)이며, `flask`/`flask-cors`/`flask-socketio`로 구현되어 있다.
> 코드 사실과 다르면 알려달라고 요청받아 실제 구현 기준으로 적었다.

## 통신 구조 상세 (Isaac Sim ↔ ROS 2)

두 개의 서로 다른 통신 레이어가 `ros2_slam_bridge.py` 한 프로세스 안에서 맞물려 있다 - 라이브러리도, DDS 도메인도 완전히 분리되어 있다.

### 레이어 1: raw DDS (Eclipse CycloneDDS) - 웹 UI ↔ Isaac Sim

- **라이브러리**: `cyclonedds` (Eclipse CycloneDDS의 Python 바인딩, `pip install cyclonedds`). ROS 2가 전혀 들어가지 않는 순수 DDS로, Unitree 자체 `unitree_sdk2py` SDK가 실제 G1과 통신할 때(`rt/lowcmd`/`rt/lowstate` 스타일)와 같은 전송 계열이다.
- **왜 ROS 2(rclpy)가 아닌가**: `rclpy`는 특정 시스템 Python에 고정 컴파일된 C 확장(예: Ubuntu 24.04 apt의 ROS 2 Jazzy는 Python 3.12 전용)이라, Isaac Lab이 설치된 `isaac` conda 환경(Python 3.11)에서 import 자체가 깨진다. `cyclonedds`는 PyPI 휠이 Python 버전별로 미리 빌드되어 있어 이 문제가 없다 (`g1_dds_types.py` 모듈 docstring).
- **참여자/도메인**: 양쪽 프로세스(`unitree_g1_web_controller_complete.py`, `g1_isaac_sim_bridge.py`, `ros2_slam_bridge.py`)가 동일한 `--dds_domain_id`(기본값 `0`)로 `DomainParticipant`를 생성해야 서로를 찾는다(UDP 멀티캐스트).
- **메시지 타입 정의**: `g1_dds_types.py`에 `cyclonedds.idl.IdlStruct` 데이터클래스로 전부 정의, 두 프로세스가 같은 모듈을 import해서 타입이 어긋나지 않도록 보장.

| 방향 | 토픽 | DDS 타입 | 필드 |
|---|---|---|---|
| 웹→Sim | `g1/cmd_vel` | `g1.CmdVel` | `linear_x`, `linear_y`(미사용), `angular_z` |
| 웹→Sim | `g1/goto_command` | `g1.GotoCommand` | `x`, `y`(m), `gait`(`"walk"`\|`"run"`) |
| 웹→Sim | `g1/goto_cancel` | `g1.Trigger` | `stamp` |
| 웹→Sim | `g1/policy_command` | `g1.PolicyCommand` | `command`(`"play"`\|`"stop"`) |
| 웹→Sim | `g1/home_position` | `g1.Trigger` | `stamp` |
| 웹→Sim | `g1/reset_sim` | `g1.Trigger` | `stamp` |
| 웹→Sim | `g1/emergency_stop` | `g1.Trigger` | `stamp` |
| Sim→웹 | `g1/robot_pose` | `g1.RobotPose` | `x,y,z`, `qw,qx,qy,qz` |
| Sim→웹 | `g1/joint_states` | `g1.JointState` | `name[]`, `position[]`, `velocity[]` |
| Sim→웹 | `g1/status` | `g1.Status` | `payload_json` (`{"mode","gait","target","distance_to_target","policy_loaded"}`) |
| Sim→웹 | `g1/camera_frame` | `g1.CameraFrame` | `width`, `height`, `jpeg_base64` |
| Sim→웹 | `g1/lidar_scan` | `g1.LidarScan` | `angle_min/max`, `angle_increment`, `range_min/max`, `ranges[]` (ROS 2 `sensor_msgs/LaserScan`과 필드 1:1 대응하도록 설계) |
| Sim(via ros2_slam_bridge)→웹 | `g1/occupancy_map` | `g1.OccupancyMap` | `width`, `height`, `resolution`, `origin_x/y`, `png_base64` (`g1.CameraFrame`과 동일하게 PNG+base64) |

### 레이어 2: ROS 2 (rclpy, Jazzy) - `ros2_slam_bridge.py` ↔ `slam_toolbox`

- **라이브러리**: `rclpy` (ROS 2 Jazzy, apt `ros-jazzy-slam-toolbox` + 시스템 ROS 2 설치), `tf2_ros`. raw DDS 쪽과 완전히 격리하기 위해 별도의 `ros2_slam` conda 환경(Python 3.12, `rclpy`의 컴파일 버전과 일치)에서 실행.
- **도메인 분리**: `ROS_DOMAIN_ID`(예 `42`)를 raw DDS의 `--dds_domain_id`(기본 `0`)와 겹치지 않게 설정 - 서로 무관한 두 DDS 네트워크(이 브리지의 raw-DDS 트래픽 vs. rclpy 자신의 ROS 2 DDS 트래픽)가 한 멀티캐스트 도메인을 공유하면 안 되기 때문.
- **`ros2_slam_bridge.py`가 발행/구독하는 ROS 2 토픽**:

| 방향 | 토픽 | ROS 2 메시지 타입 | QoS/비고 |
|---|---|---|---|
| 발행 | `/scan` | `sensor_msgs/msg/LaserScan` | `qos_profile_sensor_data`, `frame_id="base_link"` (lidar가 root/pelvis 바로 위 장착이라 별도 laser_link 오프셋 불필요) |
| 발행 (tf) | `odom` → `base_link` | `geometry_msgs/msg/TransformStamped` (`tf2_ros.TransformBroadcaster`) | sim의 ground-truth pose를 완벽한 오도메트리로 취급 |
| 구독 | `/map` | `nav_msgs/msg/OccupancyGrid` | `slam_toolbox`가 발행 (`map`→`odom` tf도 `slam_toolbox`가 자체 발행) |

- **변환 로직** (`Ros2SlamBridge` 클래스):
  - `g1.LidarScan`(raw DDS) → `sensor_msgs/LaserScan`(ROS 2): 필드가 1:1로 설계되어 있어 리샘플링 없이 그대로 복사.
  - `g1.RobotPose`(raw DDS, `qw,qx,qy,qz` 순서) → `odom→base_link` tf(`geometry_msgs/Quaternion`은 `x,y,z,w` 순서라 필드 순서를 바꿔 대입).
  - `nav_msgs/OccupancyGrid`(ROS 2, 셀 값 `-1`=unknown, `0~100`=free→occupied 확률) → `g1.OccupancyMap`(raw DDS): 그레이스케일 PNG로 인코딩(`-1`→회색 128, 확률→`255-(p*255/100)`), row 0(격자 원점)이 이미지 하단이 되도록 `flipud`.
- **`slam_toolbox` 실행**: `ros2 launch slam_toolbox online_async_launch.py slam_params_file:=controller/slam_toolbox_params.yaml use_sim_time:=false` - 파라미터는 `g1_isaac_sim_bridge.py`의 `LIDAR_MAX_RANGE`/`LIDAR_RANGE_MIN`과 맞춘 `max_laser_range`/`min_laser_range`, `scan_topic="/scan"`, `odom_frame="odom"` 등(`slam_toolbox_params.yaml`).

### 2D lidar 센서

실제 하드웨어가 아니라 **Isaac Lab의 `isaaclab.sensors.MultiMeshRayCaster`**를 `patterns.LidarPatternCfg`로 구성한 시뮬레이션 센서다(`_attach_head_lidar`, `g1_isaac_sim_bridge.py`):

| 파라미터 | 값 |
|---|---|
| 센서 클래스 | `isaaclab.sensors.MultiMeshRayCaster` |
| 패턴 | `isaaclab.sensors.ray_caster.patterns.LidarPatternCfg` |
| 채널 수 | 1 (단일 링, 2D 스캔) |
| 수직 FOV | 0.0 ~ 0.0도 (수평 단일 평면) |
| 수평 FOV | -180 ~ 180도 (360도 전체) |
| 수평 해상도 | 0.5도 |
| 최대 사거리 | 20 m (`LIDAR_MAX_RANGE`) |
| 최소 사거리 | 0.1 m (`LIDAR_RANGE_MIN`) |
| 정렬 | `ray_alignment="yaw"` - 몸통이 걷다가 pitch/roll 되어도 스캔 평면은 수평 유지 |
| 장착 위치 | 머리(`d435_link` 또는 head mount), Z+0.05m 오프셋 |
| 레이캐스트 대상 | `/World/ground`, `/World/room/*` (로봇 자신은 제외 - 실제 센서의 자기 차폐는 시뮬레이션하지 않음) |

실제 2D SLAM용 스피닝 lidar(예: RPLIDAR, Hokuyo 계열)와 동일한 "단일 평면 360도" 스캔 패턴을 재현하도록 설계되어 있으나, 특정 제조사/모델을 시뮬레이션한 것은 아니다.

## 날짜별 업데이트

### 2026-08-10 - 웹 컨트롤러 UI 최초 생성
- Flask + Flask-SocketIO 기반 웹 서버(`unitree_g1_web_controller_complete.py`) 신설: 브라우저와 Isaac Sim 프로세스 사이를 raw DDS(CycloneDDS)로 중계.
- 반응형 웹 UI(`g1_web_ui_mobile_first.html`) 추가: LOCATION MAP(레이더 클릭으로 목표 지점 지정), 조이스틱(MOVE/ROTATE 패드), BASIC ACTIONS(Stand/Walk/Run/Home/Stop/Reset), MOTION(Dance 재생) 패널. 데스크톱/모바일/태블릿 반응형.
- Isaac Sim 브리지(`g1_isaac_sim_bridge.py`) 신설: G1 로드, `manual`(조이스틱 CPG 걷기)/`goto`(목표 지점 이동)/`policy`(skrl AMP 댄스 재생)/`stand` 4가지 제어 모드, DDS 토픽 계약(`g1_dds_types.py`) 정의.
- 로봇 헤드에 전방 RGB 카메라(`isaaclab.sensors.Camera`)를 등록 태스크와 무관하게 사후 장착해 웹 UI의 CAM FEED 패널로 JPEG 프레임 스트리밍 (`_attach_head_camera`/`_encode_camera_frame`, DDS 토픽 `g1/camera_frame`). `--no_camera`로 비활성화 가능.

### 2026-08-14 - 헤드 lidar + 2D SLAM 지도, PPO Stand 정책 연동
- 로봇 머리에 **Isaac Lab `MultiMeshRayCaster` + `patterns.LidarPatternCfg` 기반 2D lidar**(단일 채널, 수평 360도, 0.5도 해상도, 최대 20m)를 장착(`_attach_head_lidar`, `g1_isaac_sim_bridge.py`). 실제 하드웨어 스펙에 대응하는 시뮬레이션 센서이며, 스캔이 부딪힐 벽/장애물이 있는 방(`_spawn_room`)도 함께 스폰. DDS 토픽 `g1/lidar_scan` (`g1_dds_types.py`의 `g1.LidarScan`)으로 발행. `--no_lidar`로 비활성화 가능.
- `ros2_slam_bridge.py` 신설: `g1/lidar_scan` + `g1/robot_pose`(raw DDS) → `/scan` + `/tf`(ROS 2)로 변환하는 브리지. `rclpy`가 `isaac` conda 환경(Python 3.11)과 바이너리 호환이 안 돼 별도의 `ros2_slam` conda 환경(Python 3.12)에서 실행.
- `slam_toolbox_params.yaml` 추가: jazzy-toolbox(`ros-jazzy-slam-toolbox`)의 `online_async_launch.py`용 파라미터 - `g1_isaac_sim_bridge.py`의 `LIDAR_MAX_RANGE`/`LIDAR_RANGE_MIN`과 맞춘 `max_laser_range`/`min_laser_range`, `scan_topic="/scan"`, `odom_frame="odom"` 등. `slam_toolbox`가 생성한 점유 격자 지도는 다시 raw DDS(`g1.OccupancyMap`)로 웹 UI에 전달되어 LOCATION MAP 배경에 표시된다(새 영역을 탐색할 때만 갱신).
- PPO 기반 **Stand** 정책을 실시간 연동: `G1-PPO-Direct-Stand-v0` 체크포인트(`--stand_policy_checkpoint`, 기본값 `logs/skrl/g1_stand/2026-08-13_02-36-26_ppo_torch/checkpoints/best_agent.pt`)를 `_StandPolicyController`가 매 스텝 이 스크립트 자신의 env/robot에 대해 직접 추론 - 로드 시 자동 실행되고, 웹 UI의 `policy_command`(`g1/policy_command` = `"play"`)로도 트리거된다.
- 학습된 Stand 정책의 관절 포즈/게인 스냅샷(`_TRAINED_STAND_POSE`/`_TRAINED_STAND_GAIN`, `_apply_trained_stand_defaults`)을 시작 시 로봇에 적용해, `stand` 모드와 `env.reset()`이 매번 이 학습된 자세로 서도록 함.
- **Reset / Home** 버튼을 명시적 `env.reset()` 트리거로 연결(`g1/home_position`, `g1/reset_sim` DDS 토픽) - 환경 자체의 자동 reset-on-fall이 꺼져 있어(`early_termination=False`) 넘어진 로봇은 이 버튼으로만 복구된다. `g1/emergency_stop`은 리셋 없이 즉시 `stand`로 전환.
