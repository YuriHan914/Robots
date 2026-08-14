# 🚀 5분 안에 시작하기

이 폴더의 파일들은 이미 저장소에 들어있다 - 새로 만들 필요 없이 바로 실행하면 된다.

- `g1_isaac_sim_bridge.py` - Isaac Sim 프로세스. G1 로드 + 물리 루프 + 걷기 제어 + skrl 정책 추론
- `unitree_g1_web_controller_complete.py` - 웹 서버 (DDS ↔ 브라우저 릴레이, 물리/추론 없음)
- `g1_web_ui_mobile_first.html` - 웹 UI

## 1️⃣ 의존성 설치 (1분)

이 프로젝트의 `isaac` conda 환경(이미 Isaac Lab이 설치되어 있는 환경)에 웹 서버용
패키지와 raw DDS 통신용 `cyclonedds`만 추가로 준비하면 된다. ROS 2는 필요 없다 - 이유는
`g1_isaac_sim_bridge.py`의 모듈 docstring 참고 (요약: ROS 2의 rclpy는 시스템 Python
버전에 고정 컴파일된 C 확장이라 Isaac Lab이 설치된 conda 환경의 Python 버전과 충돌하기
쉽고, `cyclonedds`는 pip 휠이 Python 버전별로 미리 빌드되어 있어 그 문제가 없다).

```bash
conda activate isaac
pip install flask flask-cors flask-socketio cyclonedds pillow
```
(`pillow`는 로봇 헤드 카메라 프레임을 JPEG로 인코딩하는 데 필요하다 - `--no_camera`로 카메라를
아예 안 쓰면 생략해도 된다.)

## 2️⃣ 실행 (2분)

### Terminal 1: Isaac Sim + G1 로봇 로드 + 컨트롤러
```bash
conda activate isaac
python controller/g1_isaac_sim_bridge.py
```
이 프로세스가 Isaac Sim을 띄우고 G1을 로드한 뒤, 물리 루프/걷기 제어/skrl 정책 추론/헤드
카메라 스트리밍을 전부 담당한다. `--policy_checkpoint <path>`로 재생할 skrl AMP 체크포인트를
직접 지정할 수 있고, 생략하면 `logs/skrl/g1_amp_dance`의 최신 체크포인트를 자동으로 사용한다.
카메라는 기본으로 켜져 있어 RTX 렌더링 부하가 추가되니, 필요 없으면 `--no_camera`로 끌 수
있다 (`--camera_width`/`--camera_height`로 해상도 조정, 기본 320x240).

### Terminal 2: 웹 서버 실행 (DDS ↔ 브라우저 릴레이)
```bash
conda activate isaac
python controller/unitree_g1_web_controller_complete.py
```
이 프로세스는 물리/추론을 하지 않고 웹 UI와 Terminal 1의 DDS 토픽(`g1/...`) 사이를
중계만 한다 - 토픽 목록은 `g1_isaac_sim_bridge.py`의 모듈 docstring 참고. 두 프로세스는
같은 DDS 도메인 id(`--dds_domain_id`, 기본값 0)로 실행해야 서로를 찾는다.

### Terminal 3 + 4 (선택): 머리 lidar → SLAM 지도 (LOCATION MAP에 실제 지도 표시)

Terminal 1이 `--no_lidar` 없이 실행 중이면 머리에 360도 lidar가 달리고 로봇 주위에
벽+장애물이 있는 방이 스폰된다(`_spawn_room`/`_attach_head_lidar` 참고). 이 lidar
데이터를 실제 SLAM 지도로 바꾸려면 ROS 2(Jazzy)가 필요한데, `rclpy`는 Isaac Lab이 깔린
`isaac` conda 환경(Python 3.11)과는 안 맞고(이유는 `g1_dds_types.py` 모듈 docstring
참고) **Python 3.12로 맞춘 별도 conda 환경**이 필요하다 - 이것도 conda 환경이라 지금
`isaac`처럼 `conda activate`만 하면 되고, 새로 필요한 시스템 패키지는 `slam_toolbox`
하나뿐이다:

```bash
# 최초 1회
sudo apt install ros-jazzy-slam-toolbox
conda create -n ros2_slam python=3.12 -y
conda activate ros2_slam
pip install cyclonedds pillow pyyaml numpy
conda deactivate
```

```bash
# Terminal 3: raw DDS <-> ROS 2 변환 브리지
source /opt/ros/jazzy/setup.bash
conda activate ros2_slam
export ROS_DOMAIN_ID=42   # --dds_domain_id(기본 0)와 겹치지 않는 값이면 아무거나
python3 controller/ros2_slam_bridge.py
```

```bash
# Terminal 4: slam_toolbox (같은 ros2_slam 환경)
source /opt/ros/jazzy/setup.bash
conda activate ros2_slam
export ROS_DOMAIN_ID=42
ros2 launch slam_toolbox online_async_launch.py \
    slam_params_file:=controller/slam_toolbox_params.yaml use_sim_time:=false
```

로봇을 걷게 하면 LOCATION MAP의 배경이 실제 방 지도로 채워진다 - 지도는 새로 갱신될
때만 업데이트되고(고정), 화살표만 계속 실시간으로 움직인다.

### Terminal 5 (또는 브라우저): 접속

**데스크톱:**
```
http://localhost:5000
```

**모바일:**
```
http://<your-ip>:5000
```

IP 찾기:
```bash
hostname -I
# 예: 192.168.1.100
```

---

## 🎮 조작법

### LOCATION MAP
레이더 화면을 클릭(또는 터치)하면 그 지점이 목표 지점(X/Y)으로 설정되고 빨간 십자
마커가 표시된다 - BASIC ACTIONS 패널의 X/Y 입력창에 직접 숫자를 입력해도 동일하게
동작한다. 실제 이동은 아래 **Walk**/**Run** 버튼을 눌러야 시작된다. 로봇의 현재
위치/방향(HDG)은 시안색 화살표로 실시간 표시된다. Terminal 3+4(위 SLAM 섹션)까지
띄웠다면 배경에 실제 SLAM 지도(벽/장애물)가 표시되고, 로봇이 새 구역을 탐색할 때만
갱신된다 - 지도 자체는 고정이고 화살표만 움직인다.

### CAM FEED
로봇 헤드에 마운트된 카메라의 실시간 영상. `--no_camera`로 실행했거나 시뮬레이터가
꺼져 있으면 "NO SIGNAL"이 표시된다.

### JOYSTICK CONTROLS
- **MOVE 패드**: 위/아래로 드래그 = 전진/후진, 좌/우 = 참고용(사이드스텝 미구현)
- **ROTATE 패드**: 좌/우로 드래그 = 회전 세기

### BASIC ACTIONS
- **Stand**: 정지 자세로 (수동 구동 중지)
- **Walk / Run**: 위 LOCATION MAP에서 지정한 X/Y 지점까지 걷기/뛰기로 이동
- **Home**: 기본/참조 자세로 리셋
- **Stop**: 긴급 정지 (진행 중인 이동 취소 + 정지 자세)
- **Reset**: 시뮬레이션 명시적 리셋 (넘어져도 자동 리셋 안 되므로 이 버튼 사용)

### MOTION
- **♪ Dance 1**: `scripts/skrl/train.py`로 학습한 AMP 댄스 정책 재생/정지 토글

---

## ❓ 자주 묻는 질문

**Q: 모바일에서 안 보여요**
```bash
# 1. 같은 WiFi 확인
# 2. 컴퓨터 IP 확인
hostname -I

# 3. 방화벽 허용
sudo ufw allow 5000
```

**Q: 웹 UI는 연결됐는데 로봇이 반응 안 함 (두 프로세스가 서로를 못 봄)**
```bash
# 두 프로세스가 같은 --dds_domain_id로 실행 중인지 확인 (기본값 0, 둘 다 생략하면 자동으로 맞음)
# 방화벽이 로컬 UDP 멀티캐스트를 막고 있지 않은지도 확인
```

**Q: "No module named 'flask'" / "No module named 'cyclonedds'"**
```bash
conda activate isaac
pip install flask flask-cors flask-socketio cyclonedds
```

**Q: 포트 이미 사용 중**
```bash
python controller/unitree_g1_web_controller_complete.py  # 기본 5000
lsof -i :5000   # 점유 중인 프로세스 확인 후 종료
```

---

## 📱 지원 기기

✅ **모바일**: iPhone (Safari), Android (Chrome)
✅ **데스크톱**: Windows, macOS, Linux (모든 브라우저)
✅ **태블릿**: iPad, Android 태블릿

**반응형 UI가 자동으로 최적화합니다!**

---

## ✨ 다음 단계

1. **걷기 게인 튜닝**: `g1_isaac_sim_bridge.py`의 `CpgGait` 진폭/주파수를 실제 시뮬레이션에서
   확인하며 조정 (관절 부호/축 컨벤션은 자산마다 다를 수 있음)
2. **정책 다양화**: `scripts/skrl/train.py` / `scripts/asap/train.py`로 새 정책을 학습해
   `--policy_checkpoint`로 재생
3. **실제 로봇 배포**: 동일한 DDS 토픽 계약을 실제 G1 위의 노드로 옮겨 Sim-to-Real 전이

---

더 자세한 아키텍처/커스터마이징은 `WEB_CONTROLLER_GUIDE.md` 참고.
