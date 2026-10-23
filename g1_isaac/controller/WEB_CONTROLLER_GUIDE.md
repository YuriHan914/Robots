# Unitree G1 웹 기반 컨트롤러 완전 가이드

## 📋 목차
1. 개요
2. 설치
3. 실행
4. 모바일 접속
5. 커스터마이징
6. 문제 해결

---

## 🎯 개요

### 특징

```
✅ 데스크톱 + 모바일 동시 지원
✅ 브라우저 접속 (설치 불필요)
✅ 실시간 WebSocket 통신
✅ 터치 제스처 지원
✅ 반응형 UI (자동 최적화)
✅ 클릭으로 목표 지점 지정 → 걷기/뛰기로 이동
✅ skrl로 학습된 AMP 댄스 정책 실시간 재생
✅ 로봇 헤드 카메라 실시간 스트리밍 (CAM FEED)
```

### 아키텍처

```
모바일/데스크톱 브라우저
    ↓ HTTP/WebSocket (localhost:5000)
unitree_g1_web_controller_complete.py (얇은 릴레이, 물리/추론 없음)
    ├─ raw DDS (Eclipse CycloneDDS, ROS2 아님)
    └─ WebSocket 통신
        ↓ /g1/cmd_vel, /g1/goto_command, /g1/policy_command, ... (DDS)
g1_isaac_sim_bridge.py (Isaac Sim 프로세스, isaac conda 환경에서 python으로 직접 실행)
    ├─ G1 로봇 로드 (Template-G1-Isaac-AMP-Dance-Direct-v0)
    ├─ CPG 걷기 컨트롤러 (수동/목표지점 이동)
    └─ skrl AMP 댄스 정책 추론
        ↓
Isaac Sim
```

물리 시뮬레이션과 정책 추론은 전부 `g1_isaac_sim_bridge.py` 프로세스 안에서 일어난다 -
웹 서버는 브라우저와 DDS 토픽 사이를 중계만 한다. 정확한 토픽 목록/타입은
`g1_isaac_sim_bridge.py`의 모듈 docstring을 참고.

---

## 💾 설치

두 프로세스 모두 이 프로젝트의 `isaac` conda 환경(Isaac Lab이 이미 설치된 환경)에서
실행한다 - 별도 venv를 새로 만들 필요는 없다.

### 1️⃣ 웹 서버 + DDS 의존성

```bash
conda activate isaac
pip install flask flask-cors flask-socketio cyclonedds
```

두 프로세스(웹 서버, Isaac Sim 브릿지) 모두 같은 `isaac` conda 환경에서 실행하면
`cyclonedds` 한 번만 설치하면 된다. ROS 2는 필요 없다 - `pip install cyclonedds`의
PyPI 휠은 Python 버전별로 미리 빌드되어 있어, 어떤 conda/venv Python에 설치하든
그대로 동작한다 (ROS 2의 rclpy처럼 시스템 Python 버전에 묶여 깨지는 일이 없다).

---

## 🚀 실행

### 방법 1: 간단 실행

```bash
# 1. Isaac Sim + G1 로드 + 걷기/정책 컨트롤러 (Terminal 1)
conda activate isaac
python controller/g1_isaac_sim_bridge.py
# 선택: --policy_checkpoint <path/to/checkpoint.pt> (생략 시 logs/skrl/g1_amp_dance의 최신본 사용)
# 선택: --dds_domain_id <n> (기본값 0, 웹 컨트롤러와 같은 값이어야 함)

# 2. 웹 컨트롤러 실행 (Terminal 2) - DDS <-> 브라우저 릴레이, 물리/추론 없음
conda activate isaac
python controller/unitree_g1_web_controller_complete.py

# 3. 브라우저에서 열기 (Terminal 3)
# 데스크톱: http://localhost:5000
# 모바일: http://<your-computer-ip>:5000
```

### 방법 2: SystemD 서비스로 자동 실행

```bash
sudo nano /etc/systemd/system/g1-web-controller.service
```

내용 (경로는 실제 저장소/conda 설치 위치에 맞게 수정):
```ini
[Unit]
Description=Unitree G1 Web Controller (DDS relay)
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/g1_isaac
Environment="PATH=/path/to/miniconda3/envs/isaac/bin"
ExecStart=/path/to/miniconda3/envs/isaac/bin/python controller/unitree_g1_web_controller_complete.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

활성화:
```bash
sudo systemctl enable g1-web-controller
sudo systemctl start g1-web-controller
sudo systemctl status g1-web-controller
```

Isaac Sim 프로세스(`g1_isaac_sim_bridge.py`)는 GPU/디스플레이가 필요해 보통 systemd
서비스로 돌리지 않고 직접 터미널에서 실행한다.

---

## 📱 모바일 접속

### iOS (iPhone/iPad)

1. **Safari 열기** → 주소창에 입력:
   ```
   http://<your-computer-ip>:5000
   ```

2. **홈 화면에 추가** (선택)
   - 공유 → 홈 화면에 추가
   - 앱처럼 사용 가능

### Android

1. **Chrome 열기** → 주소창에 입력:
   ```
   http://<your-computer-ip>:5000
   ```

2. **앱 설치** (선택)
   - 메뉴 → "앱 설치"
   - 홈 화면에 아이콘 추가

### 컴퓨터 IP 찾기

```bash
# 방법 1: ifconfig
ifconfig | grep "inet " | grep -v 127.0.0.1

# 방법 2: hostname
hostname -I

# 예시 출력:
# 192.168.1.100  ← 이 주소를 사용
```

---

## ⚙️ 커스터마이징

### 1. 걷기 게인 / 재생할 정책 체크포인트

CPG 걷기 게인(진폭/주파수)과 skrl 정책 체크포인트는 전부 **g1_isaac_sim_bridge.py**
(Isaac Sim 프로세스) 쪽 설정이다 - 웹 서버(unitree_g1_web_controller_complete.py)는
DDS 릴레이일 뿐 걷기/추론 로직을 갖고 있지 않다.

```python
# g1_isaac_sim_bridge.py, CpgGait.step() 안에서
amplitude = 0.35 if gait == "walk" else 0.55   # 진폭 (라디안)
frequency = (1.0 + 0.6 * speed) if gait == "walk" else (1.6 + 0.8 * speed)  # Hz
```

```bash
# 재생할 skrl 체크포인트를 직접 지정 (생략 시 logs/skrl/g1_amp_dance의 최신본 자동 사용)
python controller/g1_isaac_sim_bridge.py --policy_checkpoint logs/skrl/g1_amp_dance/<run>/checkpoints/<ckpt>.pt
```

### 2. 포트 번호 변경

```python
# unitree_g1_web_controller_complete.py 맨 아래
socketio.run(
    app,
    host='0.0.0.0',
    port=8000,  # 5000 → 8000 변경
    debug=False,
    use_reloader=False
)
```

### 3. UI 색상 변경

**g1_web_ui_mobile_first.html의 `:root` CSS 변수에서:**

```css
:root {
    --bg: oklch(16% 0.01 240);       /* 전체 배경 */
    --bg-card: oklch(20% 0.012 240); /* 패널 배경 */
    --cyan: oklch(72% 0.13 195);     /* 강조색 (레이더 화살표, 활성 버튼 등) */
    --orange: oklch(72% 0.13 55);    /* ROTATE 노브, 댄스 재생 중 색상 */
    --danger: oklch(62% 0.18 25);    /* Stop 버튼 등 */
}
```
색상값을 바꾸면 헤더/패널/버튼/레이더가 전부 일관되게 바뀐다.

### 4. 조이스틱(MOVE 패드) 민감도 조정

```javascript
// g1_web_ui_mobile_first.html에서, new DragPad('movePad', ...) 콜백 안
velocityX = -dy * 2.0;  // 2.0 → 3.0 (더 민감)
```

### 5. LOCATION MAP(레이더)의 표시 범위 변경

```javascript
// g1_web_ui_mobile_first.html에서
const RADAR_SCALE = 3.0;     // m당 픽셀 - 줄이면 더 넓은 범위가 화면에 들어옴
const RADAR_CLAMP_PX = 118;  // 레이더 반경 한계 (~280px 레이더 기준)
```

---

## 🔧 관절 각도 직접 제어 (고급)

조이스틱/목표지점 대신 정확한 관절 제어가 필요하면, `/g1/*` DDS 토픽 하나를 더 추가하는
방식으로 확장한다 - 웹 서버(`unitree_g1_web_controller_complete.py`)에 새 publisher와
`socketio.on` 핸들러를 추가하고, `g1_isaac_sim_bridge.py`의 메인 루프에서 해당 모드일 때
그 값을 `env.step()`에 넘길 `action` 텐서에 직접 반영하면 된다 (참고:
`g1_isaac_sim_bridge.py`의 `CpgGait`/`_goto_command`가 정확히 이 패턴).

---

## 🐛 트러블슈팅

### 문제 1: "ModuleNotFoundError: No module named 'flask'" / "'flask_socketio'"

```bash
conda activate isaac
pip install flask flask-cors flask-socketio
```

### 문제 2: "ModuleNotFoundError: No module named 'cyclonedds'"

```bash
conda activate isaac
pip install cyclonedds
python -c "import cyclonedds; print('ok')"
```

### 문제 3: 웹 UI에는 연결됐는데 로봇이 반응 안 함 (두 프로세스가 서로를 못 봄)

```bash
# 두 프로세스가 같은 --dds_domain_id로 실행 중인지 확인 (기본값 0)
# CycloneDDS는 기본적으로 로컬 UDP 멀티캐스트로 서로를 찾는다 - 방화벽이 막고 있지 않은지 확인
sudo ufw status

# 여러 네트워크 인터페이스가 있는 머신이면 CYCLONEDDS_URI로 인터페이스를 명시해야 할 수 있음
# (https://cyclonedds.io/docs/cyclonedds/latest/config/index.html 참고)
```

### 문제 4: 웹페이지 로드 안 됨

```bash
# 포트 사용 확인
lsof -i :5000

# 프로세스 종료
kill -9 <PID>

# 다시 실행
python controller/unitree_g1_web_controller_complete.py
```

### 문제 5: 모바일에서 접속 불가

```bash
# 1. 방화벽 확인
sudo ufw allow 5000

# 2. 라우터 설정 확인
# 같은 WiFi에 연결되어 있는지 확인

# 3. 컴퓨터 IP 다시 확인
hostname -I
```

### 문제 6: WebSocket 연결 오류

```python
# unitree_g1_web_controller_complete.py에서
socketio = SocketIO(
    app,
    cors_allowed_origins="*",  # 이 줄 확인
    ping_timeout=60,
    ping_interval=25
)
```

### 문제 7: 로봇이 걷다가 넘어짐 / 목표 지점 도달이 부정확함

`g1_isaac_sim_bridge.py`의 `CpgGait`는 학습된 정책이 아니라 손으로 짠 사인파 걸음걸이다.
관절 부호/진폭이 이 G1 자산의 실제 관절 축 컨벤션과 안 맞을 수 있으니, 시뮬레이션에서
직접 보면서 `CpgGait.step()`의 진폭/주파수/부호를 조정한다.

---

## 🔒 보안

### 1. 로컬만 접속 허용

```python
# unitree_g1_web_controller_complete.py 맨 아래
socketio.run(
    app,
    host='127.0.0.1',  # localhost만
    port=5000
)
```

### 2. 비밀번호 보호 (선택)

```python
from functools import wraps

PASSWORD = "your_password_here"

@app.before_request
def check_auth():
    if request.path == '/' and not session.get('authenticated'):
        # 인증 로직 추가
        pass
```

### 3. HTTPS 사용 (프로덕션)

```bash
# SSL 인증서 생성
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

```python
# unitree_g1_web_controller_complete.py에서
socketio.run(app, ssl_context=('cert.pem', 'key.pem'))
```

---

## 🚀 배포

```bash
# 로컬 네트워크 밖으로 공개 (선택) - ngrok으로 로컬 웹 서버를 인터넷에 노출
ngrok http 5000
```

Isaac Sim 프로세스는 GPU가 있는 로컬 머신에서 계속 돌리고, 웹 서버(및 ngrok)만
외부에 노출하는 구성을 권장한다.

---

## 📚 다음 단계

1. **걷기 게인 튜닝**: `CpgGait`를 실제 시뮬레이션에서 확인하며 조정
2. **새 정책 학습**: `scripts/skrl/train.py` / `scripts/asap/train.py`로 학습 후 `--policy_checkpoint`로 재생
3. **고급 UI**: 3D 시각화, 실시간 그래프 추가
4. **Sim-to-Real**: 동일한 DDS 토픽 계약을 실제 G1 위의 노드로 옮겨 배포

---

## 💡 팁

### 개발 중 빠른 테스트

```bash
# 핫 리로드 사용
python -m flask --app unitree_g1_web_controller_complete --debug run
```

### 모바일에서 디버깅

```javascript
// 콘솔 확인 (F12 개발자 도구)
console.log('디버그 메시지');

// 네트워크 탭에서 WebSocket 메시지 확인
```

---

**이제 완벽한 웹 기반 컨트롤러가 준비되었습니다!** 🎉

모바일과 데스크톱에서 동시에 G1을 제어할 수 있습니다. 아무 설치 없이 브라우저만 있으면 됩니다! 📱💻
