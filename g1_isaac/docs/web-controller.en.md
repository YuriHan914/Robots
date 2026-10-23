# Web Controller

*[한국어](web-controller.md)*

A browser-based web controller (under `controller/`) for driving the simulated Unitree G1.

## Components

| File | Role |
|---|---|
| `g1_isaac_sim_bridge.py` | The Isaac Sim process. Owns G1 loading, the physics loop, walking control (unitree_rl_lab locomotion policy), skrl/PPO policy inference, and head camera/lidar streaming. Run directly in the `isaac` conda environment. |
| `unitree_g1_web_controller_complete.py` | The web server. Flask + Flask-SocketIO receives HTTP/WebSocket traffic and relays it over raw DDS (Eclipse CycloneDDS) - no physics/inference logic of its own. |
| `g1_web_ui_mobile_first.html` | The web UI itself (LOCATION MAP, CAM FEED, joysticks, action buttons). |
| `g1_dds_types.py` | DDS message types/topic names shared between the web server and the Isaac Sim process. |
| `ros2_slam_bridge.py` | Translation bridge between raw DDS (`g1/lidar_scan`, `g1/robot_pose`) and ROS 2 (`/scan`, `/tf`). Runs in a separate `ros2_slam` conda environment (Python 3.12) for `slam_toolbox`. |
| `slam_toolbox_params.yaml` | Parameters for `slam_toolbox online_async_launch.py` (scan topic, frames, lidar range, etc). |
| `QUICK_START.md` | Minimal guide to get running in 5 minutes. |
| `WEB_CONTROLLER_GUIDE.md` | Detailed guide covering architecture, systemd deployment, customization, security, and troubleshooting. |

Architecture:

```
Browser (desktop/mobile)
    | HTTP/WebSocket (:5000)
unitree_g1_web_controller_complete.py  (Flask + Flask-SocketIO, thin relay)
    | raw DDS (g1/cmd_vel, g1/goto_command, g1/policy_command, ...)
g1_isaac_sim_bridge.py  (Isaac Sim process)
    +- locomotion policy (manual joystick / goto target point)
    +- PPO Stand policy / skrl AMP dance policy inference
    +- head camera (RGB) streaming
    +- head lidar scan -> ros2_slam_bridge.py -> slam_toolbox -> g1.OccupancyMap
        |
Isaac Sim
```

> Note: the web server is **Flask-based** (not FastAPI), implemented with `flask`/`flask-cors`/`flask-socketio`. Documented against the actual implementation rather than the originally assumed stack.

## Communication architecture in detail (Isaac Sim <-> ROS 2)

Two separate communication layers meet inside a single process, `ros2_slam_bridge.py` - different libraries, different DDS domains, fully isolated from each other.

### Layer 1: raw DDS (Eclipse CycloneDDS) - Web UI <-> Isaac Sim

- **Library**: `cyclonedds`, the Python binding for Eclipse CycloneDDS (`pip install cyclonedds`). Plain DDS with no ROS 2 involved at all - the same transport family Unitree's own `unitree_sdk2py` SDK uses to talk to a real G1 (its `rt/lowcmd`/`rt/lowstate`-style topics).
- **Why not ROS 2 (rclpy)**: `rclpy` ships as a C extension compiled against one specific system Python (e.g. apt's ROS 2 Jazzy on Ubuntu 24.04 targets Python 3.12 only), which simply fails to import inside the `isaac` conda environment (Python 3.11, pinned for Isaac Lab compatibility). `cyclonedds`'s PyPI wheels are pre-built per Python version, so this problem doesn't come up (see `g1_dds_types.py`'s module docstring).
- **Participants/domain**: all three processes (`unitree_g1_web_controller_complete.py`, `g1_isaac_sim_bridge.py`, `ros2_slam_bridge.py`) must create their `DomainParticipant` on the same `--dds_domain_id` (default `0`) to discover each other over UDP multicast.
- **Message type definitions**: all defined as `cyclonedds.idl.IdlStruct` dataclasses in `g1_dds_types.py`; both processes import the same module so the types can never drift apart.

| Direction | Topic | DDS type | Fields |
|---|---|---|---|
| Web -> Sim | `g1/cmd_vel` | `g1.CmdVel` | `linear_x`, `linear_y` (strafe), `angular_z` |
| Web -> Sim | `g1/goto_command` | `g1.GotoCommand` | `x`, `y` (m), `gait` (`"walk"`\|`"run"`), `speed` (m/s, 0 = gait default) |
| Web -> Sim | `g1/goto_cancel` | `g1.Trigger` | `stamp` |
| Web -> Sim | `g1/policy_command` | `g1.PolicyCommand` | `command` (`"play"`\|`"stop"`) |
| Web -> Sim | `g1/home_position` | `g1.Trigger` | `stamp` |
| Web -> Sim | `g1/reset_sim` | `g1.Trigger` | `stamp` |
| Web -> Sim | `g1/emergency_stop` | `g1.Trigger` | `stamp` |
| Web -> Sim | `g1/stand_command` | `g1.Trigger` | `stamp` |
| Sim -> Web | `g1/robot_pose` | `g1.RobotPose` | `x,y,z`, `qw,qx,qy,qz` |
| Sim -> Web | `g1/joint_states` | `g1.JointState` | `name[]`, `position[]`, `velocity[]` |
| Sim -> Web | `g1/status` | `g1.Status` | `payload_json` (`{"mode","gait","target","distance_to_target","policy_loaded","stand_policy","loco","goto_speed","camera_error"}`) |
| Sim -> Web | `g1/camera_frame` | `g1.CameraFrame` | `width`, `height`, `jpeg_base64` |
| Sim -> Web | `g1/lidar_scan` | `g1.LidarScan` | `angle_min/max`, `angle_increment`, `range_min/max`, `ranges[]` (field layout deliberately mirrors ROS 2's `sensor_msgs/LaserScan` 1:1) |
| Sim (via ros2_slam_bridge) -> Web | `g1/occupancy_map` | `g1.OccupancyMap` | `width`, `height`, `resolution`, `origin_x/y`, `png_base64` (PNG+base64, same wire pattern as `g1.CameraFrame`) |

### Layer 2: ROS 2 (rclpy, Jazzy) - `ros2_slam_bridge.py` <-> `slam_toolbox`

- **Library**: `rclpy` (ROS 2 Jazzy, apt `ros-jazzy-slam-toolbox` + a system ROS 2 install), `tf2_ros`. Runs in a separate `ros2_slam` conda environment (Python 3.12, matching `rclpy`'s compiled ABI) to stay fully isolated from the raw-DDS side.
- **Domain separation**: `ROS_DOMAIN_ID` (e.g. `42`) must not overlap with raw DDS's `--dds_domain_id` (default `0`) - two unrelated DDS networks (this bridge's raw-DDS traffic vs. rclpy's own ROS 2 DDS traffic) must not share one multicast domain.
- **ROS 2 topics published/subscribed by `ros2_slam_bridge.py`**:

| Direction | Topic | ROS 2 message type | QoS/notes |
|---|---|---|---|
| Publish | `/scan` | `sensor_msgs/msg/LaserScan` | `qos_profile_sensor_data`, `frame_id="base_link"` (the lidar is mounted directly above the root/pelvis, so no separate laser_link offset is needed) |
| Publish (tf) | `odom` -> `base_link` | `geometry_msgs/msg/TransformStamped` (via `tf2_ros.TransformBroadcaster`) | treats the sim's ground-truth pose as perfect odometry |
| Subscribe | `/map` | `nav_msgs/msg/OccupancyGrid` | published by `slam_toolbox` (which also publishes its own `map` -> `odom` tf) |

- **Translation logic** (`Ros2SlamBridge` class):
  - `g1.LidarScan` (raw DDS) -> `sensor_msgs/LaserScan` (ROS 2): fields are designed 1:1, copied directly with no resampling.
  - `g1.RobotPose` (raw DDS, `qw,qx,qy,qz` order) -> `odom -> base_link` tf (`geometry_msgs/Quaternion` is `x,y,z,w` order, so fields are reassigned accordingly).
  - `nav_msgs/OccupancyGrid` (ROS 2, cell values `-1`=unknown, `0-100`=free->occupied probability) -> `g1.OccupancyMap` (raw DDS): encoded as a grayscale PNG (`-1` -> gray 128, probability -> `255-(p*255/100)`), flipped vertically (`flipud`) so the grid's row-0 origin ends up at the bottom of the image.
- **Running `slam_toolbox`**: `ros2 launch slam_toolbox online_async_launch.py slam_params_file:=controller/slam_toolbox_params.yaml use_sim_time:=false` - parameters (`slam_toolbox_params.yaml`) include `max_laser_range`/`min_laser_range` matched to `g1_isaac_sim_bridge.py`'s `LIDAR_MAX_RANGE`/`LIDAR_RANGE_MIN`, `scan_topic="/scan"`, `odom_frame="odom"`, etc.

### 2D lidar sensor

Not real hardware - a simulated sensor built from **Isaac Lab's `isaaclab.sensors.MultiMeshRayCaster`**, configured with `patterns.LidarPatternCfg` (`_attach_head_lidar`, `g1_isaac_sim_bridge.py`):

| Parameter | Value |
|---|---|
| Sensor class | `isaaclab.sensors.MultiMeshRayCaster` |
| Pattern | `isaaclab.sensors.ray_caster.patterns.LidarPatternCfg` |
| Channels | 1 (single ring, 2D scan) |
| Vertical FOV | 0.0 to 0.0 deg (single horizontal plane) |
| Horizontal FOV | -180 to 180 deg (full 360) |
| Horizontal resolution | 0.5 deg |
| Max range | 20 m (`LIDAR_MAX_RANGE`) |
| Min range | 0.1 m (`LIDAR_RANGE_MIN`) |
| Alignment | `ray_alignment="yaw"` - the scan plane stays level even as the torso pitches/rolls during gait |
| Mount | head (`d435_link` or head mount), Z+0.05m offset |
| Raycast targets | `/World/ground`, `/World/room/*` (the robot's own body is excluded - unlike a real sensor, self-occlusion isn't simulated) |

Designed to reproduce the same "single-plane 360-degree" scan pattern as real spinning 2D-SLAM lidars (e.g. RPLIDAR, Hokuyo-family units), but it does not model any specific manufacturer or model.

## Changelog by date

### 2026-08-10 - Initial web controller UI

- New Flask + Flask-SocketIO web server (`unitree_g1_web_controller_complete.py`): relays between the browser and the Isaac Sim process over raw DDS (CycloneDDS).
- New responsive web UI (`g1_web_ui_mobile_first.html`): LOCATION MAP (click the radar to set a target point), joysticks (MOVE/ROTATE pads), BASIC ACTIONS (Stand/Walk/Run/Home/Stop/Reset), MOTION (dance playback) panel. Responsive across desktop/mobile/tablet.
- New Isaac Sim bridge (`g1_isaac_sim_bridge.py`): loads G1, defines four control modes - `manual` (joystick CPG walking), `goto` (walk to a target point), `policy` (skrl AMP dance playback), `stand` - and the DDS topic contract (`g1_dds_types.py`).
- Mounted a forward-facing RGB camera (`isaaclab.sensors.Camera`) on the robot's head, independent of the registered task, streaming JPEG frames to the web UI's CAM FEED panel (`_attach_head_camera`/`_encode_camera_frame`, DDS topic `g1/camera_frame`). Can be disabled with `--no_camera`.

### 2026-08-14 - Head lidar + 2D SLAM map, PPO Stand policy integration

- Mounted a **2D lidar built on Isaac Lab's `MultiMeshRayCaster` + `patterns.LidarPatternCfg`** on the robot's head (single channel, 360 deg horizontal, 0.5 deg resolution, 20 m max range) (`_attach_head_lidar`, `g1_isaac_sim_bridge.py`). A simulated sensor matching real-hardware specs; also spawns a room with walls/obstacles (`_spawn_room`) for the scan to hit. Published on DDS topic `g1/lidar_scan` (`g1.LidarScan` in `g1_dds_types.py`). Can be disabled with `--no_lidar`.
- New `ros2_slam_bridge.py`: bridges `g1/lidar_scan` + `g1/robot_pose` (raw DDS) into `/scan` + `/tf` (ROS 2). Runs in a separate `ros2_slam` conda environment (Python 3.12), since `rclpy` isn't binary-compatible with the `isaac` conda environment (Python 3.11).
- New `slam_toolbox_params.yaml`: parameters for jazzy-toolbox's (`ros-jazzy-slam-toolbox`) `online_async_launch.py` - `max_laser_range`/`min_laser_range` matched to `g1_isaac_sim_bridge.py`'s `LIDAR_MAX_RANGE`/`LIDAR_RANGE_MIN`, `scan_topic="/scan"`, `odom_frame="odom"`, etc. The occupancy grid map `slam_toolbox` produces is relayed back over raw DDS (`g1.OccupancyMap`) to the web UI, where it renders as the LOCATION MAP background (updated only when a new area is explored).
- Wired up a PPO-based **Stand** policy for live inference: a `G1-PPO-Direct-Stand-v0` checkpoint (`--stand_policy_checkpoint`, defaulting to `logs/skrl/g1_stand/2026-08-13_02-36-26_ppo_torch/checkpoints/best_agent.pt`) is run every step against this script's own env/robot by `_StandPolicyController` - it starts automatically on load, and can also be triggered from the web UI via `policy_command` (`g1/policy_command` = `"play"`).
- Applied the trained Stand policy's joint pose/gain snapshot (`_TRAINED_STAND_POSE`/`_TRAINED_STAND_GAIN`, `_apply_trained_stand_defaults`) to the robot at startup, so `stand` mode and every `env.reset()` hold this trained pose.
- Wired the **Reset / Home** buttons to an explicit `env.reset()` trigger (`g1/home_position`, `g1/reset_sim` DDS topics) - the environment's own automatic reset-on-fall is disabled (`early_termination=False`), so a fallen robot can only recover via these buttons. `g1/emergency_stop` drops to `stand` immediately, without resetting.

### 2026-10-23 - Joystick and go-to-point (x, y) now run a locomotion policy

- Joystick (`manual`) and go-to-point (`goto`) driving both run a **velocity-tracking locomotion policy** trained in unitree_rl_lab (rsl_rl `exported/policy.pt` + `params/deploy.yaml`) (`_LocomotionPolicyController`, `g1_isaac_sim_bridge.py`). The old CPG walking controller is gone.
  - `--loco_policy` (the TorchScript policy) and `--loco_deploy_cfg` (deploy.yaml from the same run) are required. They default to `model/locomotion_policy.pt` and `model/locomotion_deploy.yaml` (copies of `exported/policy.pt` and `params/deploy.yaml` from unitree_g1_rl_lab's `logs/rsl_rl/finetune_yaw/2026-10-22_18-14-44`), and an empty value exits with an error before Isaac Sim boots.
  - The policy was trained on a different env (unitree_rl_lab's ManagerBasedRLEnv), so this script rebuilds its observation and action from deploy.yaml. Observation terms are scaled and history-stacked in deploy.yaml's order. The action `target = offset + scale * action` is converted into the AMP env's 29-dim action. The policy runs at 50 Hz (`step_dt` 0.02 s), and its target is held in between.
  - While the policy drives the robot, legs/feet/waist/arms gains and armature are switched to deploy.yaml's trained values. They are restored when it exits.
- **Entry transition**: switching from a standing pose straight into the walking policy topples the robot, so entry is staged.
  1. `interp` (`--loco_transition_time`, default 0.3 s): the policy already runs at zero velocity while the joint target and PD gains blend (smoothstep) from the last command to the policy's output. Verified in sim: 0.3 s drifts least, 0.6-1.0 s holds but drifts ~0.5 m, and 2.0 s falls.
  2. `warmup` (`--loco_warmup_time`, default 1.0 s): the policy alone at zero velocity. It advances only after tilt (|projected gravity xy| < 0.12) and speed (< 0.3 m/s) have stayed calm for 0.5 s.
  3. `active`: the velocity command is applied.
- **Joystick**
  - `g1/cmd_vel` is normalized to [-1, 1] stick deflection (full scale `linear_x` 2.0, `linear_y` 1.0, `angular_z` 1.0, deadzone 0.05). It is then scaled into the policy's trained command range, using the full travel both ways even when the range is asymmetric. The current policy's range is vx -0.5 to 1.0 m/s, vy ±0.5 m/s, and wz ±0.5 rad/s.
  - The command rate is limited (vx/vy 1.0 m/s², wz 1.5 rad/s²), so slamming the stick produces a ramp.
  - `linear_y` (strafing) is now used.
  - Releasing the stick keeps `manual` mode, with the policy stepping in place at zero velocity.
- **Go-to-point (Walk/Run)** (`_goto_velocity_command`, `drive_loco_to_target`)
  - The heading error to the target sets the yaw rate (gain 1.5, capped at the policy's trained range).
  - Forward speed fades as the heading error grows and is zero past 60°, so the robot turns in place first.
  - A distance-based speed cap (0.8 × distance) eases the robot into the target. Within 0.2 m it counts as arrived and hands over to `manual` at zero velocity (stepping in place).
  - Speed is 0.5 m/s for Walk by default and 1.4 m/s for Run. The web UI's target input gained a **V (m/s)** field: Walk uses the entered value (clamped to 0.1-1.4, blank = default), and Run always uses 1.4 m/s. `g1.GotoCommand` gained a matching `speed` field (0 = gait default).
  - Speed builds up at 0.3 m/s², and braking uses the joystick's rate.
  - Run's 1.4 m/s is a command value past the policy's trained 1.0 m/s cap; the robot tracks ~1.02 m/s actual. In sim, 1.0-1.4 never fell, and 1.6 fell 1 of 2 times.
- **Stand button**: new `g1/stand_command` topic and a matching `stand` event on the web server.
  - While walking, the robot isn't cut off mid-stride. The command ramps to zero, and once speed stays under 0.15 m/s for 0.5 s (or 4 s after the command reached zero), the robot is handed to the Stand PPO policy. The hand-over blends over `--loco_handover_time` (default 0.5 s). In sim, a direct switch tilted ~30°, and 1.0 s or longer fell.
  - Cancelling a goto (`g1/goto_cancel`) takes the same path.
  - Without `--stand_policy_checkpoint` there is no policy to hand over to, so the walking policy at zero velocity serves as standing.
- `g1/status` gained `stand_policy`, `loco` (`phase`, plus `cmd`, the velocity command the policy is actually following), `goto_speed` and `camera_error`. The web UI uses them to show `Standing` / `Stopping` / `Walking in place`.
- Model weights now live in `g1_isaac/model/`. The Stand policy (`--stand_policy_checkpoint`) default moved from `logs/skrl/g1_stand/2026-08-13_02-36-26_ppo_torch/checkpoints/best_agent.pt` to its copy `model/stand_policy.pt`, and the walking policy reads its files from `model/` as above (the old `--loco_policy_dir` is gone).
