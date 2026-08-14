#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 (Jazzy) <-> raw-DDS translation bridge for the head lidar / SLAM pipeline.

Sits between ``g1_isaac_sim_bridge.py`` (Isaac Sim, conda ``isaac`` env, raw CycloneDDS only - see
``g1_dds_types.py``'s own docstring for why) and ``slam_toolbox`` (a normal ROS 2 node). Rather than
hand-matching ROS 2's DDS wire format from the conda side, this one small process links both
libraries in a single Python interpreter and translates in plain Python:

* raw DDS (this project's ``g1/lidar_scan`` + ``g1/robot_pose`` topics) -> ROS 2 (``/scan`` +
  ``odom -> base_link`` tf), which ``slam_toolbox`` consumes directly.
* ROS 2's ``/map`` (published by ``slam_toolbox``) -> raw DDS (``g1/occupancy_map``, PNG+base64,
  same wire pattern as ``g1/camera_frame``) -> ``unitree_g1_web_controller_complete.py`` -> browser.

**Must run under a Python that matches Jazzy's rclpy build exactly (3.12 on this machine's apt
install), not the ``isaac`` conda env used by the sim bridge** (that one's pinned to 3.11 for Isaac
Lab compatibility, and rclpy's C extension is a version-specific compiled ``.so`` - importing it
from a mismatched interpreter fails with ``ModuleNotFoundError: No module named
'rclpy._rclpy_pybind11'``). A plain conda env pinned to 3.12 works fine here (verified live on this
machine, including the full rclpy node/DDS lifecycle, not just importability) and is simpler to set
up than a system-Python venv - no ``python3.12-venv``/``ensurepip`` package needed:

.. code-block:: bash

    conda create -n ros2_slam python=3.12 -y   # once
    conda activate ros2_slam
    pip install cyclonedds pillow pyyaml numpy  # once - pyyaml/numpy are rclpy's own runtime deps,
                                                 # not provided by a bare conda env the way they
                                                 # would be by a --system-site-packages venv

    source /opt/ros/jazzy/setup.bash
    conda activate ros2_slam
    export ROS_DOMAIN_ID=42   # distinct from --dds_domain_id (default 0) - two unrelated DDS
                               # networks (this bridge's raw-DDS side vs. rclpy's own ROS 2 DDS
                               # traffic to slam_toolbox) shouldn't share one multicast domain
    python3 controller/ros2_slam_bridge.py

    # separate terminal, same environment as above:
    ros2 launch slam_toolbox online_async_launch.py \\
        slam_params_file:=controller/slam_toolbox_params.yaml use_sim_time:=false
"""

import argparse
import base64
import io

import numpy as np
import rclpy
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from g1_dds_types import (
    TOPIC_LIDAR_SCAN,
    TOPIC_OCCUPANCY_MAP,
    TOPIC_ROBOT_POSE,
    LidarScan,
    OccupancyMap,
    RobotPose,
    dds_listener,
    make_participant,
)


class Ros2SlamBridge(Node):
    def __init__(self, dds_domain_id: int):
        super().__init__("g1_ros2_slam_bridge")

        self._scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._map_sub = self.create_subscription(OccupancyGrid, "/map", self._on_map, 10)

        self.participant = make_participant(dds_domain_id)
        self._scan_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_LIDAR_SCAN, LidarScan),
            listener=dds_listener(self._on_lidar_scan),
        )
        self._pose_reader = DataReader(
            self.participant,
            Topic(self.participant, TOPIC_ROBOT_POSE, RobotPose),
            listener=dds_listener(self._on_robot_pose),
        )
        self._map_writer = DataWriter(self.participant, Topic(self.participant, TOPIC_OCCUPANCY_MAP, OccupancyMap))

        self.get_logger().info(
            f"g1_ros2_slam_bridge ready - raw DDS domain {dds_domain_id} <-> ROS 2 /scan, /map, tf"
        )

    def _on_lidar_scan(self, msg: LidarScan) -> None:
        """g1.LidarScan -> sensor_msgs/LaserScan. Field layout matches 1:1 by construction (see
        LidarScan's docstring in g1_dds_types.py), no resampling needed. frame_id is "base_link"
        directly (not a separate "laser_link" + static mount-offset transform): slam_toolbox's 2D
        map is flat in XY, and the lidar is mounted directly above the root/pelvis, so the only
        mount offset is in Z - irrelevant to a 2D scan-to-map pipeline."""
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "base_link"
        scan.angle_min = msg.angle_min
        scan.angle_max = msg.angle_max
        scan.angle_increment = msg.angle_increment
        scan.range_min = msg.range_min
        scan.range_max = msg.range_max
        scan.ranges = list(msg.ranges)
        self._scan_pub.publish(scan)

    def _on_robot_pose(self, msg: RobotPose) -> None:
        """g1.RobotPose (sim ground truth) -> odom -> base_link tf, treating the sim's own pose as
        perfect odometry. slam_toolbox publishes map -> odom itself; this is the other half of the
        chain it needs."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = msg.x
        t.transform.translation.y = msg.y
        t.transform.translation.z = msg.z
        # g1.RobotPose is (qw, qx, qy, qz); geometry_msgs/Quaternion is (x, y, z, w)
        t.transform.rotation.x = msg.qx
        t.transform.rotation.y = msg.qy
        t.transform.rotation.z = msg.qz
        t.transform.rotation.w = msg.qw
        self._tf_broadcaster.sendTransform(t)

    def _on_map(self, msg: OccupancyGrid) -> None:
        """nav_msgs/OccupancyGrid -> g1.OccupancyMap (grayscale PNG, mirrors g1.CameraFrame's JPEG
        pattern so the web relay/browser need no grid-decoding logic). Standard OccupancyGrid
        convention: -1 unknown, 0-100 free->occupied probability - mapped to gray/white->black.
        Row 0 of the grid is the origin (bottom-left); image rows go top-to-bottom, hence the flip.
        """
        width, height = msg.info.width, msg.info.height
        if width == 0 or height == 0:
            return
        cells = np.array(msg.data, dtype=np.int16).reshape((height, width))
        img = np.where(cells < 0, 128, np.clip(255 - (cells * 255 // 100), 0, 255)).astype(np.uint8)
        img = np.flipud(img)
        buf = io.BytesIO()
        Image.fromarray(img, mode="L").save(buf, format="PNG")
        self._map_writer.write(
            OccupancyMap(
                width=width,
                height=height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y,
                png_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS 2 <-> raw-DDS bridge for g1_isaac's lidar/SLAM pipeline.")
    parser.add_argument(
        "--dds_domain_id",
        type=int,
        default=0,
        help="Raw CycloneDDS domain id - must match g1_isaac_sim_bridge.py's/the web controller's --dds_domain_id.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Ros2SlamBridge(args.dds_domain_id)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
