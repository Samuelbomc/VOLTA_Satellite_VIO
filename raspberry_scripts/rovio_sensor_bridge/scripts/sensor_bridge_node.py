#!/usr/bin/env python
import os
import sys
import threading
import time

import rospy
from sensor_msgs.msg import Imu, Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import serial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
RASPBERRY_SCRIPTS_DIR = os.path.join(REPO_ROOT, "raspberry_scripts")
if RASPBERRY_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, RASPBERRY_SCRIPTS_DIR)

from camera_receptor import CameraTelemetryNode
from telemetry_types import ImuPacket
from camera_frame_provider import CameraFrameProvider


class RosTelemetryProcessor(object):
    def __init__(self, imu_pub, image_pub, bridge, camera_provider, params):
        self.imu_pub = imu_pub
        self.image_pub = image_pub
        self.bridge = bridge
        self.camera_provider = camera_provider
        self.imu_frame_id = params["imu_frame_id"]
        self.camera_frame_id = params["camera_frame_id"]
        self.image_width = params["image_width"]
        self.image_height = params["image_height"]
        self.input_encoding = params["image_encoding"]
        self.force_mono8 = params["force_mono8"]
        self.timestamp_skew_warn_s = params["timestamp_skew_warn_s"]
        self.use_latest_imu_time_for_frame = params["use_latest_imu_time_for_frame"]
        self._lock = threading.Lock()
        self._last_imu_stamp = None

    def handle_imu_packet(self, recv_time, data):
        packet = ImuPacket.from_tuple(recv_time, data)
        stamp = rospy.Time.from_sec(packet.timestamp_ms / 1000.0)

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self.imu_frame_id

        msg.angular_velocity.x = packet.gx
        msg.angular_velocity.y = packet.gy
        msg.angular_velocity.z = packet.gz

        msg.linear_acceleration.x = packet.ax
        msg.linear_acceleration.y = packet.ay
        msg.linear_acceleration.z = packet.az + 9.81

        self.imu_pub.publish(msg)

        with self._lock:
            self._last_imu_stamp = stamp.to_sec()

    def get_last_imu_stamp(self):
        with self._lock:
            return self._last_imu_stamp

    def publish_camera_frame(self, frame, stamp):
        if not frame or not frame.data:
            rospy.logdebug("Skipping empty camera frame")
            return

        image_msg = self._build_image_message(frame, stamp)
        if image_msg is None:
            return

        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.camera_frame_id
        self.image_pub.publish(image_msg)

    def _build_image_message(self, frame, stamp):
        channels = 1 if self.input_encoding == "mono8" else 3
        expected_size = self.image_width * self.image_height * channels

        if self.image_width <= 0 or self.image_height <= 0:
            rospy.logwarn_throttle(5.0, "Invalid image dimensions %dx%d", self.image_width, self.image_height)
            return None

        if len(frame.data) < expected_size:
            rospy.logwarn_throttle(
                5.0,
                "Image data size mismatch: expected %d bytes, got %d",
                expected_size,
                len(frame.data),
            )
            return None

        raw_array = np.frombuffer(frame.data, dtype=np.uint8, count=expected_size)
        if channels == 1:
            image = raw_array.reshape((self.image_height, self.image_width))
            output_encoding = "mono8"
        else:
            image = raw_array.reshape((self.image_height, self.image_width, channels))
            output_encoding = self.input_encoding

        if self.force_mono8 and output_encoding != "mono8":
            if output_encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif output_encoding == "bgr8":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                rospy.logwarn_throttle(5.0, "Unsupported encoding for mono conversion: %s", output_encoding)
                return None
            output_encoding = "mono8"

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding=output_encoding)
        last_imu_stamp = self.get_last_imu_stamp()
        if last_imu_stamp is not None:
            skew = abs(last_imu_stamp - stamp.to_sec())
            if skew > self.timestamp_skew_warn_s:
                rospy.logwarn_throttle(
                    2.0,
                    "Image/IMU timestamp skew %.6fs exceeds threshold %.6fs",
                    skew,
                    self.timestamp_skew_warn_s,
                )

        return image_msg


class RosSensorBridgeNode(object):
    def __init__(self, params):
        self.params = params
        self.bridge = CvBridge()
        self.camera_provider = CameraFrameProvider()

        self.imu_pub = rospy.Publisher("/imu", Imu, queue_size=200)
        self.image_pub = rospy.Publisher("/cam0/image_raw", Image, queue_size=10)

        self.processor = RosTelemetryProcessor(
            self.imu_pub,
            self.image_pub,
            self.bridge,
            self.camera_provider,
            params,
        )

        self.telemetry_node = CameraTelemetryNode(
            params["serial_port"],
            params["baud_rate"],
            params["serial_timeout"],
        )
        self.telemetry_node.processor = self.processor

        self._imu_thread = threading.Thread(target=self._imu_loop)
        self._imu_thread.daemon = True

        self._camera_thread = threading.Thread(target=self._camera_loop)
        self._camera_thread.daemon = True

    def start(self):
        self.telemetry_node.connect()
        self._imu_thread.start()
        self._camera_thread.start()

    def shutdown(self):
        self.telemetry_node.is_running = False
        self.telemetry_node.close()

    def _imu_loop(self):
        while not rospy.is_shutdown() and self.telemetry_node.is_running:
            status = self.telemetry_node.listen_and_decode()
            if status in {"SYNC_LOST", "TIMEOUT", "ERROR_LENGTH", "ERROR_FLUSH", "ERROR_HEADER", "ERROR_CRC"}:
                rospy.logdebug("Serial decode status: %s", status)

    def _camera_loop(self):
        rate = rospy.Rate(self.params["image_rate"])
        while not rospy.is_shutdown() and self.telemetry_node.is_running:
            stamp = self._select_camera_stamp()
            frame = self.camera_provider.capture_frame(stamp.to_sec())
            self.processor.publish_camera_frame(frame, stamp)
            rate.sleep()

    def _select_camera_stamp(self):
        if self.params["use_latest_imu_time_for_frame"]:
            last_imu_stamp = self.processor.get_last_imu_stamp()
            if last_imu_stamp is not None:
                return rospy.Time.from_sec(last_imu_stamp)
        return rospy.Time.from_sec(rospy.get_time())


def _load_params():
    return {
        "serial_port": rospy.get_param("~serial_port", "/dev/serial0"),
        "baud_rate": rospy.get_param("~baud_rate", 115200),
        "serial_timeout": rospy.get_param("~serial_timeout", 0.01),
        "imu_frame_id": rospy.get_param("~imu_frame_id", "imu_link"),
        "camera_frame_id": rospy.get_param("~camera_frame_id", "camera_frame"),
        "image_rate": rospy.get_param("~image_rate", 20.0),
        "image_width": rospy.get_param("~image_width", 640),
        "image_height": rospy.get_param("~image_height", 480),
        "image_encoding": rospy.get_param("~image_encoding", "mono8"),
        "force_mono8": rospy.get_param("~force_mono8", True),
        "timestamp_skew_warn_s": rospy.get_param("~timestamp_skew_warn_s", 0.005),
        "use_latest_imu_time_for_frame": rospy.get_param("~use_latest_imu_time_for_frame", True),
    }


def main():
    rospy.init_node("rovio_sensor_bridge", anonymous=False)
    params = _load_params()
    node = RosSensorBridgeNode(params)

    rospy.on_shutdown(node.shutdown)

    try:
        node.start()
        rospy.loginfo("ROVIO sensor bridge started")
        rospy.spin()
    except serial.SerialException as error:
        rospy.logerr("Serial error: %s", error)
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
