import logging

from telemetry_types import ImuPacket
from ros2_telemetry_publisher import Ros2TelemetryPublisher
from camera_frame_provider import CameraFrameProvider


class TelemetryProcessor:
    def __init__(self):
        self.ros_publisher = Ros2TelemetryPublisher()
        self.camera_provider = CameraFrameProvider()

    def handle_imu_packet(self, recv_time: float, data: tuple):
        packet = ImuPacket.from_tuple(recv_time, data)
        logging.debug("TelemetryProcessor received IMU packet: %s", packet)  # TEMPORARY
        self.ros_publisher.publish_imu(packet)

    def handle_camera_frame(self, frame_time: float):
        frame = self.camera_provider.capture_frame(frame_time)
        self.ros_publisher.publish_frame(frame)
