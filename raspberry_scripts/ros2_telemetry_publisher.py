import logging
from telemetry_types import ImuPacket, CameraFrame


class Ros2TelemetryPublisher:
    def __init__(self):
        self.is_ready = False

    def initialize(self):
        self.is_ready = True

    def publish_imu(self, packet: ImuPacket):
        logging.debug("ROS2 IMU publish placeholder: %s", packet)  # TEMPORARY

    def publish_frame(self, frame: CameraFrame):
        logging.debug("ROS2 frame publish placeholder: %s", frame)  # TEMPORARY
