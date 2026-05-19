import logging
from telemetry_types import CameraFrame


class CameraFrameProvider:
    def capture_frame(self, frame_time: float) -> CameraFrame:
        logging.debug("Camera frame capture placeholder at %.6f", frame_time)  # TEMPORARY
        return CameraFrame(frame_time=frame_time, data=b"")
