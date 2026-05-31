import csv
import logging
import struct
import time
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rclpy
import serial
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from pathlib import Path

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)

SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.02
READ_MAX_TIME = 0.1

SYNC_1 = 0xAA
SYNC_2 = 0x55
ID_CAM_TP = 0x11
ID_INIT_CMD = 0xF0

LEN_CAM_TP = 32
LEN_INIT_CMD = 1
MAX_PAYLOAD = 64

FRAME_INTERVAL_S = 1.0 / 30.0
IMAGE_WIDTH = 1456
IMAGE_HEIGHT = 1088


# ==========================================
# DATA TYPES
# ==========================================
@dataclass(frozen=True)
class ImuPacket:
    recv_time: float
    timestamp_ms: int
    baro_msl_m: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    @staticmethod
    def from_tuple(
        recv_time: float,
        data: Tuple[int, float, float, float, float, float, float, float],
    ) -> "ImuPacket":
        timestamp_ms, baro_msl_m, ax, ay, az, gx, gy, gz = data
        return ImuPacket(
            recv_time=recv_time,
            timestamp_ms=timestamp_ms,
            baro_msl_m=baro_msl_m,
            ax=ax,
            ay=ay,
            az=az,
            gx=gx,
            gy=gy,
            gz=gz,
        )

@dataclass(frozen=True)
class ImuSample:
    timestamp_s: float
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray

@dataclass(frozen=True)
class BaroSample:
    timestamp_s: float
    altitude_msl_m: float

@dataclass(frozen=True)
class FrameSample:
    timestamp_s: float
    image_bgr: np.ndarray

@dataclass(frozen=True)
class StateEstimate:
    timestamp_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    attitude_quat: np.ndarray
    agl_m: float

@dataclass(frozen=True)
class CameraFrame:
    frame_time: float
    data: Any


# ==========================================
# ESTIMATOR LOGIC
# ==========================================
class OpticalFlowTracker:
    """Tracks features and computes optical flow vectors between frames."""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: Tuple[int, int]):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_size = image_size
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal feature tracking state when frames are lost."""
        self.prev_gray = None
        self.prev_kp = None

    def _to_gray(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=12,
            blockSize=7,
        )

    def compute_flow(self, frame: FrameSample) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optical flow vectors (pixels) for the current frame."""
        gray = self._to_gray(frame.image_bgr)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            LOGGER.debug("TEMP: initial flow features: %s", 0 if self.prev_kp is None else len(self.prev_kp))  # TEMP log
            return np.empty((0, 2)), np.empty((0, 2))

        if self.prev_kp is None or len(self.prev_kp) < 20:
            self.prev_kp = self._detect_features(self.prev_gray)
            if self.prev_kp is None:
                self.prev_gray = gray
                LOGGER.debug("TEMP: no features detected")  # TEMP log
                return np.empty((0, 2)), np.empty((0, 2))

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_kp,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            LOGGER.debug("TEMP: optical flow failure")  # TEMP log
            return np.empty((0, 2)), np.empty((0, 2))

        status = status.reshape(-1) == 1
        prev_pts = self.prev_kp.reshape(-1, 2)[status]
        curr_pts = next_pts.reshape(-1, 2)[status]

        self.prev_gray = gray
        self.prev_kp = curr_pts.reshape(-1, 1, 2) if len(curr_pts) else self._detect_features(gray)

        if self.prev_kp is not None and len(self.prev_kp) < 20:
            self.prev_kp = self._detect_features(gray)

        LOGGER.debug("TEMP: tracked features: %d", len(curr_pts))  # TEMP log

        return prev_pts, curr_pts


class GyroDeRotation:
    """Compensates rotational flow using integrated gyro measurements."""

    def __init__(self):
        self.last_imu_time: Optional[float] = None
        self.last_omega = np.zeros(3)
        # Rotation matrix from camera to IMU (empirical). Kept for real-world mapping.
        self.r_cam_to_imu = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)

    def reset(self) -> None:
        """Clear stored IMU integration window."""
        self.last_imu_time = None

    def integrate_gyro(self, imu: ImuSample) -> None:
        """Integrate gyroscope measurements to accumulate delta rotation."""
        self.last_imu_time = imu.timestamp_s
        self.last_omega = imu.gyro_rad_s

    def imu_to_camera(self, omega_imu: np.ndarray) -> np.ndarray:
        # Convert angular rate from IMU frame to camera frame
        return self.r_cam_to_imu.T @ omega_imu

    def compensate_flow(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        omega_rad_s: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subtract rotational optical flow given accumulated delta rotation."""
        if omega_rad_s is None or dt <= 0.0:
            return pts_prev, pts_curr

        omega_x, omega_y, omega_z = omega_rad_s
        x = pts_prev[:, 0]
        y = pts_prev[:, 1]

        dx_rot = (-(1.0 + x * x) * omega_y + x * y * omega_x + y * omega_z) * dt
        dy_rot = (-(x * y) * omega_y + (1.0 + y * y) * omega_x - x * omega_z) * dt

        pts_curr_comp = np.column_stack((pts_curr[:, 0] - dx_rot, pts_curr[:, 1] - dy_rot))
        return pts_prev, pts_curr_comp


class BaroAltitudeModel:
    """Calculates AGL from barometer using a fixed MSL reference."""

    def __init__(self, reference_msl_m: float):
        self.reference_msl_m = reference_msl_m

    def agl_from_baro(self, baro: BaroSample) -> float:
        """Compute above-ground-level height from barometer MSL reading."""
        agl_m = baro.altitude_msl_m - self.reference_msl_m
        return max(agl_m, 0.1)


class RobustFlowImuBaroEstimator:
    """Estimator combining de-rotated optical flow, IMU integration and barometer.

    Adapted from the Gazebo SimpleEstimator: retains diagnostics (logs),
    low-cost looming compensation, gyro de-rotation and a simple EMA filter.
    Uses fisheye undistortion (real-world camera) as in the original main node.
    """

    def __init__(self, reference_msl_m: float = 916.0):
        # Real-world calibration kept from main_node
        if hasattr(self, 'camera_matrix'):
            pass
        self.camera_matrix = np.array(
            [[441.4601, 0.0, 670.8540], [0.0, 441.7088, 514.5791], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.array([-0.01209933, -0.00455076, 0.00472093, -0.00210166], dtype=np.float64)

        self.flow = OpticalFlowTracker(self.camera_matrix, self.dist_coeffs, (IMAGE_WIDTH, IMAGE_HEIGHT))
        self.derotation = GyroDeRotation()
        self.baro_model = BaroAltitudeModel(reference_msl_m)
        self.state: Optional[StateEstimate] = None

        self.last_frame_time: Optional[float] = None
        self.last_baro: Optional[BaroSample] = None
        self.imu_buffer: list[ImuSample] = []

        # Diagnostics/logging container (exposed for CSV output)
        self.logs = {}

    def ingest_imu(self, imu: ImuSample) -> None:
        """Buffer IMU samples for later integration over the frame interval."""
        self.imu_buffer.append(imu)

    def ingest_baro(self, baro: BaroSample) -> None:
        """Store last barometer reading used for altitude-based scaling."""
        self.last_baro = baro

    def ingest_frame(self, frame: FrameSample) -> None:
        """Process a camera frame and update internal state with diagnostics.

        Steps:
        - compute average gyro in the frame window
        - compute optical flow and undistort (fisheye)
        - de-rotate flow using gyro
        - compensate for looming (vertical motion)
        - compute velocity in camera frame and map to body
        - low-pass blend into state
        """
        # reset diagnostics
        self.logs = {k: 0.0 for k in [
            'dt','features','omega_cx','omega_cy','omega_cz',
            'dx_raw','dy_raw','dx_derot','dy_derot','dx_comp','dy_comp',
            'vz_baro','vx_raw','vy_raw'
        ]}

        if self.last_frame_time is None:
            self.last_frame_time = frame.timestamp_s
            self.flow.reset()
            self.flow.compute_flow(frame)
            return

        dt = frame.timestamp_s - self.last_frame_time
        if dt <= 0.0:
            self.last_frame_time = frame.timestamp_s
            return

        self.logs['dt'] = dt

        if self.last_baro is None:
            return
        z_agl = self.baro_model.agl_from_baro(self.last_baro)
        if z_agl <= 0.0:
            return

        # collect IMU samples within the last-frame -> current-frame window
        imu_window = [imu for imu in self.imu_buffer if self.last_frame_time < imu.timestamp_s <= frame.timestamp_s]
        self.imu_buffer = [imu for imu in self.imu_buffer if imu.timestamp_s > frame.timestamp_s]
        if not imu_window:
            self.last_frame_time = frame.timestamp_s
            return

        omega_avg = np.mean([imu.gyro_rad_s for imu in imu_window], axis=0)
        omega_cam = self.derotation.imu_to_camera(omega_avg)
        self.logs['omega_cx'], self.logs['omega_cy'], self.logs['omega_cz'] = omega_cam.tolist()

        pts_prev, pts_curr = self.flow.compute_flow(frame)
        self.last_frame_time = frame.timestamp_s

        if pts_prev.size == 0 or pts_curr.size == 0:
            return

        self.logs['features'] = float(len(pts_curr))

        # Use fisheye undistortions (real-world camera model)
        try:
            pts_prev_norm = cv2.fisheye.undistortPoints(pts_prev.reshape(-1,1,2), self.camera_matrix, self.dist_coeffs).reshape(-1,2)
            pts_curr_norm = cv2.fisheye.undistortPoints(pts_curr.reshape(-1,1,2), self.camera_matrix, self.dist_coeffs).reshape(-1,2)
        except Exception:
            pts_prev_norm = pts_prev.astype(np.float64)
            pts_curr_norm = pts_curr.astype(np.float64)

        raw_flow = pts_curr_norm - pts_prev_norm
        self.logs['dx_raw'] = float(np.median(raw_flow[:,0]))
        self.logs['dy_raw'] = float(np.median(raw_flow[:,1]))

        # De-rotate flow using averaged gyro
        pts_prev_norm, pts_curr_norm = self.derotation.compensate_flow(pts_prev_norm, pts_curr_norm, omega_cam, dt)
        derot_flow = pts_curr_norm - pts_prev_norm
        self.logs['dx_derot'] = float(np.median(derot_flow[:,0]))
        self.logs['dy_derot'] = float(np.median(derot_flow[:,1]))

        # Looming compensation (vertical motion scale)
        vz_baro = 0.0 if self.state is None else (z_agl - self.state.agl_m) / dt
        self.logs['vz_baro'] = vz_baro

        x_norm = pts_prev_norm[:,0]
        y_norm = pts_prev_norm[:,1]
        dx_trans = derot_flow[:,0] + (x_norm * vz_baro * dt / z_agl)
        dy_trans = derot_flow[:,1] + (y_norm * vz_baro * dt / z_agl)

        self.logs['dx_comp'] = float(np.median(dx_trans))
        self.logs['dy_comp'] = float(np.median(dy_trans))

        # Compute camera velocity and rotate to body frame
        v_cam = np.array([-(self.logs['dx_comp'] * z_agl) / dt, -(self.logs['dy_comp'] * z_agl) / dt, 0.0])
        v_body = self.derotation.r_cam_to_imu @ v_cam
        vx_raw, vy_raw = float(v_body[0]), float(v_body[1])

        self.logs['vx_raw'] = vx_raw
        self.logs['vy_raw'] = vy_raw

        # Low-pass blend for stability
        alpha = float(np.clip(100.0 / z_agl, 0.05, 0.8))

        if self.state is None:
            vx, vy = vx_raw, vy_raw
            position = np.zeros(3, dtype=np.float64)
            velocity = np.array([vx, vy, vz_baro], dtype=np.float64)
            attitude = np.array([0.0,0.0,0.0,1.0], dtype=np.float64)
        else:
            vx = self.state.velocity_m_s[0] * (1.0 - alpha) + vx_raw * alpha
            vy = self.state.velocity_m_s[1] * (1.0 - alpha) + vy_raw * alpha
            velocity = np.array([vx, vy, vz_baro], dtype=np.float64)
            position = self.state.position_m + velocity * dt
            attitude = self.state.attitude_quat if hasattr(self.state, 'attitude_quat') else np.array([0.0,0.0,0.0,1.0], dtype=np.float64)

        self.state = StateEstimate(
            timestamp_s=frame.timestamp_s,
            position_m=position,
            velocity_m_s=velocity,
            attitude_quat=attitude,
            agl_m=z_agl,
        )

    def current_state(self) -> Optional[StateEstimate]:
        return self.state


# ==========================================
# ROS 2 NODE AND MAIN LOOP
# ==========================================
class FlowEstimatorNode(Node):
    def __init__(self, port: str, baud_rate: int, timeout: float):
        super().__init__('flow_imu_baro_node')
        self.port_name = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None

        self.estimator = RobustFlowImuBaroEstimator()
        self.bridge = CvBridge()
        
        self.imu_pub = self.create_publisher(Imu, '/imu0', 100)
        self.cam_pub = self.create_publisher(Image, '/cam0/image_raw', 10)
        self.state_pub = self.create_publisher(Odometry, '/odom', 10)

        self.is_running = False
        self.last_init_time = 0.0
        self.last_imu_time = time.monotonic()
        self.last_watchdog_check = time.monotonic()
        self.last_frame_capture = time.monotonic()
        self.init_received = False
        self.last_baro_msl: Optional[float] = None
        self.has_imu_data = False

        # Diagnostics: always enabled (write full logs)
        self.enable_diagnostics = True
        # Maximum number of corrected images to save
        self.max_saved_corrected = 1000

        # Results directory for raw / corrected frames
        self.results_dir = Path('results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.saved_corrected_count = 0
        # track last time we wrote images to results/ (save every 2 seconds)
        self.last_results_save_time = 0.0
        LOGGER.debug("TEMP: results directory initialized at %s", str(self.results_dir))  # TEMP log

        # CSV: always include extended diagnostics fields
        self.csv_file = open('results.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Extended diagnostic header (matches estimator.logs keys)
        header = [
            'timestamp', 'dt', 'pos_x', 'pos_y', 'agl',
            'vel_x_filt', 'vel_y_filt', 'vel_x_raw', 'vel_y_raw', 'vz_baro',
            'features', 'dx_raw', 'dy_raw', 'dx_derot', 'dy_derot', 'dx_comp', 'dy_comp',
            'omega_cx', 'omega_cy', 'omega_cz',
            'imu_ax', 'imu_ay', 'imu_az'
        ]
        self.csv_writer.writerow(header)


    def connect(self) -> None:
        self.serial_conn = serial.Serial(
            port=self.port_name,
            baudrate=self.baud_rate,
            timeout=self.timeout,
        )
        self.is_running = True
        self.last_imu_time = time.monotonic()
        LOGGER.info("TEMP: Serial connected")  # TEMP log
        LOGGER.info("Sistema de vuelo iniciado en %s a %d baud", self.port_name, self.baud_rate)

    def close(self) -> None:
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        LOGGER.info("TEMP: shutdown complete")  # TEMP log
        LOGGER.info("Apagado completado")

    @staticmethod
    def calculate_crc16(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021)
                else:
                    crc = (crc << 1)
                crc &= 0xFFFF
        return crc

    def read_exact_bytes(self, n_bytes: int, max_time: float = READ_MAX_TIME) -> Optional[bytes]:
        start_time = time.monotonic()
        data = b''
        while len(data) < n_bytes:
            if (time.monotonic() - start_time) > max_time:
                LOGGER.debug("TEMP: read_exact_bytes timeout after %.3f s", max_time)  # TEMP log
                return None
            chunk = self.serial_conn.read(n_bytes - len(data))
            if chunk:
                data += chunk
        LOGGER.debug("TEMP: read_exact_bytes success (%d bytes)", len(data))  # TEMP log
        return data

    def wait_for_sync(self, max_attempts: int = 2000) -> bool:
        state = 0
        attempts = 0
        while attempts < max_attempts:
            chunk = self.serial_conn.read(1)
            if not chunk:
                LOGGER.debug("TEMP: wait_for_sync read empty chunk")  # TEMP log
                return False
            b = chunk[0]
            attempts += 1
            if state == 0:
                if b == SYNC_1:
                    state = 1
                    LOGGER.debug("TEMP: saw SYNC_1")  # TEMP log
            elif state == 1:
                if b == SYNC_2:
                    LOGGER.debug("TEMP: sync established")  # TEMP log
                    return True
                if b == SYNC_1:
                    state = 1
                else:
                    state = 0
        LOGGER.debug("TEMP: wait_for_sync attempts exhausted")  # TEMP log
        return False

    def handle_buffer_overflow(self) -> None:
        in_waiting = self.serial_conn.in_waiting
        if in_waiting > 128:
            LOGGER.warning("TEMP: buffer overflow (%d bytes). Resetting input buffer.", in_waiting)  # TEMP log
            self.serial_conn.reset_input_buffer()
        elif in_waiting > 0:
            LOGGER.debug("TEMP: small data remaining in buffer: %d bytes", in_waiting)  # TEMP log
            self.serial_conn.read(1)

    def send_init_response(self) -> None:
        payload = bytes([ID_CAM_TP])
        payload_len = len(payload)
        data_for_crc = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, payload_len]) + payload
        crc_calculated = self.calculate_crc16(data_for_crc)
        crc_bytes = struct.pack('<H', crc_calculated)
        packet = bytes([SYNC_1, SYNC_2, ID_INIT_CMD, payload_len]) + payload + crc_bytes
        self.serial_conn.write(packet)
        self.serial_conn.flush()
        LOGGER.debug("TEMP: init response sent")  # TEMP log

    def listen_and_decode(self) -> str:
        if not self.wait_for_sync():
            LOGGER.debug("TEMP: sync not found")  # TEMP log
            return "SYNC_LOST"

        header = self.read_exact_bytes(2)
        if not header:
            LOGGER.debug("TEMP: header timeout")  # TEMP log
            return "TIMEOUT"

        packet_id = header[0]
        payload_len = header[1]

        if payload_len > MAX_PAYLOAD:
            LOGGER.debug("TEMP: payload too large: %d", payload_len)  # TEMP log
            self.handle_buffer_overflow()
            return "ERROR_LENGTH"

        if not self.init_received and packet_id != ID_INIT_CMD:
            LOGGER.debug("TEMP: waiting for init")  # TEMP log
            flushed = self.read_exact_bytes(payload_len + 2)
            if not flushed:
                self.handle_buffer_overflow()
                return "ERROR_FLUSH"
            return "WAIT_INIT"

        is_valid_header = False
        if packet_id == ID_CAM_TP and payload_len == LEN_CAM_TP:
            is_valid_header = True
        elif packet_id == ID_INIT_CMD and payload_len == LEN_INIT_CMD:
            is_valid_header = True

        if not is_valid_header:
            LOGGER.debug("TEMP: invalid header id=0x%02X len=%d", packet_id, payload_len)  # TEMP log
            flushed = self.read_exact_bytes(payload_len + 2)
            if not flushed:
                self.handle_buffer_overflow()
                return "ERROR_FLUSH"
            return "ERROR_HEADER"

        payload = self.read_exact_bytes(payload_len)
        if not payload:
            LOGGER.debug("TEMP: payload timeout")  # TEMP log
            return "TIMEOUT"

        crc_bytes = self.read_exact_bytes(2)
        if not crc_bytes:
            LOGGER.debug("TEMP: crc timeout")  # TEMP log
            return "TIMEOUT"

        received_crc = struct.unpack('<H', crc_bytes)[0]
        data_for_crc = bytes([SYNC_1, SYNC_2, packet_id, payload_len]) + payload

        if self.calculate_crc16(data_for_crc) != received_crc:
            LOGGER.debug("TEMP: crc mismatch")  # TEMP log
            return "ERROR_CRC"

        if packet_id == ID_CAM_TP:
            if not self.init_received:
                LOGGER.debug("TEMP: IMU received before init")  # TEMP log
                return "IMU_BEFORE_INIT"
            data = struct.unpack('<I f 6f', payload)
            recv_time = time.monotonic()

            self.last_imu_time = recv_time
            self.handle_imu_packet(recv_time, data)
            LOGGER.debug("TEMP: imu packet processed")  # TEMP log

            return "IMU_OK"

        if packet_id == ID_INIT_CMD:
            target_id = payload[0]
            current_time = time.monotonic()

            if current_time - self.last_init_time < 2.0:
                LOGGER.debug("TEMP: init spam")  # TEMP log
                return "INIT_SPAM"

            self.last_init_time = current_time
            if target_id == ID_CAM_TP:
                self.send_init_response()
                self.init_received = True
                LOGGER.debug("TEMP: init acknowledged")  # TEMP log
            return "INIT_OK"

        return "UNKNOWN_ID"

    def handle_imu_packet(self, recv_time: float, data: tuple) -> None:
        packet = ImuPacket.from_tuple(recv_time, data)
        LOGGER.debug("TEMP: parsed IMU packet")  # TEMP log
        self.has_imu_data = True
        imu_sample = ImuSample(
            timestamp_s=packet.recv_time,
            gyro_rad_s=np.array([packet.gx, packet.gy, packet.gz], dtype=np.float64),
            accel_m_s2=np.array([packet.ax, packet.ay, packet.az], dtype=np.float64),
        )
        self.estimator.ingest_imu(imu_sample)
        LOGGER.debug("TEMP: imu ingested into estimator")  # TEMP log
        if self.last_baro_msl is None or abs(packet.baro_msl_m - self.last_baro_msl) > 1e-3:
            self.last_baro_msl = packet.baro_msl_m
            self.estimator.ingest_baro(
                BaroSample(timestamp_s=packet.recv_time, altitude_msl_m=packet.baro_msl_m)
            )
            LOGGER.debug("TEMP: baro ingested: %.3f", packet.baro_msl_m)  # TEMP log
        self.publish_imu(packet)

    def publish_imu(self, packet: ImuPacket) -> None:
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        msg.linear_acceleration.x = float(packet.ax)
        msg.linear_acceleration.y = float(packet.ay)
        msg.linear_acceleration.z = float(packet.az)
        msg.angular_velocity.x = float(packet.gx)
        msg.angular_velocity.y = float(packet.gy)
        msg.angular_velocity.z = float(packet.gz)
        self.imu_pub.publish(msg)
        LOGGER.debug("TEMP: published IMU message")  # TEMP log

    def capture_frame(self, frame_time: float) -> None:
        if not self.has_imu_data:
            LOGGER.debug("TEMP: skipping frame until IMU data available")  # TEMP log
            return

        nombre_salida = "/tmp/captura_gris.jpg"
        
        # Optimized parameters for instant photo capture:
        # -t 0 removes preview delay.
        # --immediate forces immediate sensor capture.
        comando = [
            "rpicam-still",
            "-o", nombre_salida,
            "--saturation", "0",
            "-t", "0",
            "--immediate"
        ]

        try:
            # Execute capture as a subprocess
            LOGGER.debug("TEMP: running capture command: %s", comando)  # TEMP log
            subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            LOGGER.debug("TEMP: capture subprocess completed")  # TEMP log
            
            # Attempt to read the saved frame
            if os.path.exists(nombre_salida):
                frame = cv2.imread(nombre_salida)
                if frame is None:
                    LOGGER.debug("TEMP: failed to read capture (empty image)")  # TEMP log
                    return
            else:
                LOGGER.debug("TEMP: capture file not found: %s", nombre_salida)  # TEMP log
                return
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().strip() if getattr(e, 'stderr', None) else str(e)
            LOGGER.error(f"TEMP: error taking photo with rpicam-still: {stderr}")
            return
        except Exception as e:
            LOGGER.error(f"TEMP: unexpected error while taking photo: {e}")
            return

        # Save raw/corrected frames to results every 2 seconds
        now = frame_time
        if now - self.last_results_save_time >= 2.0:
            # Save raw frame to results directory
            try:
                raw_filename = self.results_dir / f'raw_{now:.6f}.png'
                cv2.imwrite(str(raw_filename), frame)
                LOGGER.debug("TEMP: saved raw frame %s", raw_filename)  # TEMP log
            except Exception as e:
                LOGGER.error("TEMP: failed to save raw frame: %s", e)

            # Save a corrected (undistorted) version using estimator intrinsics
            try:
                if hasattr(self.estimator, 'camera_matrix') and hasattr(self.estimator, 'dist_coeffs'):
                    corrected = cv2.undistort(frame, self.estimator.camera_matrix, self.estimator.dist_coeffs)
                    if self.saved_corrected_count < self.max_saved_corrected:
                        corrected_filename = self.results_dir / f'corrected_{self.saved_corrected_count:04d}.png'
                        cv2.imwrite(str(corrected_filename), corrected)
                        LOGGER.debug("TEMP: saved corrected frame %s", corrected_filename)  # TEMP log
                        self.saved_corrected_count += 1
            except Exception as e:
                LOGGER.error("TEMP: failed to save corrected frame: %s", e)

            self.last_results_save_time = now

        frame_sample = FrameSample(timestamp_s=frame_time, image_bgr=frame)
        self.estimator.ingest_frame(frame_sample)
        LOGGER.debug("TEMP: frame ingested into estimator")  # TEMP log
        self.publish_frame(frame_sample)

        state = self.estimator.current_state()
        if state is not None:
            # Always write extended diagnostics (estimator.logs present)
            logs = getattr(self.estimator, 'logs', {})
            imu_sample = self.last_imu
            row = [
                state.timestamp_s,
                logs.get('dt', 0.0),
                state.position_m[0],
                state.position_m[1],
                state.agl_m,
                state.velocity_m_s[0],
                state.velocity_m_s[1],
                logs.get('vx_raw', 0.0),
                logs.get('vy_raw', 0.0),
                logs.get('vz_baro', 0.0),
                logs.get('features', 0),
                logs.get('dx_raw', 0.0),
                logs.get('dy_raw', 0.0),
                logs.get('dx_derot', 0.0),
                logs.get('dy_derot', 0.0),
                logs.get('dx_comp', 0.0),
                logs.get('dy_comp', 0.0),
                logs.get('omega_cx', 0.0),
                logs.get('omega_cy', 0.0),
                logs.get('omega_cz', 0.0),
                None if imu_sample is None else imu_sample.accel_m_s2[0],
                None if imu_sample is None else imu_sample.accel_m_s2[1],
                None if imu_sample is None else imu_sample.accel_m_s2[2],
            ]
            self.csv_writer.writerow(row)
            LOGGER.debug("TEMP: diagnostics row written to CSV: t=%.3f", state.timestamp_s)
        else:
            LOGGER.debug("TEMP: no valid state estimate available for CSV")  # TEMP log

    def publish_frame(self, frame: FrameSample) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(frame.image_bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            self.cam_pub.publish(msg)
            LOGGER.debug("TEMP: published camera frame")  # TEMP log
        except Exception as exc:
            self.get_logger().error(f"TEMP: error converting image: {exc}")

    def publish_state(self, state: StateEstimate) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(state.position_m[0])
        msg.pose.pose.position.y = float(state.position_m[1])
        msg.pose.pose.position.z = float(state.agl_m)
        msg.pose.pose.orientation.x = float(state.attitude_quat[0])
        msg.pose.pose.orientation.y = float(state.attitude_quat[1])
        msg.pose.pose.orientation.z = float(state.attitude_quat[2])
        msg.pose.pose.orientation.w = float(state.attitude_quat[3])
        msg.twist.twist.linear.x = float(state.velocity_m_s[0])
        msg.twist.twist.linear.y = float(state.velocity_m_s[1])
        msg.twist.twist.linear.z = float(state.velocity_m_s[2])
        self.state_pub.publish(msg)
        LOGGER.debug("TEMP: published state message")  # TEMP log

    def read_baro_msl(self) -> Optional[float]:
        """Preventive wrapper to avoid AttributeErrors from the main loop."""
        return self.last_baro_msl


def main() -> None:
    rclpy.init()
    node = FlowEstimatorNode(SERIAL_PORT, BAUD_RATE, SERIAL_TIMEOUT)

    try:
        node.connect()
        LOGGER.info("TEMP: system operational. Watchdog running.")  # TEMP log

        while node.is_running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

            node.listen_and_decode()

            now = time.monotonic()

            # Do not process frames or barometer until init handshake is complete
            if node.init_received:
                if now - node.last_frame_capture >= FRAME_INTERVAL_S:
                    node.last_frame_capture = now
                    node.capture_frame(now)

                baro_msl = node.read_baro_msl()
                if baro_msl is not None:
                    node.estimator.ingest_baro(BaroSample(timestamp_s=now, altitude_msl_m=baro_msl))

            # Watchdog timing (checks run regardless of init state but action below requires init)
            if now - node.last_watchdog_check < 0.5:
                continue
            node.last_watchdog_check = now

            if not node.init_received:
                continue

            if now - node.last_imu_time > 1:
                LOGGER.error("TEMP: WATCHDOG ALARM: triggering hardware reset.")  # TEMP log
                if node.serial_conn and node.serial_conn.is_open:
                    node.serial_conn.reset_input_buffer()
                    node.serial_conn.reset_output_buffer()
                node.last_imu_time = time.monotonic()

    except serial.SerialException as error:
        LOGGER.critical("TEMP: critical serial port error: %s", error)  # TEMP log
    except KeyboardInterrupt:
        LOGGER.info("TEMP: manual shutdown requested")  # TEMP log
    finally:
        LOGGER.info("TEMP: initiating safe shutdown...")  # TEMP log
        node.is_running = False
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        LOGGER.info("TEMP: system closed safely.")  # TEMP log


if __name__ == '__main__':
    main()