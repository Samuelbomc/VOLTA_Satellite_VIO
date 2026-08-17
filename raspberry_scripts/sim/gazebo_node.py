import csv
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, Imu, CameraInfo
from ros_gz_interfaces.msg import Altimeter
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)

# ==========================================
# DATA TYPES
# ==========================================
@dataclass(frozen=True)
class ImuSample:
    timestamp_s: float
    gyro_rad_s: np.ndarray       # Angular velocity in body frame
    accel_m_s2: np.ndarray       # Gravity-compensated linear acceleration in body frame
    raw_accel_m_s2: np.ndarray   # Raw accelerometer output
    gravity_body: np.ndarray     # Calculated gravity vector projected in body frame
    orientation_q: np.ndarray    # Attitude quaternion

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

# ==========================================
# ESTIMATOR LOGIC (Adapted from IRL)
# ==========================================
class OpticalFlowTracker:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_kp = None

    def _to_gray(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=12, blockSize=7
        )

    def compute_flow(self, frame: FrameSample) -> Tuple[np.ndarray, np.ndarray]:
        gray = self._to_gray(frame.image_bgr)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            return np.empty((0, 2)), np.empty((0, 2))

        if self.prev_kp is None or len(self.prev_kp) < 20:
            self.prev_kp = self._detect_features(self.prev_gray)
            if self.prev_kp is None:
                self.prev_gray = gray
                return np.empty((0, 2)), np.empty((0, 2))

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_kp, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        if next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_kp = self._detect_features(gray)
            return np.empty((0, 2)), np.empty((0, 2))

        status = status.reshape(-1) == 1
        prev_pts = self.prev_kp.reshape(-1, 2)[status]
        curr_pts = next_pts.reshape(-1, 2)[status]

        self.prev_gray = gray
        self.prev_kp = curr_pts.reshape(-1, 1, 2) if len(curr_pts) else self._detect_features(gray)

        if self.prev_kp is not None and len(self.prev_kp) < 20:
            self.prev_kp = self._detect_features(gray)

        return prev_pts, curr_pts


class GyroDeRotation:
    def __init__(self):
        self.r_cam_to_imu = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)

    def imu_to_camera(self, omega_imu: np.ndarray) -> np.ndarray:
        return self.r_cam_to_imu.T @ omega_imu

    def compensate_flow(self, pts_prev: np.ndarray, pts_curr: np.ndarray, omega_rad_s: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if omega_rad_s is None or dt <= 0.0:
            return pts_prev, pts_curr

        omega_x, omega_y, omega_z = omega_rad_s
        x, y = pts_prev[:, 0], pts_prev[:, 1]

        dx_rot = (-(1.0 + x * x) * omega_y + x * y * omega_x + y * omega_z) * dt
        dy_rot = (-(x * y) * omega_y + (1.0 + y * y) * omega_x - x * omega_z) * dt

        pts_curr_comp = np.column_stack((pts_curr[:, 0] - dx_rot, pts_curr[:, 1] - dy_rot))
        return pts_prev, pts_curr_comp


class BaroAltitudeModel:
    def __init__(self, reference_msl_m: float):
        self.reference_msl_m = reference_msl_m

    def agl_from_baro(self, baro: BaroSample) -> float:
        agl_m = baro.altitude_msl_m - self.reference_msl_m
        return max(agl_m, 0.1)


class RobustFlowImuBaroEstimator:
    def __init__(self, reference_msl_m: float = 3200.0, camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None):
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.eye(3)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4)

        self.flow = OpticalFlowTracker(self.camera_matrix, self.dist_coeffs)
        self.derotation = GyroDeRotation()
        self.baro_model = BaroAltitudeModel(reference_msl_m)
        self.state: Optional[StateEstimate] = None

        self.last_frame_time: Optional[float] = None
        self.last_baro: Optional[BaroSample] = None
        self.imu_buffer: list[ImuSample] = []
        self.logs = {}

    def ingest_imu(self, imu: ImuSample) -> None:
        self.imu_buffer.append(imu)

    def ingest_baro(self, baro: BaroSample) -> None:
        self.last_baro = baro

    def ingest_frame(self, frame: FrameSample, current_quat: np.ndarray) -> None:
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
            return

        self.logs['dt'] = dt

        if self.last_baro is None: return
        z_agl = self.baro_model.agl_from_baro(self.last_baro)
        if z_agl <= 0.0: return

        imu_window = [imu for imu in self.imu_buffer if self.last_frame_time < imu.timestamp_s <= frame.timestamp_s]
        self.imu_buffer = [imu for imu in self.imu_buffer if imu.timestamp_s > frame.timestamp_s]
        if not imu_window: return

        omega_avg = np.mean([imu.gyro_rad_s for imu in imu_window], axis=0)
        omega_cam = self.derotation.imu_to_camera(omega_avg)
        self.logs['omega_cx'], self.logs['omega_cy'], self.logs['omega_cz'] = omega_cam.tolist()

        pts_prev, pts_curr = self.flow.compute_flow(frame)
        self.last_frame_time = frame.timestamp_s

        if pts_prev.size == 0 or pts_curr.size == 0: return

        self.logs['features'] = float(len(pts_curr))

        # Standard Gazebo Plumb-bob undistortion
        try:
            pts_prev_norm = cv2.undistortPoints(pts_prev.reshape(-1,1,2), self.camera_matrix, self.dist_coeffs).reshape(-1,2)
            pts_curr_norm = cv2.undistortPoints(pts_curr.reshape(-1,1,2), self.camera_matrix, self.dist_coeffs).reshape(-1,2)
        except Exception:
            pts_prev_norm = pts_prev.astype(np.float64)
            pts_curr_norm = pts_curr.astype(np.float64)

        raw_flow = pts_curr_norm - pts_prev_norm
        self.logs['dx_raw'] = float(np.median(raw_flow[:,0]))
        self.logs['dy_raw'] = float(np.median(raw_flow[:,1]))

        pts_prev_norm, pts_curr_norm = self.derotation.compensate_flow(pts_prev_norm, pts_curr_norm, omega_cam, dt)
        derot_flow = pts_curr_norm - pts_prev_norm
        self.logs['dx_derot'] = float(np.median(derot_flow[:,0]))
        self.logs['dy_derot'] = float(np.median(derot_flow[:,1]))

        vz_baro = 0.0 if self.state is None else (z_agl - self.state.agl_m) / dt
        self.logs['vz_baro'] = vz_baro

        x_norm = pts_prev_norm[:,0]
        y_norm = pts_prev_norm[:,1]
        dx_trans = derot_flow[:,0] + (x_norm * vz_baro * dt / z_agl)
        dy_trans = derot_flow[:,1] + (y_norm * vz_baro * dt / z_agl)

        self.logs['dx_comp'] = float(np.median(dx_trans))
        self.logs['dy_comp'] = float(np.median(dy_trans))

        # 1. Pixel to Camera Metric Velocity
        v_cam = np.array([-(self.logs['dx_comp'] * z_agl) / dt, -(self.logs['dy_comp'] * z_agl) / dt, 0.0])
        
        # 2. Camera to Body Frame Velocity
        v_body = self.derotation.r_cam_to_imu @ v_cam

        # >>> TILT-FALL MIRAGE COMPENSATION & FRAME REFERENCE UPDATES <<<
        if np.linalg.norm(current_quat) > 0.5:
            # 3. Dynamic Gravity subtraction using active attitude
            r_world2body = R.from_quat(current_quat).inv()
            gravity_body = r_world2body.apply([0.0, 0.0, 1.0])
            v_body[0] -= gravity_body[0] * vz_baro
            v_body[1] -= gravity_body[1] * vz_baro

            # 4. Reference Frame alignment: Body to World Frame Integration
            r_body2world = R.from_quat(current_quat)
            v_world = r_body2world.apply(v_body)
        else:
            v_world = v_body

        vx_raw, vy_raw = float(v_world[0]), float(v_world[1])

        self.logs['vx_raw'] = vx_raw
        self.logs['vy_raw'] = vy_raw

        alpha = float(np.clip(100.0 / z_agl, 0.05, 0.8))

        if self.state is None:
            vx, vy = vx_raw, vy_raw
            position = np.zeros(3, dtype=np.float64)
            velocity = np.array([vx, vy, vz_baro], dtype=np.float64)
        else:
            vx = self.state.velocity_m_s[0] * (1.0 - alpha) + vx_raw * alpha
            vy = self.state.velocity_m_s[1] * (1.0 - alpha) + vy_raw * alpha
            velocity = np.array([vx, vy, vz_baro], dtype=np.float64)
            position = self.state.position_m + velocity * dt

        self.state = StateEstimate(
            timestamp_s=frame.timestamp_s,
            position_m=position,
            velocity_m_s=velocity,
            attitude_quat=current_quat,
            agl_m=z_agl,
        )

    def current_state(self) -> Optional[StateEstimate]:
        return self.state


# ==========================================
# ROS 2 NODE AND MAIN LOOP
# ==========================================
class GazeboFlowEstimatorNode(Node):
    def __init__(self):
        super().__init__('gazebo_flow_imu_baro_node')
        self.bridge = CvBridge()
        self.has_camera_info = False

        self.results_dir = Path('sim_output')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = open('results.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        header = [
            'timestamp', 'dt', 'pos_x', 'pos_y', 'agl',
            'vel_x_filt', 'vel_y_filt', 'vel_x_raw', 'vel_y_raw', 'vz_baro',
            'features', 'dx_raw', 'dy_raw', 'dx_derot', 'dy_derot', 'dx_comp', 'dy_comp',
            'omega_cx', 'omega_cy', 'omega_cz',
            'imu_ax', 'imu_ay', 'imu_az',
            'gravity_x', 'gravity_y', 'gravity_z',
            'gt_x', 'gt_y', 'gt_z', 'gt_qx', 'gt_qy', 'gt_qz', 'gt_qw'
        ]
        self.csv_writer.writerow(header)

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/imu/raw', self.imu_callback, qos)
        self.create_subscription(Image, '/camera/raw', self.image_callback, qos)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, qos)
        self.create_subscription(Altimeter, '/altimeter/raw', self.altimeter_callback, QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, '/validation', self.ground_truth_callback, qos)

        self.estimator: Optional[RobustFlowImuBaroEstimator] = None
        self.last_imu: Optional[ImuSample] = None
        self.last_gt: Optional[Odometry] = None

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if not self.has_camera_info and len(msg.k) == 9:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64) if msg.d else np.zeros(4, dtype=np.float64)
            self.estimator = RobustFlowImuBaroEstimator(reference_msl_m=0.0, camera_matrix=K, dist_coeffs=D)
            self.has_camera_info = True

    def imu_callback(self, msg: Imu) -> None:
        if not self.has_camera_info or not self.estimator: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        raw_accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float64)
        q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float64)
        
        # Subtract gravity using active orientation vector transformations
        if np.linalg.norm(q) > 0.5:
            r_body2world = R.from_quat(q)
            # Projects World gravity [0.0, 0.0, 9.81] into the Body Coordinate System
            gravity_body = r_body2world.inv().apply([0.0, 0.0, 9.81])
        else:
            gravity_body = np.array([0.0, 0.0, 9.81], dtype=np.float64)
            
        linear_accel = raw_accel - gravity_body
        
        imu_sample = ImuSample(
            timestamp_s=t,
            gyro_rad_s=np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64),
            accel_m_s2=linear_accel,
            raw_accel_m_s2=raw_accel,
            gravity_body=gravity_body,
            orientation_q=q
        )
        self.last_imu = imu_sample
        self.estimator.ingest_imu(imu_sample)

    def altimeter_callback(self, msg: Altimeter) -> None:
        if not self.has_camera_info or not self.estimator: return
        t = self.get_clock().now().nanoseconds * 1e-9
        self.estimator.ingest_baro(BaroSample(t, 3200.0 + float(msg.vertical_position)))

    def image_callback(self, msg: Image) -> None:
        if not self.has_camera_info or not self.estimator: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        quat = self.last_imu.orientation_q if self.last_imu else np.array([0., 0., 0., 1.])
        
        self.estimator.ingest_frame(FrameSample(t, frame_bgr), quat)

        st = self.estimator.state
        if st is not None:
            lg = self.estimator.logs
            imu = self.last_imu
            gt = self.last_gt

            self.csv_writer.writerow([
                st.timestamp_s, lg.get('dt', 0.0), st.position_m[0], st.position_m[1], st.agl_m,
                st.velocity_m_s[0], st.velocity_m_s[1], lg.get('vx_raw', 0.0), lg.get('vy_raw', 0.0), lg.get('vz_baro', 0.0),
                lg.get('features', 0), lg.get('dx_raw', 0.0), lg.get('dy_raw', 0.0),
                lg.get('dx_derot', 0.0), lg.get('dy_derot', 0.0), lg.get('dx_comp', 0.0), lg.get('dy_comp', 0.0),
                lg.get('omega_cx', 0.0), lg.get('omega_cy', 0.0), lg.get('omega_cz', 0.0),
                imu.accel_m_s2[0] if imu else 0.0, imu.accel_m_s2[1] if imu else 0.0, imu.accel_m_s2[2] if imu else 0.0,
                imu.gravity_body[0] if imu else 0.0, imu.gravity_body[1] if imu else 0.0, imu.gravity_body[2] if imu else 0.0,
                gt.pose.pose.position.x if gt else 0.0, gt.pose.pose.position.y if gt else 0.0, gt.pose.pose.position.z if gt else 0.0,
                gt.pose.pose.orientation.x if gt else 0.0, gt.pose.pose.orientation.y if gt else 0.0, gt.pose.pose.orientation.z if gt else 0.0, gt.pose.pose.orientation.w if gt else 0.0,
            ])

    def ground_truth_callback(self, msg: Odometry) -> None:
        self.last_gt = msg

def main() -> None:
    rclpy.init()
    node = GazeboFlowEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()