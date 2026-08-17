#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import EntityWrench, Entity
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
import math
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

class ParacaidasForzado(Node):
    def __init__(self):
        super().__init__('generador_perturbaciones')

        # Publisher for forces and torques toward Gazebo.
        self.publisher_ = self.create_publisher(EntityWrench, '/perturbaciones_gz', 10)
        
        # Subscribe to ground truth to read the actual orientation.
        self.create_subscription(Odometry, '/validation', self.odom_callback, 10)
        self.current_q = np.array([0.0, 0.0, 0.0, 1.0]) # Default quaternion.

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.t = 0.0
        
        # Flag used to track whether the initial velocity impulse was applied.
        self.impulso_aplicado = False
        
        self.get_logger().info('Injecting impulse, parachute forces, and tilt limiter (<120°)...')

    def odom_callback(self, msg: Odometry):
        """Update the current satellite orientation from the simulator."""
        self.current_q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

    def timer_callback(self):
        msg = EntityWrench()
        msg.entity.name = "base_link" # Make sure this matches your URDF/SDF.
        msg.entity.type = Entity.LINK
        
        if not self.impulso_aplicado:
            # --- INITIAL IMPULSE PHASE ---
            msg.wrench.force.x = 4000.0
            msg.wrench.force.y = 4000.0
            msg.wrench.force.z = 0.0
            
            msg.wrench.torque.x = 0.0
            msg.wrench.torque.y = 0.0
            msg.wrench.torque.z = 0.0
            
            self.get_logger().info('Applying initial 20 m/s impulse in X and Y!')
            self.impulso_aplicado = True
            
        else:
            # --- PARACHUTE AND PERTURBATION PHASE ---
            # 1. PARACHUTE (Z-axis resistance)
            fuerza_z = random.gauss(100, 200) 
            
            # 2. LATERAL DRIFT (Wind)
            fuerza_x = 20 * math.sin(2.0 * math.pi * 0.1 * self.t)
            fuerza_y = 20 * math.cos(2.0 * math.pi * 0.1 * self.t)
            
            # 3. PARACHUTE SWAY (Random torques)
            torque_x = 10 * math.sin(2.0 * math.pi * 0.4 * self.t)
            torque_y = 10 * math.cos(2.0 * math.pi * 0.4 * self.t)
            torque_z = 0.5 * math.sin(2.0 * math.pi * 0.4 * self.t)

            # =================================================================
            # 4. CONTROLADOR DE ACTITUD (Evitar giro > 120 grados)
            # =================================================================
            # Compute where the drone's Z axis points in the world frame.
            r = R.from_quat(self.current_q)
            z_body = r.apply([0.0, 0.0, 1.0])
            z_world = np.array([0.0, 0.0, 1.0]) # Absolute "up" direction.
            
            # Tilt angle (0° = perfectly upright, 180° = upside down).
            tilt_angle = math.acos(np.clip(z_body[2], -1.0, 1.0))
            tilt_deg = math.degrees(tilt_angle)
            
            torque_restaurador = np.array([0.0, 0.0, 0.0])
            
            # If the vehicle starts tipping past 90 degrees, stabilize it.
            if tilt_deg > 90.0:
                # K_p grows rapidly as the angle approaches 120°.
                # At 120°, K_p = 30 * 5 = 150 N·m, enough to right the drone.
                K_p = (tilt_deg - 90.0) * 5.0 
                
                # The cross product gives the exact rotation axis to bring it back upright.
                restoring_axis = np.cross(z_body, z_world)
                norm_eje = np.linalg.norm(restoring_axis)
                
                if norm_eje > 0.001:
                    restoring_axis /= norm_eje
                    torque_restaurador = restoring_axis * K_p

            msg.wrench.force.x = fuerza_x
            msg.wrench.force.y = fuerza_y
            msg.wrench.force.z = fuerza_z
            
            # Apply the normal sway plus the rescue restoring torque.
            msg.wrench.torque.x = torque_x + torque_restaurador[0]
            msg.wrench.torque.y = torque_y + torque_restaurador[1]
            msg.wrench.torque.z = torque_z + torque_restaurador[2]

        self.publisher_.publish(msg)
        self.t += 0.05

def main(args=None):
    rclpy.init(args=args)
    nodo = ParacaidasForzado()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()