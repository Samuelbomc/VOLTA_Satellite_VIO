#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import EntityWrench, Entity
from rclpy.executors import ExternalShutdownException
import math
import random

class ParacaidasForzado(Node):
    def __init__(self):
        super().__init__('generador_perturbaciones')

        self.publisher_ = self.create_publisher(EntityWrench, '/perturbaciones_gz', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.t = 0.0
        
        # --- NEW: Flag to track the initial velocity impulse ---
        self.impulso_aplicado = False
        
        self.get_logger().info('Inyectando impulso inicial, seguido de fuerzas de paracaídas y torques suaves...')

    def timer_callback(self):
        msg = EntityWrench()
        msg.entity.name = "base_link" # Make sure this matches exactly what Gazebo expects
        msg.entity.type = Entity.LINK
        
        if not self.impulso_aplicado:
            # --- INITIAL IMPULSE PHASE ---
            # 4000 N applied for 0.05s = 200 Ns of momentum (20 m/s for a 10kg mass)
            msg.wrench.force.x = 4000.0
            msg.wrench.force.y = 4000.0
            msg.wrench.force.z = 0.0
            
            msg.wrench.torque.x = 0.0
            msg.wrench.torque.y = 0.0
            msg.wrench.torque.z = 0.0
            
            self.get_logger().info('¡Aplicando impulso inicial de 20 m/s en X y Y!')
            self.impulso_aplicado = True
            
        else:
            # --- PARACHUTE PHASE (Original Logic) ---
            # 1. PARACAÍDAS
            fuerza_z = random.gauss(10, 20) 
            
            # 2. PARALLAX LATERAL
            fuerza_x = 2 * math.sin(2.0 * math.pi * 0.1 * self.t)
            fuerza_y = 2 * math.cos(2.0 * math.pi * 0.1 * self.t)
            
            # 3. TAMBALEO DEL PARACAIDAS
            torque_x = 0.5 * math.sin(2.0 * math.pi * 0.4 * self.t)
            torque_y = 0.5 * math.cos(2.0 * math.pi * 0.4 * self.t)
            torque_z = 0.5 * math.sin(2.0 * math.pi * 0.4 * self.t)

            msg.wrench.force.x = fuerza_x
            msg.wrench.force.y = fuerza_y
            msg.wrench.force.z = fuerza_z
            
            msg.wrench.torque.x = torque_x
            msg.wrench.torque.y = torque_y
            msg.wrench.torque.z = torque_z

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