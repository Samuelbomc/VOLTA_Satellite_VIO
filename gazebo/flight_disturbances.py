#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import EntityWrench, Entity
from rclpy.executors import ExternalShutdownException
import math

class GeneradorPerturbaciones(Node):
    def __init__(self):
        super().__init__('generador_perturbaciones')
        self.publisher_ = self.create_publisher(EntityWrench, '/perturbaciones_gz', 10)
        self.timer = self.create_timer(0.05, self.timer_callback) # 20 Hz
        self.t = 0.0
        self.get_logger().info('Inyectando perturbaciones aerodinámicas...')

    def timer_callback(self):
        msg = EntityWrench()
        msg.entity.name = "base_link" 
        msg.entity.type = Entity.LINK
        
        A = 20.0  
        B = 5.0   
        f_p = 0.4 
        f_s = 0.1 
        
        msg.wrench.torque.x = A * math.sin(2.0 * math.pi * f_p * self.t)
        msg.wrench.torque.y = A * math.cos(2.0 * math.pi * f_p * self.t)
        msg.wrench.torque.z = B * math.sin(2.0 * math.pi * f_s * self.t)
        
        self.publisher_.publish(msg)
        self.t += 0.05

def main(args=None):
    rclpy.init(args=args)
    nodo = GeneradorPerturbaciones()
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