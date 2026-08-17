"""Convert camera RGB frames to mono8 and republish them for downstream nodes."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
import cv2
from cv_bridge import CvBridge

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('rgb_to_mono_node')

        # Use the sensor-data QoS because the image stream is best-effort and high rate.
        self.subscription = self.create_subscription(
            Image,
            '/camera/raw',
            self.listener_callback,
            qos_profile_sensor_data)

        # Publish the grayscale stream on the topic expected by VINS/Basalt-style pipelines.
        self.publisher_ = self.create_publisher(Image, '/cam0/image_raw', 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        try:
            # Convert from ROS to OpenCV, then to grayscale, and republish with the same header.
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            ros_image = self.bridge.cv2_to_imgmsg(gray_image, encoding='mono8')
            ros_image.header = data.header

            self.publisher_.publish(ros_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    processor = ImageProcessor()
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
