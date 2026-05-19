cmake_minimum_required(VERSION 3.0.2)
project(rovio_sensor_bridge)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sensor_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
