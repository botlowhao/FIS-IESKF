cmake_minimum_required(VERSION 3.0.2)
project(fis_wo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  vwio_eskf
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs vwio_eskf

)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/fis_wo.py  # 添加你的脚本路径
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

