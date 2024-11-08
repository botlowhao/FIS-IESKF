cmake_minimum_required(VERSION 3.0.2)
project(fis_vo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  vwio_eskf
  message_generation
  message_runtime
)

add_message_files(
  FILES
  VOFISData.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs vwio_eskf message_runtime

)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# catkin_install_python(PROGRAMS
#   scripts/fis_vo_node.py  # 添加你的脚本路径
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Declare the executable
add_executable(orbdata_construct_node src/orbdata_construct.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(orbdata_construct_node
  ${catkin_LIBRARIES}
)

