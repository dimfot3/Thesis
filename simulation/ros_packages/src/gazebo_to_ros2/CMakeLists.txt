cmake_minimum_required(VERSION 3.5)
project(gazebo_to_ros2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gz-transport12 QUIET REQUIRED OPTIONAL_COMPONENTS log)
set(GZ_TRANSPORT_VER ${gz-transport12_VERSION_MAJOR})

# Build the executable
add_executable(gazebo_to_ros2_node src/gazebo_to_ros2_node.cpp)
ament_target_dependencies(gazebo_to_ros2_node rclcpp geometry_msgs sensor_msgs)
target_link_libraries(gazebo_to_ros2_node gz-transport${GZ_TRANSPORT_VER}::core ${PCL_LIBRARIES})
# Install the executable
install(TARGETS gazebo_to_ros2_node DESTINATION lib/${PROJECT_NAME})


# Build the package
ament_package()








