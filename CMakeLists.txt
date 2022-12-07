cmake_minimum_required(VERSION 3.12)

project(test)
set(CMAKE_C_COMPILER CXX)

# Find the Gazebo_Transport library
find_package(gz-transport12 REQUIRED OPTIONAL_COMPONENTS log)
set(GZ_TRANSPORT_VER ${gz-transport12_VERSION_MAJOR})
include_directories(${CMAKE_BINARY_DIR})

if (EXISTS "${CMAKE_SOURCE_DIR}/test.cc")
  add_executable(lidar_node test.cc)
  target_link_libraries(lidar_node gz-transport${GZ_TRANSPORT_VER}::core pthread)
endif()
