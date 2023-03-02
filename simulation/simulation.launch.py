from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os
import yaml


def parse_arguments():
    args = {}
    with open('config.yaml', 'r') as file:
        args_yaml = yaml.safe_load(file)
    args['simulation_env'] = args_yaml['simulation_env']
    args['rviz2'] = args_yaml['rviz2']
    humans = [human_name for human_name in list(args_yaml.keys()) if human_name[:5] == 'human']
    lidars = [lidar_name for lidar_name in list(args_yaml.keys()) if lidar_name[:5] == 'lidar']
    args['humans'] = {}
    args['lidars'] = {}
    for human in humans:
        args['humans'][human] = args_yaml[human]
    for lidar in lidars:
        args['lidars'][lidar] = args_yaml[lidar]
    return args

def generate_launch_description():
    # load arguments
    args = parse_arguments()
    simulation_env_arg = args['simulation_env']
    rviz2_arg = args['rviz2']

    # set environment paths
    os.environ['AMENT_PREFIX_PATH'] +=f':{os.getcwd()}/ros_packages/install/gazebo_to_ros2'

    # init lidar gazebo to ros2 node
    lidar_node = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_to_ros2', 'gazebo_to_ros2_node'],
        output='screen'
    )

    # init simulation node
    gazebo_node = ExecuteProcess(
    cmd=['gz', 'sim', f'indoor_spaces/{simulation_env_arg}/{simulation_env_arg}.sdf'],
    output='screen')

    # init rviz2 monitor
    rviz2_node = ExecuteProcess(
    cmd=['rviz2'],
    output='screen')

    # create and return launcher
    ld = LaunchDescription()
    ld.add_action(lidar_node)
    ld.add_action(gazebo_node)
    if rviz2_arg: ld.add_action(rviz2_node)
    return ld

