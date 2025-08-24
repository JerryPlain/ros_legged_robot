from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    urdf_path = os.path.join(
        os.getcwd(), 'src', 'talos_description', 'robots', 'talos_reduced.urdf'
    )
    with open(urdf_path, 'r') as infp:
        robot_description = infp.read()

    rviz_config_path = os.path.join(
        os.getcwd(), 'src', 'ros_visuals', 'config', 'talos.rviz'
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen',
        ),
    ])