from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_visuals',
            executable='t13',
            name='t13_node',
            output='screen'
        ),
        ExecuteProcess(
            cmd=[
                'rviz2',
                '-d',
                '/workspaces/ros_ws/src/ros_visuals/config/t13.rviz'
            ],
            output='screen'
        )
    ])