import sys
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


def process_feedback(feedback):
    p = feedback.pose.position
    print(f'{feedback.marker_name} is now at ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})')


def make_6dof_marker():
    marker = InteractiveMarker()
    marker.header.frame_id = 'base_link'
    marker.name = 'target_marker'
    marker.description = '6-DOF Control'
    marker.scale = 0.3

    # visualize the cube 
    box_marker = Marker()
    box_marker.type = Marker.CUBE
    box_marker.scale.x = 0.1
    box_marker.scale.y = 0.1
    box_marker.scale.z = 0.1
    box_marker.color.r = 0.0
    box_marker.color.g = 0.5
    box_marker.color.b = 0.5
    box_marker.color.a = 1.0

    box_control = InteractiveMarkerControl()
    box_control.always_visible = True
    box_control.markers.append(box_marker)
    marker.controls.append(box_control)

    # Add 6 Dof control
    for axis in ['x', 'y', 'z']:
        rotate = InteractiveMarkerControl()
        rotate.name = f'rotate_{axis}'
        rotate.orientation.w = 1.0
        setattr(rotate.orientation, axis, 1.0)
        rotate.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        marker.controls.append(rotate)

        move = InteractiveMarkerControl()
        move.name = f'move_{axis}'
        move.orientation.w = 1.0
        setattr(move.orientation, axis, 1.0)
        move.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        marker.controls.append(move)

    return marker


def main():
    rclpy.init()
    node = rclpy.create_node('interactive_target_node')

    server = InteractiveMarkerServer(node, 'interactive_target')
    marker = make_6dof_marker()
    server.insert(marker)
    server.setCallback(marker.name, process_feedback)
    server.applyChanges()

    print("[INFO] Interactive marker started.")
    rclpy.spin(node)
    server.clear()
    server.applyChanges()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()