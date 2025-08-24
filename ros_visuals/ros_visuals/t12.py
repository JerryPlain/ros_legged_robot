import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, TwistStamped
from ros_visuals import tf_transformations

# define the twist coordinate transform function
def transform_twist(T: pin.SE3, V: pin.Motion) -> pin.Motion:
    return T.act(V)

class T12Node(Node):
    def __init__(self):
        super().__init__('t12_node')
        self.get_logger().info('t12 node started!')

        # initialization
        L = 0.4
        H = 0.5
        offsets = [
            [-L, -L, -H], [ L, -L, -H], [-L,  L, -H], [ L,  L, -H],
            [-L, -L,  H], [ L, -L,  H], [-L,  L,  H], [ L,  L,  H],
        ]

        # set up a cube and put each transform of the corner point SE(3) into self.frames 
        self.frames = []
        for i, offset in enumerate(offsets):
            R = np.eye(3)
            p = np.array(offset)
            T = pin.SE3(R, p)
            self.frames.append(T)
            self.get_logger().info(f"Frame O{i}: position = {p}")

        self.T_oc = pin.SE3.Identity()
        self.dt = 0.1

        self.br = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_frames)
        # create 2 ROS Topics:
        self.twist_pub = self.create_publisher(TwistStamped, 'twist_w', 10) # publish the twist from 0o to world
        self.twist_pub_o6_inv = self.create_publisher(TwistStamped, 'twist_o6', 10) # publish the twist from world to O6

    def broadcast_frames(self):
        now = self.get_clock().now().to_msg()

        # Step 1: setup
        twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.])
        twist = pin.Motion(twist_vec)
        delta_T = pin.exp6(twist * self.dt)
        self.T_oc = self.T_oc * delta_T

        # Step 2: TF world → Oc
        p = self.T_oc.translation
        R = self.T_oc.rotation
        T_matrix = np.vstack((np.hstack((R, np.zeros((3, 1)))), np.array([[0, 0, 0, 1]])))
        q = tf_transformations.quaternion_from_matrix(T_matrix)

        t_center = TransformStamped()
        t_center.header.stamp = now
        t_center.header.frame_id = 'world'
        t_center.child_frame_id = 'Oc'
        t_center.transform.translation.x = p[0]
        t_center.transform.translation.y = p[1]
        t_center.transform.translation.z = p[2]
        t_center.transform.rotation.x = q[0]
        t_center.transform.rotation.y = q[1]
        t_center.transform.rotation.z = q[2]
        t_center.transform.rotation.w = q[3]
        self.br.sendTransform(t_center)

        # Step 3: Oc → O0~O7 
        for i, T in enumerate(self.frames):
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = 'Oc'
            tf.child_frame_id = f'O{i}'
            tf.transform.translation.x = T.translation[0]
            tf.transform.translation.y = T.translation[1]
            tf.transform.translation.z = T.translation[2]
            tf.transform.rotation.x = 0.0
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = 0.0
            tf.transform.rotation.w = 1.0
            self.br.sendTransform(tf)

        # Step 4: Define twist in o0, express in world
        T_c_to_O0 = self.frames[0]
        T_w_to_O0 = self.T_oc * T_c_to_O0 # get the transform from WOlrd to O0
        cV = pin.Motion(np.array([0., 0., 0.1, 0., 1.0, 0.])) # twist defined in O0
        wV = transform_twist(T_w_to_O0, cV) # express in World

        msg = TwistStamped()
        msg.header.stamp = now
        msg.header.frame_id = 'world'
        msg.twist.angular.x = wV.angular[0]
        msg.twist.angular.y = wV.angular[1]
        msg.twist.angular.z = wV.angular[2]
        msg.twist.linear.x = wV.linear[0]
        msg.twist.linear.y = wV.linear[1]
        msg.twist.linear.z = wV.linear[2]
        self.twist_pub.publish(msg)

        # Step 5: define twist in world, express in O6
        T_w_to_O6 = self.T_oc * self.frames[6]
        T_o6_to_w = T_w_to_O6.inverse()
        wV = pin.Motion(np.array([0., 0., 0.1, 0., 1.0, 0.]))
        c2V = transform_twist(T_o6_to_w, wV)

        msg_o6 = TwistStamped()
        msg_o6.header.stamp = now
        msg_o6.header.frame_id = 'world'
        msg_o6.twist.angular.x = c2V.angular[0]
        msg_o6.twist.angular.y = c2V.angular[1]
        msg_o6.twist.angular.z = c2V.angular[2]
        msg_o6.twist.linear.x = c2V.linear[0]
        msg_o6.twist.linear.y = c2V.linear[1]
        msg_o6.twist.linear.z = c2V.linear[2]
        self.twist_pub_o6_inv.publish(msg_o6)

        # Step 6: comparison between .act() & .action
        V1 = T_o6_to_w.act(wV) 
        V2_vec = T_o6_to_w.action @ wV.vector 
        V2 = pin.Motion(V2_vec)
        error = np.linalg.norm(V1.vector - V2.vector)
        self.get_logger().info(f"[Step6] Test the Difference of using Act from Pinocchio & T.action manually: {error:.2e}")

        # Step 8: When cage is spiralling, the angular component is stable, but the linear component will change with the change of reference location

def main(args=None):
    rclpy.init(args=args)
    node = T12Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()