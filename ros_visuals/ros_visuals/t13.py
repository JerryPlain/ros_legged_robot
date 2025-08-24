import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, WrenchStamped
from ros_visuals import tf_transformations

def transform_wrench(T: pin.SE3, F: pin.Force) -> pin.Force:
    return T.actInv(F) # from world to local (wrench)

class T13Node(Node):
    def __init__(self):
        super().__init__('t13_node')
        self.get_logger().info('t13 node started!')

        L = 0.4
        H = 0.5
        offsets = [
            [-L, -L, -H],
            [ L, -L, -H],
            [-L,  L, -H],
            [ L,  L, -H],
            [-L, -L,  H],
            [ L, -L,  H],
            [-L,  L,  H],
            [ L,  L,  H],
        ]

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
        self.timer = self.create_timer(self.dt, self.broadcast_frames)

        # set up two topics
        self.wrench_pub = self.create_publisher(WrenchStamped, 'wrench_w', 10)
        self.wrench_pub_o6 = self.create_publisher(WrenchStamped, 'wrench_o6', 10)

    def broadcast_frames(self):
        now = self.get_clock().now().to_msg()

        twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.])
        twist = pin.Motion(twist_vec)
        delta_T = pin.exp6(twist * self.dt) 
        self.T_oc = self.T_oc * delta_T

        # get R and P From T_oc, in order to turn it into quaternion
        p = self.T_oc.translation
        R = self.T_oc.rotation
        T_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array([[0, 0, 0, 1]])))
        q = tf_transformations.quaternion_from_matrix(T_matrix)

        # publish TF：world → Oc
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

        # publish Oc → O0~O7（8 corner points)
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

        # Step 3: publish /wrench_w
        T_c_to_O0 = self.frames[0]
        T_w_to_Oc = self.T_oc
        T_w_to_O0 = T_w_to_Oc * T_c_to_O0

        # define wrench
        cW = pin.Force(np.array([0., 0., 1.0]), np.array([5.0, 0., 0.]))  # torque, force
        # use .actInv() from O0 to world expression
        wW = transform_wrench(T_w_to_O0, cW)

        msg = WrenchStamped()
        msg.header.stamp = now
        msg.header.frame_id = 'world'
        msg.wrench.torque.x = wW.angular[0]
        msg.wrench.torque.y = wW.angular[1]
        msg.wrench.torque.z = wW.angular[2]
        msg.wrench.force.x  = wW.linear[0]
        msg.wrench.force.y  = wW.linear[1]
        msg.wrench.force.z  = wW.linear[2]

        self.wrench_pub.publish(msg)

        # Step 4: publish /wrench_o6
        T_c_to_O6 = self.frames[6]
        T_w_to_O6 = T_w_to_Oc * T_c_to_O6
        T_o6_to_w = T_w_to_O6.inverse() # get world → O6 's SE(3) and then inverse

        wW2 = pin.Force(np.array([0., 0., 1.0]), np.array([5.0, 0., 0.]))  # same wrench in world
        c2W = transform_wrench(T_o6_to_w, wW2)

        # Step 5：comparison between two methods
        c2W_pin = c2W  # M1：actInv

        # M2: Manually
        AdT_T = T_o6_to_w.action.T 
        c2W_vec_manual = AdT_T @ wW2.vector 
        c2W_manual = pin.Force(c2W_vec_manual)

        diff = np.linalg.norm(c2W_pin.vector - c2W_manual.vector)
        self.get_logger().info(f"[Step6] wrench actInv vs matrix diff: {diff:.2e}")
        
        msg_o6 = WrenchStamped()
        msg_o6.header.stamp = now
        msg_o6.header.frame_id = 'world'
        msg_o6.wrench.torque.x = c2W.angular[0]
        msg_o6.wrench.torque.y = c2W.angular[1]
        msg_o6.wrench.torque.z = c2W.angular[2]
        msg_o6.wrench.force.x = c2W.linear[0]
        msg_o6.wrench.force.y = c2W.linear[1]
        msg_o6.wrench.force.z = c2W.linear[2]
        self.wrench_pub_o6.publish(msg_o6)

def main(args=None):
    rclpy.init(args=args)
    node = T13Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()