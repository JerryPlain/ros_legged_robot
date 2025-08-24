import rclpy
from rclpy.node import Node
import numpy as np
import pinocchio as pin
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ros_visuals import tf_transformations

class T11Node(Node):
    def __init__(self):
        super().__init__('t11_node')
        self.get_logger().info('t11 node started!')
        
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
        # set up 8 SE(3)
        self.frames = []
        for i, offset in enumerate(offsets):
            R = np.eye(3)
            p = np.array(offset)
            T = pin.SE3(R, p)
            self.frames.append(T)
            self.get_logger().info(f"Frame O{i}: position = {p}")
        
        # set up the location of the center of T_oc in the world frame
        self.T_oc = pin.SE3.Identity()
        self.dt = 0.1 

        # set up the broadcaster and timer (conduct broadcast_frames every 0.1s, update TF in real time)
        self.br = TransformBroadcaster(self) #
        self.timer = self.create_timer(0.1, self.broadcast_frames) 
    
    # define the broadcast function
    def broadcast_frames(self):
        # get the current timestamp
        now = self.get_clock().now().to_msg()
        
        # define the twist (angular & linear velocity)
        twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.])  # [wx, wy, wz, vx, vy, vz]
        twist = pin.Motion(twist_vec)

        # update T_oc using the exp transform from Lie algebra to Lie group
        delta_T = pin.exp6(twist * self.dt)
        self.T_oc = self.T_oc * delta_T

        # From SE(3) transform to get translation and rotation matrix
        p = self.T_oc.translation
        R = self.T_oc.rotation

        # construct 4x4 homogeneous matrix to extract quaternion (for ROS)
        T_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array([[0, 0, 0, 1]])))
        q = tf_transformations.quaternion_from_matrix(T_matrix)

        # publish TF of World -> Oc
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

        # Publish TF of Oc → O0~O7 
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

        # print the current location
        self.get_logger().info(f"T_oc pos: {p}, yaw ≈ {np.arctan2(R[1,0], R[0,0]):.2f} rad")

def main(args=None):
    rclpy.init(args=args)
    node = T11Node()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()