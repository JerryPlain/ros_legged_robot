"""
task 1:
1. use pinocchio to set up 8 SE(3) frames representing the corners of a box centered at Oc  
2. Define a center frame T_oc that moves with a constant twist (angular and linear velocity)
3. Use pinocchio.exp6 to update the pose of T_oc over time
4. Broadcast the TFs of World -> Oc and Oc -> O0~O7 using ROS2 tf2
5. Print the current position and yaw angle of T_oc in the world
"""

import rclpy # this is ROS2 Python Client API
from rclpy.node import Node # import Node class, which is defined in rclpy.node module
import numpy as np
import pinocchio as pin # process SE(3), exp6, Motion etc.
from tf2_ros import TransformBroadcaster # tf2_ros is the ROS2 tf2 package for Python, TransformBroadcaster is the class for broadcasting TFs
from geometry_msgs.msg import TransformStamped # geometry_msgs is the ROS2 package that defines geometry-related message types, TransformStamped is the message type for TFs
from ros_visuals import tf_transformations # functions we defined for quaternion conversions

class T11Node(Node):
    """
    A ROS2 node that broadcasts TF frames for a moving box defined by 8 corner frames.
    """
    # 1. define the node
    def __init__(self):
        super().__init__('t11_node')
        self.get_logger().info('t11 node started!')

    # 2. set up the 8 corner frames
        L = 0.4
        H = 0.5
        # define the offsets of the 8 corners relative to the center Oc
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
        # 3. create SE(3) frames for each corner
        self.frames = []
        for i, offset in enumerate(offsets):
            R = np.eye(3) # identity rotation
            p = np.array(offset) # position vector
            T = pin.SE3(R, p) # create SE(3) frame
            # add the frame to the list
            self.frames.append(T)
            # use logger to print the position of each frame
            # self.get_logger().info() is used for logging information in ROS2
            self.get_logger().info(f"Frame O{i}: position = {p}")
        
        # 4. initialize the center frame T_oc and time step dt
        # T_oc represents the transform from world to center Oc (position and orientation)
        # It is initialized to the identity transform (no translation or rotation)
        self.T_oc = pin.SE3.Identity() # start at origin
        self.dt = 0.1  # time step for updates

        # 5. set up the broadcaster and timer (conduct broadcast_frames every 0.1s, update TF in real time)
        self.br = TransformBroadcaster(self)
        # publish TFs every 0.1 seconds
        self.timer = self.create_timer(0.1, self.broadcast_frames)

    # define the broadcast function
    """
    Broadcast the TF frames for the moving box.
    Steps:
    1. Get the current timestamp
    2. Publish the TF from world to Oc
    3. Publish the TFs from Oc to O0~O7
    """
    def broadcast_frames(self):
        # get the current timestamp
        now = self.get_clock().now().to_msg()
        
        # 1. define the 6D twist (angular and linear velocity)  e.g. [wx, wy, wz, vx, vy, vz]
        # this twist is lie algebra 
        # SE(3) is lie group, its lie algebra is se(3), twist is an element of se(3)
        twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.])
        # create a pinocchio Motion object from the twist vector, e.g. from [wx, wy, wz, vx, vy, vz] to Motion object
        # In pinocchio, Motion is used to represent twists and spatial velocities
        # Motion is a 6D vector that represents the twist (angular and linear velocity)
        twist = pin.Motion(twist_vec)

        # 2. compute the delta_T using exp6
        # exp6 maps a twist (element of se(3)) to a transformation (element of SE(3))
        delta_T = pin.exp6(twist * self.dt)  # compute the incremental transform over time dt
        # update the center frame T_oc by applying the incremental transform
        # new pose = old pose * delta_T
        self.T_oc = self.T_oc * delta_T

        # From SE(3) transform to get translation and rotation matrix
        p = self.T_oc.translation
        R = self.T_oc.rotation

        # construct 4x4 homogeneous matrix to extract quaternion (for ROS)
        # np.vstack and np.hstack are used to stack arrays vertically and horizontally
        # because TF in ROS uses quaternion representation for rotation
        T_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array([[0, 0, 0, 1]])))
        q = tf_transformations.quaternion_from_matrix(T_matrix)

        # publish TF of World -> Oc
        t_center = TransformStamped() # create a TransformStamped message, which is used to represent a TF
        t_center.header.stamp = now # set the timestamp

        # define the parent and child frame IDs
        t_center.header.frame_id = 'world'
        t_center.child_frame_id = 'Oc'

        # p is the translation vector, q is the quaternion e.g. [x, y, z, w]
        t_center.transform.translation.x = p[0]
        t_center.transform.translation.y = p[1]
        t_center.transform.translation.z = p[2]
        t_center.transform.rotation.x = q[0]
        t_center.transform.rotation.y = q[1]
        t_center.transform.rotation.z = q[2]
        t_center.transform.rotation.w = q[3]
        self.br.sendTransform(t_center) # broadcast the TF

        # Publish TF of Oc → O0~O7 
        for i, T in enumerate(self.frames):
            tf = TransformStamped() 
            tf.header.stamp = now

            # define the parent and child frame IDs
            tf.header.frame_id = 'Oc'
            tf.child_frame_id = f'O{i}' # frame O0~O7

            # set translation and rotation (identity rotation here)
            tf.transform.translation.x = T.translation[0]  # T.translation[0] is the x-coordinate
            tf.transform.translation.y = T.translation[1]  # T.translation[1] is the y-coordinate
            tf.transform.translation.z = T.translation[2]  # T.translation[2] is the z-coordinate
            tf.transform.rotation.x = 0.0  # identity rotation
            tf.transform.rotation.y = 0.0
            tf.transform.rotation.z = 0.0
            tf.transform.rotation.w = 1.0  # identity rotation
            self.br.sendTransform(tf)

        # print the current location
        self.get_logger().info(f"T_oc pos: {p}, yaw ≈ {np.arctan2(R[1,0], R[0,0]):.2f} rad")

def main(args=None):
    # initialize ROS2
    rclpy.init(args=args)
    # create the node
    node = T11Node()
    # spin the node to keep it running
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__': # run the main function
    main()

"""
Process:
1. run the node: ros2 run ros_visuals t11
2. initialize ros2 and create the node
3. go into the spin loop to keep the node running
4. in the node constructor:
   a. define the 8 corner frames O0~O7 using pinocchio SE(3)
   b. initialize the center frame T_oc and time step dt
   c. set up the TransformBroadcaster and timer to call broadcast_frames every 0.1s
5. in broadcast_frames function:
   a. get the current timestamp
   b. define the twist (angular and linear velocity)
   c. compute the incremental transform delta_T using exp6
   d. update the center frame T_oc
   e. extract translation and rotation from T_oc
   f. publish the TF from world to Oc
   g. publish the TFs from Oc to O0~O7
   h. print the current position and yaw angle of T_oc in the world frame
"""