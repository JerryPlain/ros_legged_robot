import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from ros_visuals.tsid_wrapper import TSIDWrapper
import ros_visuals.config as conf

# ROS
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState # for publishing joint states
from geometry_msgs.msg import PoseStamped # for publishing poses
from geometry_msgs.msg import TransformStamped # for broadcasting transforms

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot):
    # 构造函数 __init__ 初始化机器人模型、ROS节点、消息发布者和tf广播器
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False):
        # TODO call base class constructor
        # Fix: set proper base position to avoid ground collision
        super().__init__(simulator, urdf, model.model(), 
                        basePosition=np.array([0, 0, 1.15]), 
                        q=q, verbose=verbose, useFixedBase=useFixedBase)
        self._node = rclpy.create_node('tutorial_4_standing_node')
        self._tsid_model = model  # Store TSID model separately
        
        # TODO add publisher
        self._joint_state_pub = self._node.create_publisher(JointState, 'joint_states', 10)
        
        # TODO add tf broadcaster
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)
        pass

    # 更新函数 update 用于更新机器人状态
    def update(self):
        # TODO update base class
        super().update()
        pass

    # publish(T_b_w) 负责将机器人关节状态通过 ROS 发布出去，并发布 base_link 相对于 world 的变换（tf）
    def publish(self, T_b_w):
        # TODO publish jointstate
        joint_msg = JointState()
        joint_msg.header.stamp = self._node.get_clock().now().to_msg()
        joint_msg.name = [name for name in self._model.names[1:]]  # Skip universe
        joint_msg.position = self.q()[7:].tolist()  # Skip floating base (7 DOF), call q as a method
        joint_msg.velocity = self.v()[6:].tolist() if hasattr(self, 'v') else [0.0] * len(joint_msg.position)
        joint_msg.effort = [0.0] * len(joint_msg.position)
        self._joint_state_pub.publish(joint_msg)
        
        # TODO broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self._node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"
        
        # T_b_w is (SE3_object, Motion_object) tuple
        se3_obj, motion_obj = T_b_w
        xyzquat = pin.SE3ToXYZQUAT(se3_obj)

        tf_msg.transform.translation.x = float(xyzquat[0])
        tf_msg.transform.translation.y = float(xyzquat[1])
        tf_msg.transform.translation.z = float(xyzquat[2])
        tf_msg.transform.rotation.x = float(xyzquat[3])
        tf_msg.transform.rotation.y = float(xyzquat[4])
        tf_msg.transform.rotation.z = float(xyzquat[5])
        tf_msg.transform.rotation.w = float(xyzquat[6])
        
        self._tf_broadcaster.sendTransform(tf_msg)
        
        pass

################################################################################
# main
################################################################################

def main(): 
    # first, instantiate everything before the main loop
    # create a node
    # rclpy.init() # Initialize ROS
    # Create a node for the ROS 2 system
    # node = rclpy.create_node('tutorial_4_standing_node')
    rclpy.init() # Initialize ROS
    node = rclpy.create_node('tutorial_4_one_leg_stand_node')
    
    # TODO init TSIDWrapper
    tsid_wrapper = TSIDWrapper(conf)  # Initialize TSID controller with configuration
    # TODO init Simulator
    simulator = PybulletWrapper()  # 初始化仿真环境
    # TODO init ROBOT
    robot = Talos(simulator, conf.urdf, tsid_wrapper.robot,  # 初始化机器人模型
                  q=conf.q_home, useFixedBase=False)
    
    t_publish = 0.0
    # Define a tiktok time 
    start_time = simulator.simTime()  # 记录开始时间，方便后续计时

    while rclpy.ok(): # Main loop

        # elaped time
        t = simulator.simTime() # 当前仿真时间

        # TODO: update the simulator and the robot
        simulator.step() # 更新模拟器状态
        robot.update() # 更新机器人状态

        # TODO: update TSID controller
        current_q = np.asarray(robot.q(), dtype=np.float64)
        current_v = np.asarray(robot.v(), dtype=np.float64)
        tau_sol, dv_sol = tsid_wrapper.update(current_q, current_v, t)

        # command to the robot
        robot.setActuatedJointTorques(tau_sol)

        # update the COM, 参考到RF XY， 保持Z不变
        # get the current COM position
        p_com = tsid_wrapper.comState().pos()
        # get the right foot position
        p_rf = tsid_wrapper.get_placement_RF().translation
        # construct a new COM position - X,Y取右脚，Z保持COM原高度
        new_com_ref = np.array([p_rf[0], p_rf[1], p_com[2]])
        # set the new COM reference
        tsid_wrapper.setComRefState(new_com_ref)

        # 2秒后移除左脚接触，抬起左脚
        if t - start_time > 2.0:
            # remove left foot contact
            tsid_wrapper.remove_contact_LF()
            # lift the left foot 获取左脚位置，Z轴抬高0.3米
            p_lf = tsid_wrapper.get_placement_LF()
            # p_lf[2] += 0.3 这里会报错，因为SE3不能这样用
            p_lf.translation[2] += 0.3 # 修改z轴高度

            # set the new left foot placement
            tsid_wrapper.set_LF_pose_ref(p_lf)

        # publish to ros 发布机器人状态和ROS回调，保持不变
        if t - t_publish > 1./30.:
            t_publish = t
            # get current BASE Pose
            T_b_w = tsid_wrapper.baseState()
            robot.publish(T_b_w)
            rclpy.spin_once(robot._node, timeout_sec=0)
            
    
if __name__ == '__main__': 
    rclpy.init()
    main()
