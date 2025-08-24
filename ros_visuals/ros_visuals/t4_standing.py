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
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped

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
        # self._model 是一个 pin.Model 对象，它包含的是刚体结构
        # self._data = self._model.createData() 是配套的数据结构，用于临时缓存 forward kinematics、Jacobians、CoM 位置等计算中间量
        self._data = self._model.createData()  # 在 Talos 类中 没有定义 self._data（Pinocchio 的 Data 对象），但在 main() 里用了 robot._data
        
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

# main 函数是程序的入口点，负责初始化 ROS、创建机器人和模拟器实例，并运行主循环
def main(): 
    # node = rclpy.create_node('tutorial_4_standing_node')
    rclpy.init() # Initialize ROS
   
    # TODO init TSIDWrapper
    tsid_wrapper = TSIDWrapper(conf) # 初始化TSID控制器，传入配置文件
    
    # TODO init Simulator
    simulator = PybulletWrapper() # 初始化模拟器，使用PyBullet作为物理引擎
    
    # TODO init ROBOT
    robot = Talos(simulator, conf.urdf, tsid_wrapper.robot,  # 初始化机器人模型
                  q=conf.q_home, useFixedBase=False)
    
    t_publish = 0.0

    while rclpy.ok(): # Main loop


        # elaped time
        t = simulator.simTime()

        # TODO: update the simulator and the robot
        simulator.step() # 更新模拟器状态
        robot.update() # 更新机器人状态
        
        # TODO: update TSID controller
        # Fix: pass q, v, and t to the update method
        # Ensure q and v are float64 numpy arrays for C++ bindings
        # Call q and v as methods
        current_q = np.asarray(robot.q(), dtype=np.float64) # 获取当前关节位置
        # Ensure current_v is a float64 numpy array
        current_v = np.asarray(robot.v(), dtype=np.float64) # 获取当前关节速度
        # Update TSID controller with current state and time
        tau_sol, dv_sol = tsid_wrapper.update(current_q, current_v, t) # 更新TSID控制器，获取关节力矩和速度增量
        # command to the robot
        robot.setActuatedJointTorques(tau_sol) # 设置机器人关节的力矩命令

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            # get current BASE Pose
            T_b_w = tsid_wrapper.baseState()
            robot.publish(T_b_w)
            # Process ROS callbacks
            rclpy.spin_once(robot._node, timeout_sec=0)
    
if __name__ == '__main__':  # 程序执行从这里开始，进入主循环。
    # rclpy.init()
    main()