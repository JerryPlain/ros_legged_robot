"""Tutorial 4 - Exercise 1: Stand still with TSID.

This script runs a floating-base Talos model in PyBullet and closes the loop
with TSID at torque level. 
-> Use TSID (Task-Space Inverse Dynamics) to compute the required joint torques to maintain a standing posture.
It also publishes:
1) `joint_states` for RViz
2) `world -> base_link` TF
"""

import numpy as np
import pinocchio as pin

from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

from ros_visuals.tsid_wrapper import TSIDWrapper
import ros_visuals.config as conf

import rclpy
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

# ROS publishing settings
PUBLISH_HZ = 30.0
WORLD_FRAME = "world"
BASE_FRAME = "base_link"


class Talos(Robot):
    # inherits from Robot class and adds ROS publishers for joint states and TF broadcasting

    """Talos robot wrapper with ROS publishers used in this tutorial."""

    def __init__(self, simulator, urdf, tsid_robot, q=None, verbose=True, useFixedBase=False):
        # Build pybullet+pinocchio robot interface from TSID model.
        super().__init__(
            simulator,
            urdf,
            tsid_robot.model(),
            basePosition=np.array([0.0, 0.0, 1.15]), # Set proper base position to avoid ground collision
            q=q,
            verbose=verbose,
            useFixedBase=useFixedBase, # We want a floating base for this tutorial
        )
        # create ROS node, publishers, and TF broadcaster
        self._node = rclpy.create_node("tutorial_4_standing_node")
        self._joint_state_pub = self._node.create_publisher(JointState, "joint_states", 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)

    def update(self):
        """Update robot internal state from simulator."""
        super().update()

    def publish(self, base_state):
        # TF（world -> base_link）告诉 RViz：机器人“整体”在世界哪里、朝向如何。
        # joint_states 告诉 RViz：机器人各个关节的角度位置是什么。
        # 只发 TF：机器人能移动到正确位置，但关节不会按当前姿态弯曲。
        # 只发 joint_states：关节形状对了，但整台机器人会卡在默认原点/默认朝向。
        # 两者结合，RViz 才能显示“在世界中正确位置和姿态”的整台机器人。

        """Publish joint states and floating-base TF."""
        stamp = self._node.get_clock().now().to_msg()
        n_actuated = len(self.actuatedJointNames()) # Number of actuated joints (excluding floating base)

        # base: 描述机器人整体在世界中的“漂浮位姿”（平移+朝向），是自由飞体自由度。
        # joints: 描述机器人各个关节的角度位置，是机械臂的自由度。
        # q = [x, y, z, qx, qy, qz, qw, joints...] q = [base_pose(7), joint_positions(n)]
        # v = [vx, vy, vz, wx, wy, wz, joints...] v = [base_twist(6), joint_velocities(n)]
        q = self.q()
        v = self.v()

        # Publish joint states (exclude floating base) and TF for base_link relative to world
        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.name = self.actuatedJointNames()
        joint_msg.position = q[7:7 + n_actuated].tolist() # Skip floating base (7 DOF) and take only joint positions
        joint_msg.velocity = v[6:6 + n_actuated].tolist() # Skip floating base velocities
        joint_msg.effort = [0.0] * n_actuated
        self._joint_state_pub.publish(joint_msg)

        # 把机器人底座 base_link 在世界坐标 world 里的位姿，发成 ROS TF，这样 RViz 就能正确显示机器人在世界中的位置和朝向。
        T_b_w, _ = base_state # base_state is (T_b_w, v_b_w) where T_b_w is the SE3 pose of the base in the world frame
        xyzquat = pin.SE3ToXYZQUAT(T_b_w) # Convert SE3 pose to (x, y, z, qx, qy, qz, qw) format for ROS TF

        # Publish TF for base_link relative to world
        tf_msg = TransformStamped() # Create a TransformStamped message for ROS TF
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = WORLD_FRAME # "world" father frame
        tf_msg.child_frame_id = BASE_FRAME # "base_link" child frame

        # Fill in the translation and rotation from the SE3 pose
        tf_msg.transform.translation.x = float(xyzquat[0])
        tf_msg.transform.translation.y = float(xyzquat[1])
        tf_msg.transform.translation.z = float(xyzquat[2])
        tf_msg.transform.rotation.x = float(xyzquat[3])
        tf_msg.transform.rotation.y = float(xyzquat[4])
        tf_msg.transform.rotation.z = float(xyzquat[5])
        tf_msg.transform.rotation.w = float(xyzquat[6])
        self._tf_broadcaster.sendTransform(tf_msg) # 把这条变换广播出去，RViz 才知道机器人底座在哪、朝哪。

def main():
    # simulator 提供世界 -> robot 读状态 -> tsid_wrapper 算控制 -> robot 下发力矩到 simulator。
    """Initialize TSID + simulator and run standing control loop."""
    rclpy.init() # Initialize ROS

    tsid_wrapper = TSIDWrapper(conf) # Initialize TSID controller with configuration from config.py 控制器层负责建 QP 任务（CoM、接触、posture 等），输入 q,v，输出关节力矩 tau。
    simulator = PybulletWrapper() # Initialize Pybullet simulator
    
    # Initialize Talos robot in the simulator with the TSID model and homing configuration
    robot = Talos( # 桥接层（接口层）把 TSID 的模型格式、PyBullet 的状态/执行接口、ROS 发布整合在一起。通过它读 q,v、写 tau、发 joint_states/TF。
        simulator, 
        conf.urdf,
        tsid_wrapper.robot,
        q=conf.q_home,
        useFixedBase=False,
    )

    # main control loop: step simulator, update robot state, compute control, apply torques, publish ROS messages
    publish_period = 1.0 / PUBLISH_HZ
    t_last_publish = 0.0

    try:
        # 1. Step the simulator to advance time and get new state
        # 2. Update the robot's internal state (q, v) from the simulator
        # 3. Use TSID to compute the required joint torques (tau_sol) based on the current state and time
        # 4. Apply the computed torques to the robot in the simulator
        # 5. Periodically publish joint states and TF for visualization in RViz
        while rclpy.ok():
            simulator.step() # Step the simulator to advance time and get new state
            robot.update() # Update robot state from simulator (q, v)

            # Get current simulation time, joint positions, and velocities
            t = simulator.simTime() # Get current simulation time
            q = np.asarray(robot.q(), dtype=np.float64) # Get current joint positions (including floating base) as a numpy array
            v = np.asarray(robot.v(), dtype=np.float64) # Get current joint velocities (including floating base) as a numpy array

            tau_sol, _ = tsid_wrapper.update(q, v, t) # use TSID to compute control torques based on current state and time
            robot.setActuatedJointTorques(tau_sol) # Apply the computed torques to the robot in the simulator

            # Periodically publish joint states and TF for visualization in RViz
            if t - t_last_publish >= publish_period:
                t_last_publish = t
                # take the base state (pose and twist) from TSID and publish it along with joint states to ROS topics
                robot.publish(tsid_wrapper.baseState())
                rclpy.spin_once(robot._node, timeout_sec=0.0)
                
    except KeyboardInterrupt:
        pass
    finally:
        robot._node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
