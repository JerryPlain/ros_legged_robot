import pybullet as pb
import numpy as np
import pinocchio as pin
import time

from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def main():
    """
    Tutorial 2 – Exercise 4
    本质上是在 Exercise 3 的基础上加了一层“ROS2 标准接口”：把机器人关节状态发布到 joint_states，让 robot_state_publisher 和 RViz 能画出来。
    - 创建node
    - 创建publisher
    - 构建并发布sensor_msgs/JointState消息
    目的：让 RViz 通过 robot_state_publisher 可视化机器人姿态，而不是只在 PyBullet 里看。
    
    ----------------------
    Visualization of the Talos robot state using ROS 2.

    This script:
    1) Runs a PyBullet simulation with Pinocchio-based control
    2) Drives the robot to a stable home posture using interpolation
    3) Publishes joint states (position, velocity, effort) to ROS 2
       at a throttled rate (30 Hz)
    4) Enables visualization with robot_state_publisher and RViz
    """

    # ============================================================
    # Robot description
    # ============================================================
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    # ============================================================
    # Initial configuration
    # ============================================================
    z_init = 1.15

    # Talos has 32 actuated joints
    q_actuated_init = np.zeros(32)

    # Full Pinocchio configuration:
    # q = [base position (3), base quaternion (4), joint positions (32)]
    q_init = np.hstack([
        np.array([0.0, 0.0, z_init, 0.0, 0.0, 0.0, 1.0]),
        q_actuated_init
    ])

    # ============================================================
    # Build Pinocchio model (floating base)
    # ============================================================
    model_wrap = pin.RobotWrapper.BuildFromURDF(
        urdf,
        path_meshes,
        pin.JointModelFreeFlyer(),
        True,
        None
    )
    model = model_wrap.model
    data = model.createData()

    # ============================================================
    # Joint-space PD gains
    # ============================================================
    # model.nv = 6 (floating base) + 32 (actuated joints)
    n = model.nv - 6

    Kp_diag = np.ones(n)
    Kd_diag = np.ones(n)

    # Legs (load-bearing)
    Kp_diag[0:12] = 800.0
    Kd_diag[0:12] = 60.0

    # Torso
    Kp_diag[12:14] = 150.0
    Kd_diag[12:14] = 20.0

    # Arms
    Kp_diag[14:30] = 2.0
    Kd_diag[14:30] = 0.2

    # Head
    Kp_diag[30:32] = 2.0
    Kd_diag[30:32] = 0.2

    Kp = np.diag(Kp_diag)
    Kd = np.diag(Kd_diag)

    # ============================================================
    # Initialize simulator and robot
    # ============================================================
    simulator = PybulletWrapper(sim_rate=1000)

    robot = Robot(
        simulator,
        urdf,
        model,
        base_pos=[0.0, 0.0, z_init],
        base_quat=[0.0, 0.0, 0.0, 1.0],
        q=q_init,
        useFixedBase=False,
        verbose=True
    )

    simulator.addLinkDebugFrame(-1, -1)

    simulator.step()
    robot.update()

    # ============================================================
    # Initial and home configurations
    # ============================================================
    q_ini = robot.q().copy()

    # Home posture (full Pinocchio configuration)
    q_home = pin.neutral(model)
    q_home[2] = z_init
    q_home[3:7] = [0.0, 0.0, 0.0, 1.0]

    # Legs
    q_home[7:13]  = [0, 0, -0.44, 0.9, -0.45, 0]
    q_home[13:19] = [0, 0, -0.44, 0.9, -0.45, 0]

    # Torso
    q_home[19:21] = [0.0, 0.0]

    # Arms
    q_home[21:28] = [0, -0.24, 0, -1, 0, 0, 0]
    q_home[28:35] = [0, -0.24, 0, -1, 0, 0, 0]

    # Head
    q_home[35:37] = [0.0, 0.0]

    # ============================================================
    # Interpolation parameters
    # ============================================================
    T_total = 0.5  # seconds
    t_start = time.time()

    # ============================================================
    # PyBullet visualization
    # ============================================================
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8]
    )

    # ============================================================
    # ROS 2 initialization
    # ============================================================
    """
    Overall data flow:
    PyBullet → Robot wrapper → Pinocchio → 计算 τ → 写回 PyBullet
    同时：
    Robot wrapper → ROS JointState → robot_state_publisher → TF → RViz RobotModel
    """

    """
    rclpy.init()：启动 ROS2 通信系统（DDS），没它你发不出去消息
    node = rclpy.create_node("talos_joint_state_publisher")：创建一个 ROS2 节点，节点名叫 talos_joint_state_publisher
    publisher = node.create_publisher(JointState, "joint_states", 10)：创建一个发布器，发布 JointState 消息到 joint_states 话题，队列长度10
    之后在主循环里调用 publish_joint_states(q, dq, tau) 来发布消息
    这样 talos_joint_state_publisher 节点才能订阅到 joint_states 话题并进行可视化
    具体参考 ROS2 官方教程：https://docs.ros.org/en/foxy/Tutorials/Intermediate/Publishing-and-Subscribing-to-Topics
    """
    rclpy.init()
    node = rclpy.create_node("talos_joint_state_publisher")
    publisher = node.create_publisher(JointState, "joint_states", 10)

    joint_names = robot.actuatedJointNames()

    def publish_joint_states(q, dq, tau):
        """
        Publish joint positions, velocities and efforts
        to the ROS 2 joint_states topic.
        """
        msg = JointState()

        # Fill in the message fields
        msg.header.stamp = node.get_clock().now().to_msg() # current time
        msg.name = joint_names # list of joint names

        # Pinocchio q, v, tau are numpy arrays; convert to lists for ROS messages
        msg.position = q.tolist()
        msg.velocity = dq.tolist()
        msg.effort = tau.tolist()

        # Publish the joint state message
        publisher.publish(msg)

    # ============================================================
    # Main control loop
    # ============================================================
    last_pub_time = time.time()

    while rclpy.ok():
        simulator.step()
        simulator.debug()
        robot.update()

        # --------------------------------------------------------
        # Interpolated desired configuration
        # --------------------------------------------------------
        elapsed = time.time() - t_start
        alpha = min(elapsed / T_total, 1.0)
        q_des = pin.interpolate(model, q_ini, q_home, alpha)

        # --------------------------------------------------------
        # Read actuated joint states
        # --------------------------------------------------------
        q = robot.q()[7:]     # joint positions
        dq = robot.v()[6:]    # joint velocities

        # --------------------------------------------------------
        # Nonlinear effects (gravity + Coriolis)
        # --------------------------------------------------------
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())
        b_actuated = b[6:]

        # --------------------------------------------------------
        # Joint-space PD control
        # --------------------------------------------------------
        tau = (
            b_actuated
            + Kp @ (q_des[7:] - q)
            - Kd @ dq
        )

        tau = np.clip(tau, -200.0, 200.0)
        robot.setActuatedJointTorques(tau)

        # --------------------------------------------------------
        # Throttled ROS 2 publication (30 Hz)
        # --------------------------------------------------------
        if time.time() - last_pub_time > 1.0 / 30.0:
            publish_joint_states(q, dq, tau) # publish current joint states by calling the function; 这个节点只需要发布，不订阅、不定时器、不服务
            last_pub_time = time.time()

    rclpy.shutdown()


if __name__ == "__main__":
    main()