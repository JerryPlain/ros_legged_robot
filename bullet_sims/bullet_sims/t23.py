import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def main():
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    z_init = 1.15
    q_actuated_home = np.zeros(32)
    q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

    modelWrap = pin.RobotWrapper.BuildFromURDF(urdf, path_meshes, pin.JointModelFreeFlyer(), True, None)
    model = modelWrap.model
    data = model.createData()

    n = model.nv - 6
    Kp_diag = np.ones(n)
    Kd_diag = np.ones(n)
    Kp_diag[0:12] = 800.0; Kd_diag[0:12] = 60.0
    Kp_diag[12:14] = 150.0; Kd_diag[12:14] = 20.0
    Kp_diag[14:22] = 2.0; Kd_diag[14:22] = 0.2
    Kp_diag[22:30] = 2.0; Kd_diag[22:30] = 0.2
    Kp_diag[30:32] = 2; Kd_diag[30:32] = 0.2
    Kp = np.diag(Kp_diag)
    Kd = np.diag(Kd_diag)

    simulator = PybulletWrapper(sim_rate=1000)
    robot = Robot(simulator, urdf, model, [0, 0, z_init], [0, 0, 0, 1], q=q_home, useFixedBase=False, verbose=True)
    simulator.addLinkDebugFrame(-1, -1)
    simulator.step()
    robot.update()

    # initialize q_home
    q_ini = q_home.copy()
    q_home = pin.neutral(model)
    q_home[2] = z_init
    q_home[3:7] = [0, 0, 0, 1]
    q_home[7+0:7+6] = np.array([0, 0, -0.44, 0.9, -0.45, 0])
    q_home[7+6:7+12] = np.array([0, 0, -0.44, 0.9, -0.45, 0])
    q_home[7+12:7+14] = np.array([0.0, 0.0])
    q_home[7+14:7+21] = np.array([0, -0.24, 0, -1, 0, 0, 0])
    q_home[7+21:7+28] = np.array([0, -0.24, 0, -1, 0, 0, 0])
    q_home[7+28:7+30] = np.array([0, 0])

    T_total = 0.5
    t_start = time.time()

    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=90, cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8])

    # initialize ROS2 publisher
    rclpy.init()
    node = rclpy.create_node('joint_state_publisher')
    publisher = node.create_publisher(JointState, 'joint_states', 10)
    joint_names = robot.actuatedJointNames()

    def publish_joint_states(q, dq, tau):
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = q.tolist()
        msg.velocity = dq.tolist()
        msg.effort = tau.tolist()
        publisher.publish(msg)

    # main loop
    last_pub_time = time.time()
    while rclpy.ok():
        simulator.step()
        simulator.debug()
        robot.update()

        t_now = time.time()
        elapsed = t_now - t_start

        if elapsed < T_total:
            alpha = min(elapsed / T_total, 1.0)
            q_desired = pin.interpolate(model, q_ini, q_home, alpha)
        else:
            q_desired = q_home.copy()

        q = robot.q()[7:]
        dq = robot.v()[6:]
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())
        b_actuated = b[6:]
        tau = b_actuated + Kp @ (q_desired[7:] - q) - Kd @ dq
        tau = np.clip(tau, -200, 200)
        robot.setActuatedJointTorques(tau)

        # 每 1/30 秒发布一次 joint_states
        if time.time() - last_pub_time > 1.0 / 30.0:
            publish_joint_states(q, dq, tau)
            last_pub_time = time.time()

    rclpy.shutdown()


if __name__ == '__main__':
    main()