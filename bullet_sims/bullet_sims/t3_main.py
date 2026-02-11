import numpy as np

from enum import Enum

from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import rclpy
from sensor_msgs.msg import JointState

from scipy.interpolate import CubicHermiteSpline


# Two-phase control flow in this tutorial:
# 1) JOINT_SPLINE: move from q_init to q_home with joint-space inverse dynamics.
# 2) CART_SPLINE: switch to Cartesian control and hold right-hand pose.
class State(Enum):
    JOINT_SPLINE = 0
    CART_SPLINE = 1


class Talos(Robot):
    def __init__(self, simulator, node, q=None, verbose=True, use_fixed_base=True):
        urdf_path = "src/talos_description/robots/talos_reduced.urdf"

        # Build a fixed-base Pinocchio model from URDF.
        self._wrapper = RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=[],
            root_joint=None,
        )

        super().__init__(
            simulator=simulator,
            filename=urdf_path,
            model=self._wrapper.model,
            q=q,
            useFixedBase=use_fixed_base,
            verbose=verbose,
        )

        self.node = node
        self.publisher = self.node.create_publisher(JointState, "/joint_states", 10)
        # Pinocchio fixed-base: names[0] is "universe", actuated joints start at index 1.
        self.joint_names = self._wrapper.model.names[1:]

    def wrapper(self):
        return self._wrapper

    def data(self):
        return self._wrapper.data

    def update(self):
        super().update()

        # Get latest state from simulator in Pinocchio convention.
        q_current = self.q()
        v_current = self.v()

        # Keep kinematics data current for controllers.
        pin.forwardKinematics(self._wrapper.model, self._wrapper.data, q_current, v_current)
        pin.updateFramePlacements(self._wrapper.model, self._wrapper.data)

    def publish(self):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.q().tolist()
        msg.velocity = self.v().tolist()
        self.publisher.publish(msg)


class JointSpaceController:
    def __init__(self, robot, kp, kd):
        self.robot = robot
        self.kp = kp
        self.kd = kd

    def update(self, q_r, q_r_dot, q_r_ddot):
        # q_r, q_r_dot, q_r_ddot are time-varying references from the spline.
        # The controller is a tracker (not just a fixed-posture regulator).
        q = self.robot.q()
        q_dot = self.robot.v()

        model = self.robot.wrapper().model
        data = self.robot.wrapper().data

        # Dynamics terms for feedback linearization:
        # M(q) from CRBA and h(q, qdot) from nonLinearEffects.
        m_mat = pin.crba(model, data, q)
        h_vec = pin.nonLinearEffects(model, data, q, q_dot)

        e = q - q_r
        e_dot = q_dot - q_r_dot

        # Eq.(4): tau = M(q) * (qddot_ref - Kd*edot - Kp*e) + h(q, qdot)
        tau = m_mat @ (q_r_ddot - self.kd @ e_dot - self.kp @ e) + h_vec
        return tau


class CartesianSpaceController:
    def __init__(self, robot, joint_name, kp, kd, damping=1e-6):
        self.robot = robot
        self.kp = kp
        self.kd = kd
        self.damping = damping

        model = self.robot.wrapper().model
        # Convert human-readable frame name to fast integer id once.
        self.frame_id = model.getFrameId(joint_name)

    def update(self, x_r, x_dot_r, x_ddot_r):
        q = self.robot.q()
        v = self.robot.v()

        model = self.robot.wrapper().model
        data = self.robot.wrapper().data

        pin.forwardKinematics(model, data, q, v)
        pin.updateFramePlacement(model, data, self.frame_id)

        # Differential map between joint-space and task-space.
        j_mat = pin.computeFrameJacobian(model, data, q, self.frame_id, pin.LOCAL)

        x_cur = data.oMf[self.frame_id]
        x_dot_cur = pin.getFrameVelocity(model, data, self.frame_id, pin.LOCAL).vector

        # Pose error in SE(3) tangent space (6D: translation + rotation).
        x_err = pin.log(x_r.inverse() * x_cur).vector
        x_dot_err = x_dot_cur - x_dot_r
        x_ddot_des = x_ddot_r - self.kd @ x_dot_err - self.kp @ x_err

        # From Xddot = J*qddot + Jdot*qdot, this is the Jdot*qdot term.
        a_frame = pin.getFrameClassicalAcceleration(model, data, self.frame_id, pin.LOCAL)
        jdot_v = a_frame.vector

        # Damped pseudo-inverse for numerical robustness near singularities.
        jj_t = j_mat @ j_mat.T
        j_pinv = j_mat.T @ np.linalg.inv(jj_t + self.damping * np.eye(6))

        m_mat = pin.crba(model, data, q)
        h_vec = pin.nonLinearEffects(model, data, q, v)

        # Eq.(7) idea:
        # 1) map desired task acceleration to qddot via J#,
        # 2) map qddot to torques with inverse dynamics.
        tau = m_mat @ j_pinv @ (x_ddot_des - jdot_v) + h_vec
        return tau


class Environment:
    def __init__(self, node):
        self.node = node
        self.cur_state = State.JOINT_SPLINE

        self.simulator = PybulletWrapper()

        # 32 DoF fixed-base Talos reduced model.
        self.q_init = np.zeros(32)
        self.q_home = np.zeros(32)
        self.q_home[14:22] = np.array([0.0, 0.45, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        self.q_home[22:30] = np.array([0.0, -0.45, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])

        self.robot = Talos(self.simulator, node=self.node, q=self.q_init)

        self.duration = 5.0
        # One spline per joint. Boundary velocities are zero for smooth start/stop.
        self.q_splines = [
            CubicHermiteSpline(
                [0.0, self.duration],
                [self.q_init[i], self.q_home[i]],
                [0.0, 0.0],
            )
            for i in range(32)
        ]

        self.joint_controller = JointSpaceController(
            self.robot,
            kp=np.diag([100.0] * 32),
            kd=np.diag([10.0] * 32),
        )

        self.cartesian_controller = CartesianSpaceController(
            self.robot,
            joint_name="arm_right_7_joint",
            kp=np.diag([400.0] * 6),
            kd=np.diag([40.0] * 6),
        )

        self.switch_to_cartesian = False
        self.x_goal = None

        # Publish joint states at 100 Hz to ROS.
        self.t_publish = 0.0
        self.publish_period = 0.01

    def update(self, t, dt):
        self.robot.update()

        if self.cur_state == State.JOINT_SPLINE:
            # Clip time so references stay at final point after duration.
            t_clipped = min(t, self.duration)
            q_r = np.array([spline(t_clipped) for spline in self.q_splines])
            q_r_dot = np.array([spline.derivative(1)(t_clipped) for spline in self.q_splines])
            q_r_ddot = np.array([spline.derivative(2)(t_clipped) for spline in self.q_splines])

            tau = self.joint_controller.update(q_r, q_r_dot, q_r_ddot)

            if t > self.duration and not self.switch_to_cartesian:
                self.node.get_logger().info("Switch to cartesian control")
                self.switch_to_cartesian = True
                self.cur_state = State.CART_SPLINE

                # Save current right-hand pose as initial Cartesian goal to avoid jump at switch.
                frame_id = self.robot.wrapper().model.getFrameId("arm_right_7_joint")
                self.x_goal = self.robot.wrapper().data.oMf[frame_id].copy()

        else:
            # Hold the captured Cartesian pose (zero desired velocity/acceleration).
            x_r = self.x_goal
            x_dot_r = np.zeros(6)
            x_ddot_r = np.zeros(6)
            tau = self.cartesian_controller.update(x_r, x_dot_r, x_ddot_r)

        # Send torque command to actuated joints in simulator.
        self.robot.setActuatedJointTorques(tau)

        self.t_publish += dt
        if self.t_publish >= self.publish_period:
            self.robot.publish()
            self.t_publish -= self.publish_period


def main():
    # Single ROS node shared across app + joint state publishing.
    rclpy.init()
    node = rclpy.create_node("tutorial_3_robot_sim")

    env = Environment(node=node)

    try:
        while rclpy.ok():
            t = env.simulator.simTime()
            dt = env.simulator.stepTime()

            env.update(t, dt)
            env.simulator.debug()
            env.simulator.step()

            rclpy.spin_once(node, timeout_sec=0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
