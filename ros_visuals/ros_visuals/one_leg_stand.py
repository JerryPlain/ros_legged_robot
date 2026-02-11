"""Tutorial 4 - Exercise 2: Balance on one foot with TSID.

Behavior:
1) Shift CoM toward the right foot (keep CoM height unchanged).
2) After a delay, remove left-foot contact.
3) Command the left foot to a fixed lifted pose.
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


PUBLISH_HZ = 30.0
WORLD_FRAME = "world"
BASE_FRAME = "base_link"
CONTACT_SWITCH_TIME = 2.0
LEFT_FOOT_LIFT = 0.30


class Talos(Robot):
    """Talos robot wrapper with ROS publishers for RViz visualization."""

    def __init__(self, simulator, urdf, tsid_robot, q=None, verbose=True, useFixedBase=False):
        super().__init__(
            simulator,
            urdf,
            tsid_robot.model(),
            basePosition=np.array([0.0, 0.0, 1.15]),
            q=q,
            verbose=verbose,
            useFixedBase=useFixedBase,
        )
        self._node = rclpy.create_node("tutorial_4_one_leg_stand_node")
        self._joint_state_pub = self._node.create_publisher(JointState, "joint_states", 10)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)

    def update(self):
        """Update robot internal state from simulator."""
        super().update()

    def publish(self, base_state):
        """Publish `joint_states` and TF `world -> base_link`."""
        stamp = self._node.get_clock().now().to_msg()
        n_actuated = len(self.actuatedJointNames())

        q = self.q()
        v = self.v()

        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.name = self.actuatedJointNames()
        joint_msg.position = q[7:7 + n_actuated].tolist()
        joint_msg.velocity = v[6:6 + n_actuated].tolist()
        joint_msg.effort = [0.0] * n_actuated
        self._joint_state_pub.publish(joint_msg)

        T_b_w, _ = base_state
        xyzquat = pin.SE3ToXYZQUAT(T_b_w)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = WORLD_FRAME
        tf_msg.child_frame_id = BASE_FRAME
        tf_msg.transform.translation.x = float(xyzquat[0])
        tf_msg.transform.translation.y = float(xyzquat[1])
        tf_msg.transform.translation.z = float(xyzquat[2])
        tf_msg.transform.rotation.x = float(xyzquat[3])
        tf_msg.transform.rotation.y = float(xyzquat[4])
        tf_msg.transform.rotation.z = float(xyzquat[5])
        tf_msg.transform.rotation.w = float(xyzquat[6])
        self._tf_broadcaster.sendTransform(tf_msg)


def main():
    """Run one-leg balancing behavior using TSID."""
    rclpy.init()

    tsid_wrapper = TSIDWrapper(conf)
    simulator = PybulletWrapper()
    robot = Talos(
        simulator,
        conf.urdf,
        tsid_wrapper.robot,
        q=conf.q_home,
        useFixedBase=False,
    )

    # Build a constant CoM target: right-foot XY, current CoM Z.
    # 这样做的目的是让机器人把重心移到右脚上方，增加右脚的支持力，减少左脚的负载，从而更容易单脚站立。
    p_com0 = tsid_wrapper.comState().value() # Get the current CoM position from TSID state
    p_rf0 = tsid_wrapper.get_placement_RF().translation # Get the current position of the right foot
   
    com_target = np.array([p_rf0[0], p_rf0[1], p_com0[2]]) # Set CoM target to be above the right foot, but keep the same height as current CoM

    # 下面的控制循环里，机器人会先在双脚支撑上把 CoM 移到右脚上方，增加右脚的支持力。然后在 CONTACT_SWITCH_TIME 秒后，切换到单脚支撑，抬起左脚。TSID 会根据当前状态和任务权重，自动计算出合适的关节力矩，让机器人保持平衡。
    publish_period = 1.0 / PUBLISH_HZ
    t_last_publish = 0.0
    t_start = simulator.simTime()

    left_contact_removed = False
    left_foot_target_pose = None

    try:
        while rclpy.ok():
            simulator.step()
            robot.update()

            t = simulator.simTime()
            q = np.asarray(robot.q(), dtype=np.float64)
            v = np.asarray(robot.v(), dtype=np.float64)

            # Keep CoM reference over the right support foot.
            tsid_wrapper.setComRefState(com_target)

            # After the transition time, switch to single support and lift LF.
            if (not left_contact_removed) and (t - t_start >= CONTACT_SWITCH_TIME):
                tsid_wrapper.remove_contact_LF()
                left_contact_removed = True

                lf_pose_now = tsid_wrapper.get_placement_LF()
                left_foot_target_pose = pin.SE3(
                    lf_pose_now.rotation.copy(),
                    lf_pose_now.translation.copy(),
                )
                left_foot_target_pose.translation[2] += LEFT_FOOT_LIFT

            if left_contact_removed:
                tsid_wrapper.set_LF_pose_ref(left_foot_target_pose)

            tau_sol, _ = tsid_wrapper.update(q, v, t)
            robot.setActuatedJointTorques(tau_sol)

            if t - t_last_publish >= publish_period:
                t_last_publish = t
                robot.publish(tsid_wrapper.baseState())
                rclpy.spin_once(robot._node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        robot._node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
