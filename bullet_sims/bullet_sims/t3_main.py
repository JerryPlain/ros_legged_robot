import numpy as np
import numpy.linalg as la

# simulator (#TODO: set your own import path!)
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# modeling
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from enum import Enum

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from scipy.interpolate import CubicHermiteSpline

################################################################################
# utility functions
################################################################################

class State(Enum):
    JOINT_SPLINE = 0,
    CART_SPLINE = 1

################################################################################
# Robot
################################################################################

class Talos(Robot):
    def __init__(self, simulator, q=None, verbose=True, useFixedBase=True):
        #TODO: Create RobotWrapper (fixed base), Call base class constructor, make publisher  
        # set up the urdf_path
        urdf_path = "src/talos_description/robots/talos_reduced.urdf"
        # set up the pinocchio wrapper
        wrapper = RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs = [],
            root_joint = None # NO floating base
        )
       
        # extract model for PyBullet
        model = wrapper.model
        
        # call Robot
        super().__init__(
            simulator = simulator,
            filename = urdf_path,
            model = model,
            q = q,
            useFixedBase = useFixedBase,
            verbose = verbose
        )

        # save wrapper for controller
        self._wrapper = wrapper
        print("[INFO] model.nq =", self._wrapper.model.nq)  # should be 32

        # make publisher for joint states
        self.node = rclpy.create_node('talos_node') # create a node
        self.publisher = self.node.create_publisher(JointState, "/joint_states", 10)

        # save the joint names
        self.joint_names = model.names[2:]  # jump 'universe' & 'base_link'

    def update(self):
        # TODO: update base class, update pinocchio robot wrapper's kinematics
        super().update() # update the state of Pybullet
        
        q_current = self.q()
        v_current = self.v()
        self._q = q_current        # for publish
        self._q_dot = v_current

        # update the pinocchio model、
        pin.forwardKinematics(self._wrapper.model, self._wrapper.data, q_current, v_current)
        pin.updateFramePlacements(self._wrapper.model, self._wrapper.data)
   
    def wrapper(self):
        return self._wrapper

    def data(self):
        return self._wrapper.data
    
    def publish(self):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self._q.tolist()
        msg.velocity = self._q_dot.tolist()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        self.publisher.publish(msg)

################################################################################
# Controllers
################################################################################

class JointSpaceController:
    """JointSpaceController
    Tracking controller in jointspace
    """
    def __init__(self, robot, Kp, Kd):        
        # Save gains, robot ref
        self.robot = robot
        self.Kp = Kp
        self.Kd = Kd

    def update(self, q_r, q_r_dot, q_r_ddot):
        # Compute jointspace torque, return torque
        q = self.robot.q()
        q_dot = self.robot.v()

        model = self.robot.wrapper().model
        data = self.robot.wrapper().data

        M = pin.crba(model, data, q)
        h = pin.nonLinearEffects(model, data, q, q_dot)
        e = q - q_r
        e_dot = q_dot -q_r_dot
        tau = M @ (q_r_ddot - self.Kd @ e_dot - self.Kp @ e) + h

        return tau
        
class CartesianSpaceController:
    """CartesianSpaceController
    Tracking controller in cartspace
    """
    def __init__(self, robot, joint_name, Kp, Kd):
        # save gains, robot ref
        self.robot = robot
        self.joint_name = joint_name
        self.Kp = Kp
        self.Kd = Kd

        # acquire the joint ID and frame ID for control
        self.joint_id = self.robot.wrapper().model.getJointId(joint_name)
        self.frame_id = self.robot.wrapper().model.getFrameId(joint_name) # used for calcualtions
        
    def update(self, X_r, X_dot_r, X_ddot_r):
        # compute cartesian control torque, return torque

        # Step 1: get current state
        q = self.robot.q()
        v = self.robot.v()
        model = self.robot.wrapper().model
        data = self.robot.wrapper().data

        # Step 2: compute Jacobian
        pin.forwardKinematics(model, data, q, v)
        J = pin.computeFrameJacobian(model, data, q, self.frame_id, pin.LOCAL)

        # Step 3: get current Cartesian pose and velocity
        pin.updateFramePlacement(model, data, self.frame_id)
        oMf = data.oMf[self.frame_id] # data.oMf is a frame-placement table
        v_frame = pin.getFrameVelocity(model, data, self.frame_id, pin.LOCAL)

        # Step 4: compute desired Cartesian acceleration using Eq (8)
        err_pos = pin.log(X_r.inverse() * oMf).vector
        err_vel = v_frame.vector - X_dot_r
        X_ddot_d = X_ddot_r - self.Kd @ err_vel - self.Kp @ err_pos

        # Step 5: compute Jdot * v
        a_frame = pin.getFrameClassicalAcceleration(model, data, self.frame_id, pin.LOCAL)
        Jdot_v = a_frame.vector

        # Step 6: compute damped pseudo-inverse of Jacobian
        JJ_T = J @ J.T
        damping = 1e-6
        J_sharp = J.T @ np.linalg.inv(JJ_T + damping * np.eye(6))

        # Step 7: compute torque using Eq (7)
        M = pin.crba(model, data, q)
        h = pin.nonLinearEffects(model, data, q, v)
        tau = M @ J_sharp @ (X_ddot_d - Jdot_v) + h

        return tau

################################################################################
# Application
################################################################################
    
class Envionment:
    def __init__(self):        
        # state
        self.cur_state = State.JOINT_SPLINE
        
        # create simulation
        self.simulator = PybulletWrapper()
        
        ########################################################################
        # spawn the robot
        ########################################################################
        self.q_home = np.zeros(32)
        self.q_home[14:22] = np.array([0, +0.45, 0, -1, 0, 0, 0, 0 ])
        self.q_home[22:30] = np.array([0, -0.45, 0, -1, 0, 0, 0, 0 ])
        
        self.q_init = np.zeros(32)

        # TODO: spawn robot
        self.robot = Talos(self.simulator, q=self.q_init)

        ########################################################################
        # joint space spline: init -> home
        ########################################################################
        # TODO: create a joint spline 
        q0 = self.q_init
        q1 = self.q_home
        v0 = np.zeros(32)
        v1 = np.zeros(32)
        self.duration = 5.0  # spline time

        self.q_splines = [CubicHermiteSpline([0, self.duration], [q0[i], q1[i]], [v0[i], v1[i]]) for i in range(32)]       
        
        # TODO: create a joint controller
        self.controller = JointSpaceController(self.robot, Kp=np.diag([100.0] * 32), Kd=np.diag([10.0] * 32))

        ########################################################################
        # cart space: hand motion
        ########################################################################

        # TODO: create a cartesian controller
        self.cartesian_controller = CartesianSpaceController(
            robot=self.robot,
            joint_name="arm_right_7_joint",
            Kp=np.diag([400.0] * 6),
            Kd=np.diag([40.0] * 6)
            )
        self.switch_to_cartesian = False
        self.X_goal = None
        
        ########################################################################
        # logging
        ########################################################################
        
        # TODO: publish robot state every 0.01 s to ROS
        self.t_publish = 0.0
        self.publish_period = 0.01
        
    def update(self, t, dt):
        # TODO: update the robot and model
        self.robot.update()

        # update the controllers
        # TODO: Do inital jointspace, switch to cartesianspace control
        if self.cur_state == State.JOINT_SPLINE:
            t_clipped = min(t, self.duration)
            q_r = np.array([spline(t_clipped) for spline in self.q_splines])
            q_r_dot = np.array([spline.derivative(1)(t_clipped) for spline in self.q_splines])
            q_r_ddot = np.array([spline.derivative(2)(t_clipped) for spline in self.q_splines])
            tau = self.controller.update(q_r, q_r_dot, q_r_ddot)

            if t > self.duration and not self.switch_to_cartesian:
                print("==> switch to cartesian control")
                self.switch_to_cartesian = True
                self.cur_state = State.CART_SPLINE
                frame_id = self.robot.wrapper().model.getFrameId("arm_right_7_joint")
                self.X_goal = self.robot.wrapper().data.oMf[frame_id]

        elif self.cur_state == State.CART_SPLINE:
            X_r = self.X_goal
            X_dot_r = np.zeros(6)
            X_ddot_r = np.zeros(6)
            tau = self.cartesian_controller.update(X_r, X_dot_r, X_ddot_r) 
        
        # command the robot
        self.robot.setActuatedJointTorques(tau)

        # TODO: publish ros stuff
        self.t_publish += dt
        if self.t_publish >= self.publish_period:
            self.robot.publish()
            self.t_publish = 0.0
            
def main():
    rclpy.init()
    
    # create ROS2 node for internal communication
    node = rclpy.create_node('tutorial_3_robot_sim')
 
    # create the environment
    env = Envionment()

    # Instantiate the controller (even though env already creates one)
    controller = JointSpaceController(env.robot, 
                                      Kp = np.diag([100.0]*32), 
                                      Kd = np.diag([10.0]*32))

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

if __name__ == '__main__':
    main()