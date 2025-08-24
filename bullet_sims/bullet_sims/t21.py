import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin

def main():
    # For Talos robot
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    z_init = 1.15
    q_actuated_home = np.zeros(32)
    q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0]) # left legs
    q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0]) # right legs
    q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ]) # left arms
    q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ]) # right arms

    q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

    modelWrap = pin.RobotWrapper.BuildFromURDF(urdf, path_meshes, pin.JointModelFreeFlyer(), True, None) # 使用 pin.RobotWrapper 加载 URDF 构建模型
    model = modelWrap.model
    data = model.createData()

    # set up the PD gain and desired position
    n = model.nv - 6 # remove 6 Dof of Floating base, what left is 32 actuated joints
    Kp_diag = np.ones(n)
    Kd_diag = np.ones(n)

    # 0-11 leg_left 1-6 & leg_right 7-12
    # 12-13 torso_1_joint, torso_2_joint
    # 14-21 arm_left_joint
    # 22-29 arm_right_joint
    # 30-31 head_1_joint, head_2_joint

    # Legs
    Kp_diag[0:12] = 800.0
    Kd_diag[0:12] = 60.0
    # Torso
    Kp_diag[12:14] = 150.0
    Kd_diag[12:14] = 20.0
    # Left Arm
    Kp_diag[14:22] = 5.0
    Kd_diag[14:22] = 0.5
    # Right Arm
    Kp_diag[22:30] = 5.0
    Kd_diag[22:30] = 0.5
    # Head
    Kp_diag[30:32] = 2
    Kd_diag[30:32] = 0.2
    # Convert to diagonal matrix
    Kp = np.diag(Kp_diag)
    Kd = np.diag(Kd_diag)
    
    # set up the simulator
    simulator = PybulletWrapper(sim_rate=1000)
    # set up the robot (before the q_desired)
    robot = Robot(simulator, urdf, model, [0, 0, z_init], [0,0,0,1], q=q_home, useFixedBase=False, verbose=True)
    simulator.addLinkDebugFrame(-1, -1)
    # step & update to ensure that the robot.q is valid
    simulator.step()
    robot.update()
    # set up the desired state
    q_desired = robot.q()[7:].copy() # the initial state
    # set up the pybullet camera
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8])

    done = False
    while not done:
        simulator.step()
        simulator.debug()
        robot.update()
        
        q = robot.q()[7:]     # current joint location
        v = robot.v()[6:]     # current joint velocity
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())
        b_actuated = b[6:] 
        tau = b_actuated + Kp @ (q_desired - q) - Kd @ v
        tau = np.clip(tau, -200, 200)
        robot.setActuatedJointTorques(tau)

if __name__ == '__main__':
    main()