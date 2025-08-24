import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin
import time

def main():
    # For Talos robot
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    z_init = 1.15
    q_actuated_home = np.zeros(32)
    q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0]) 
    q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0]) 
    q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ]) 
    q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ]) 

    q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

    modelWrap = pin.RobotWrapper.BuildFromURDF(urdf, path_meshes, pin.JointModelFreeFlyer(), True, None) # 使用 pin.RobotWrapper 加载 URDF 构建模型
    model = modelWrap.model 
    data = model.createData() 

    # set up the PD gain and desired position
    n = model.nv - 6
    Kp_diag = np.ones(n)
    Kd_diag = np.ones(n)
    # 0-12 leg_left 1-6 & leg_right 7-12
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
    Kp_diag[14:22] = 2.0
    Kd_diag[14:22] = 0.2
    # Right Arm
    Kp_diag[22:30] = 2.0
    Kd_diag[22:30] = 0.2
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

    # set up the initial state
    q_ini = q_home.copy()  

    # set up q_home
    q_home = pin.neutral(model)  # 生成一个39维的中立姿态 q[0:7] 是 base 位姿（位置 + 四元数）
    q_home[2] = z_init            # 设置 base_z
    q_home[3:7] = [0, 0, 0, 1]    # 设置 base_orientation

    # 0-12 leg_left 1-6 & leg_right 7-12
    # 12-13 torso_1_joint, torso_2_joint
    # 14-21 arm_left_joint
    # 22-29 arm_right_joint
    # 30-31 head_1_joint, head_2_joint
    
    # Legs
    q_home[7+0:7+6] = np.array([0, 0, -0.44, 0.9, -0.45, 0])     # left_leg
    q_home[7+6:7+12] = np.array([0, 0, -0.44, 0.9, -0.45, 0])     # right_leg

    # Torso
    q_home[7+12:7+14] = np.array([0.0, 0.0])                       # torso

    # Arms
    q_home[7+14:7+21] = np.array([0, -0.24, 0, -1, 0, 0, 0])       # left_arm
    q_home[7+21:7+28] = np.array([0, -0.24, 0, -1, 0, 0, 0])       # right_arm
    
    # Head
    q_home[7+28:7+30] = np.array([0, 0])                           # head

    # set up the interpolation time
    T_total = 0.5
    t_start = time.time() 

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

        t_now = time.time()
        elapsed = t_now - t_start # the time has passed
        if elapsed < T_total: 
            alpha = min(elapsed / T_total, 1.0) 
            q_desired = pin.interpolate(model, q_ini, q_home, alpha) # pin.interpolate(...) 的功能是 计算两个姿态之间的插值结果，而 alpha 决定了插值的“进度”——即我们希望机器人现在走到哪个百分比位置
        else:
            q_desired = q_home.copy()
        
        q = robot.q()[7:]    
        v = robot.v()[6:]   
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())
        b_actuated = b[6:] 
        tau = b_actuated + Kp @ (q_desired[7:] - q) - Kd @ v
        tau = np.clip(tau, -200, 200)
        robot.setActuatedJointTorques(tau)

if __name__ == '__main__':
    main()