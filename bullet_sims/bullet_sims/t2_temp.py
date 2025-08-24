import pybullet as pb
import numpy as np
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot
import pinocchio as pin
import time

def main():
# For REEM-C robot
#urdf = "src/reemc_description/robots/reemc.urdf"
#path_meshes = "src/reemc_description/meshes/../.."
    
# For REEM-C robot
#urdf = "src/reemc_description/robots/reemc.urdf"
#path_meshes = "src/reemc_description/meshes/../.."

    # For Talos robot
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    '''
Talos
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20, 21  # left arm
22, 23, 24, 25, 26, 27, 28, 29  # right arm
30, 31                          # head

REEMC
0, 1, 2, 3, 4, 5, 			    # left leg
6, 7, 8, 9, 10, 11, 			# right leg
12, 13,                         # torso
14, 15, 16, 17, 18, 19, 20,     # left arm
21, 22, 23, 24, 25, 26, 27,     # right arm
28, 29                          # head
'''
    # Initial condition for the simulator an model
    z_init = 1.15
    q_actuated_home = np.zeros(32)
    q_actuated_home[:6] = np.array([0, 0, 0, 0, 0, 0])
    q_actuated_home[6:12] = np.array([0, 0, 0, 0, 0, 0])
    q_actuated_home[14:22] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ])
    q_actuated_home[22:30] = np.array([0, 0, 0, 0, 0, 0, 0, 0 ])
    
    # Initialization position including floating base
    q_home = np.hstack([np.array([0, 0, z_init, 0, 0, 0, 1]), q_actuated_home])

    # setup the task stack
    modelWrap = pin.RobotWrapper.BuildFromURDF(urdf, path_meshes, pin.JointModelFreeFlyer(), True, None)
    model = modelWrap.model
    data = model.createData()

    simulator = PybulletWrapper(sim_rate=1000)

    robot = Robot(simulator, urdf, model, [0, 0, z_init], [0,0,0,1], q=q_home, useFixedBase=False, verbose=True)
    simulator.step()
    robot.update()

    q = robot.q()     
    v = robot.v()       
    M = pin.crba(model, data, q)                    
    b = pin.nonLinearEffects(model, data, q, v)  

    print("Inertia Matrix M:\n", M)
    print("Nonlinear Effects Vector b:\n", b)
    
    simulator.addLinkDebugFrame(-1, -1)

    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8])

    tau = q_actuated_home * 0

    done = False

    start_time = time.time() 
    while not done:
        simulator.step()
        simulator.debug()
        robot.update()
        robot.setActuatedJointTorques(tau)

        if time.time() - start_time > 10.0:
            done = True

if __name__ == '__main__':
    main()