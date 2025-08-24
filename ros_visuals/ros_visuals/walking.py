"""
talos walking simulation
"""

import numpy as np
import pinocchio as pin
import pybullet as pb
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

# simulator
#>>>>TODO: import simulator
from simulator.pybullet_wrapper import PybulletWrapper

# robot configs
#>>>>TODO: Import talos walking config file
import talos_conf as conf

# modules
#>>>>TODO: Import all previously coded modules
from talos import Talos
from footstep_planner import FootStepPlanner, Side
from lip_mpc import LIPMPC, LIPInterpolator, generate_zmp_reference
from foot_trajectory import SwingFootTrajectory
        
################################################################################
# main
################################################################################  
    
def main(): 
    
    ############################################################################
    # setup
    ############################################################################
    
    #>>>>TODO: setup ros
    rclpy.init()
    node = rclpy.create_node('tutorial_7_walking_node')
    
    # setup the simulator
    #>>>>TODO: simulation
    simulator = PybulletWrapper()
    
    # setup the robot
    #>>>> TODO: robot
    robot = Talos(simulator=simulator, node=node, conf=conf)

    T_swing_w = robot.swingFootPose()       # set intial swing foot pose to left foot
    T_support_w = robot.supportFootPose()   # set intial support foot pose to right foot 
    robot.tsid.set_LF_pose_ref(T_swing_w)
    robot.tsid.set_RF_pose_ref(T_support_w)

    # setup the plan with 20 steps
    no_steps = 20
    planner = FootStepPlanner(conf)   #>>>>TODO: Create the planner
    plan = planner.planLine(T_swing_w, robot.swing_foot, no_steps)   #>>>>TODO: Create the plan

    mpc = LIPMPC(conf) 
    
    # generate reference
    #>>>>TODO: Generate the mpc reference
    ZMP_ref = mpc.generate_zmp_reference(plan)
    
    #>>>>TODO: plot the plan (make sure this workes first)
    planner.plot(simulator) 

    # Assume the com is over the first support foot
    com_init = T_support_w.translation[:2]
    x0 = np.array([com_init[0], 0.0, com_init[1], 0.0])   #>>>>TODO: Build the intial mpc state vector
    interpolator = LIPInterpolator(x0, conf)#>>>>TODO: Create the interpolator and set the inital state
    
    # set the com task reference to the inital support foot
    c, c_dot, c_ddot = interpolator.comState()

    # setup foot trajectory spline
    foot_traj = SwingFootTrajectory(T_support_w, T_swing_w, 1*conf.step_dur, height=0.05)
    
    ############################################################################
    # stand still
    ############################################################################
    
    pre_dur = 3.0   # Time to wait befor walking should start
    
    ############################################################################
    # logging
    ############################################################################

    # Compute number of iterations:
    N_pre = int(pre_dur / conf.dt)                  # number of sim steps before walking starts 
    N_sim = (no_steps+2) * conf.no_sim_per_step         # total number of sim steps during walking
    N_mpc = (no_steps+2) * conf.no_mpc_samples_per_step # total number of mpc steps during walking
    
    # Print debug information
    print(f"Walking simulation steps: {N_sim}")
    
    # Create vectors to log all the data of the simulation
    # - COM_POS, COM_VEL, COM_ACC (from the planned reference, pinocchio and pybullet)
    # - Angular momentum (from pinocchio)
    # - Left and right foot POS, VEL, ACC (from planned reference, pinocchio) 
    # - ZMP (from planned reference, from estimator )
    # - DCM (from estimtor)
    # - Normal forces in right and left foot (from pybullet ft sensors, from pinocchio)
    # COM state（pinocchio、pybullet）
    COM_REF_POS_VEL = np.nan * np.ones((N_sim, 4))
    COM_REF_ACC = np.nan * np.ones((N_sim, 3))
    COM_PIN_POS = np.nan * np.ones((N_sim, 3))
    COM_PIN_VEL = np.nan * np.ones((N_sim, 3))
    COM_PIN_ACC = np.nan * np.ones((N_sim, 3))
    COM_BULLET_POS = np.nan * np.ones((N_sim, 3))
    
    # Angular momentum（Pinocchio）
    ANG_MOMENTUM = np.nan * np.ones((N_sim, 3))
    
    # foot（Pinocchio）
    LF_POS_REF = np.nan * np.ones((N_sim, 3))
    LF_VEL_REF = np.nan * np.ones((N_sim, 3))
    LF_ACC_REF = np.nan * np.ones((N_sim, 3))
    RF_POS_REF = np.nan * np.ones((N_sim, 3))
    RF_VEL_REF = np.nan * np.ones((N_sim, 3))
    RF_ACC_REF = np.nan * np.ones((N_sim, 3))
    LF_POS_PIN = np.nan * np.ones((N_sim, 3))
    RF_POS_PIN = np.nan * np.ones((N_sim, 3))
    
    # ZMP
    ZMP_REF_VEC = np.nan * np.ones((N_sim, 2))
    ZMP_VEC = np.nan * np.ones((N_sim, 2))
    
    # DCM
    DCM_VEC = np.nan * np.ones((N_sim, 2))
    
    # force
    LF_FORCE_PB = np.nan * np.ones((N_sim,))
    RF_FORCE_PB = np.nan * np.ones((N_sim,))
    LF_FORCE_PIN = np.nan * np.ones((N_sim,))
    RF_FORCE_PIN = np.nan * np.ones((N_sim,))
    TIME = np.nan*np.empty(N_sim)
    
    ############################################################################
    # main loop
    ############################################################################
    
    # Initialize variables
    k = 0                                               # current MPC index                          
    plan_idx = 1                                        # current index of the step within foot step plan
    t_step_elapsed = 0.0                                # elapsed time within current step (use to evaluate spline)
    t_publish = 0.0                                     # last publish time (last time we published something)
    u_k = ZMP_ref[0]
    com_line_id = -1
    ref_line_id = -1
    
    for i in range(-N_pre, N_sim):
        t = simulator.simTime()    # simulator time
        dt = simulator.stepTime()  # simulator dt
        
        ########################################################################
        # update the mpc very no_sim_per_mpc steps
        ########################################################################

        if i == -N_pre+1000: # if we are at the first step, we need to set the CoM reference
            print("CoM Begins to move")
            rf_pos = robot.tsid.get_placement_RF().translation # get the right foot position
            c = rf_pos.copy() # set CoM to the right foot position
            c[2] = conf.h
            robot.tsid.setComRefState(c, np.zeros(3), np.zeros(3)) # set CoM reference state

        if i == 0: # if we are at the first step, we need to set the CoM state
            # Initialize the CoM state
            com = robot.tsid.comState().value()  # CoM
            com_dot = robot.tsid.comState().derivative() # CoM Vel
            x_k = np.array([com[0], com_dot[0], com[1], com_dot[1]]) # get real CoM state
            interpolator.x = x_k # NOTE: correct com state before start walking
            print("Start Walking!")
        
        if i >= 0 and i % conf.no_sim_per_mpc == 0:   # when to update mpc
            # Implement MPC update
            com = robot.tsid.comState().value()  # CoM
            com_dot = robot.tsid.comState().derivative() # CoM Vel
            x_k = interpolator.x.copy() # get simulated CoM state
            # run mpc fetch valid zmp
            u_k = mpc.lip_mpc_loop(k, ZMP_ref, x_k)
            
            k += 1

            # TODO: Use addGlobalDebugTrajectory to visualize trajetory
            ZMP_REF_VEC[k] = u_k
            COM_REF_POS_VEL[k] = x_k
            COM_PIN_POS[k] = com + np.array([0.0, 0.0, 0.85])
            X = ZMP_REF_VEC[:k, 0]
            Y = ZMP_REF_VEC[:k, 1]
            Z = np.ones_like(X) * 0.02
            simulator.line_ids = simulator.addGlobalDebugTrajectory(
                X, Y, Z,
                color=[0, 0, 1],
                line_ids=getattr(simulator, 'line_ids', []),
                lineWidth=3
            )
            X = COM_REF_POS_VEL[:k, 0]
            Y = COM_REF_POS_VEL[:k, 2]
            Z = np.ones_like(X) * (conf.h + 0.85)
            simulator.line1_ids = simulator.addGlobalDebugTrajectory(
                X, Y, Z,
                color=[1, 1, 0.5],
                line_ids=getattr(simulator, 'line1_ids', []),
                lineWidth=3
            )
            X = COM_PIN_POS[:k, 0]
            Y = COM_PIN_POS[:k, 1]
            Z = COM_PIN_POS[:k, 2]
            simulator.line3_ids = simulator.addGlobalDebugTrajectory(
                X, Y, Z,
                color=[1, 0.1, 0.5],
                line_ids=getattr(simulator, 'line3_ids', []),
                lineWidth=3
            )

        ########################################################################
        # update the foot spline 
        ########################################################################

        if i >= 0 and plan_idx < len(plan) - 1 and i % conf.no_sim_per_step == 0: # when to update spline
            # Start next step
            print("Step", plan_idx, "Swing", plan[plan_idx-1].side)
            robot.setSupportFoot(plan[plan_idx].side)
            robot.setSwingFoot(plan[plan_idx-1].side)
            T0 = plan[plan_idx-1].poseInWorld()
            T1 = plan[plan_idx+1].poseInWorld()
            foot_traj.reset(T0, T1)
            # switch support foot & swing foot
            t_step_elapsed = 0.0
            plan_idx += 1
            
            # TODO: visualize foot trajectory
            ts = np.linspace(0, conf.step_dur, 10)
            positions = np.array([foot_traj.evaluate(t)[0].translation if hasattr(foot_traj.evaluate(t)[0], "translation") else foot_traj.evaluate(t)[0] for t in ts])
            X = positions[:, 0]
            Y = positions[:, 1]
            Z = positions[:, 2]
            simulator.line2_ids = simulator.addGlobalDebugTrajectory(
                X, Y, Z,
                color=[0.2, 0.2, 0.2],
                line_ids=getattr(simulator, 'line2_ids', []),
                lineWidth=3
            )

        if t_step_elapsed > 0.5 and plan_idx == len(plan) - 1 and i % conf.no_sim_per_step == 0:
            print("Walking Complete!")
            plan_idx += 1
            robot.tsid.add_contact_LF()
            robot.tsid.add_contact_RF()
            if plan[-1].side == Side.LEFT:
                robot.tsid.set_LF_pose_ref(plan[-1].poseInWorld())
            else:
                robot.tsid.set_RF_pose_ref(plan[-1].poseInWorld()) 
            if plan[-2].side == Side.LEFT:
                robot.tsid.set_LF_pose_ref(plan[-2].poseInWorld())
            else:
                robot.tsid.set_RF_pose_ref(plan[-2].poseInWorld())
            last_com = (plan[-1].poseInWorld().translation + plan[-2].poseInWorld().translation)/2
            last_com[2] = conf.h
            robot.tsid.setComRefState(last_com, np.zeros(3), np.zeros(3))
            ref_line_id = simulator.addGlobalDebugLine(last_com, last_com, ref_line_id, color=[0,1,0])
            
        ########################################################################
        # in every iteration when walking
        ########################################################################
        
        if i >= 0 and plan_idx < len(plan):
            t_step_elapsed += dt
            
            """CoM Task"""
            com = robot.tsid.comState().value()  # CoM
            com_line_id = simulator.addGlobalDebugLine(com, np.array([com[0], com[1], 0.0]), com_line_id, color=[1,0,0])
            # update CoM ref
            x_i = interpolator.integrate(u_k)
            # set com ref
            c, c_dot, c_ddot = interpolator.comState()
            ref_line_id = simulator.addGlobalDebugLine(c, c + 0.5*c_ddot, ref_line_id, color=[0,1,0])
            robot.tsid.setComRefState(c, 1.0*c_dot, 0.85*c_ddot)
            
            """Swing Foot Task"""
            # NOTE: add double support phase
            if t_step_elapsed > 1*conf.step_dur:
                if robot.swing_foot == Side.LEFT:
                    robot.tsid.add_contact_LF()
                else:
                    robot.tsid.add_contact_RF()
                foot_ref = foot_traj.evaluate(1*conf.step_dur)
            else:
                foot_ref = foot_traj.evaluate(t_step_elapsed)
            # set swing foot pose ref
            if robot.swing_foot == Side.LEFT:
                pos = foot_ref[0].translation
                vel = np.zeros(3)  # 3D zero velocity
                acc = np.zeros(3)  # 3D zero acceleration
                robot.tsid.set_LF_pos_ref(pos, vel, acc)
            else:
                pos = foot_ref[0].translation
                vel = np.zeros(3)
                acc = np.zeros(3)
                robot.tsid.set_RF_pos_ref(pos, vel, acc)

        ########################################################################
        # update the simulation
        ########################################################################

        # update the simulator and the robot
        simulator.step()
        robot.update()

        # publish to ros
        if t - t_publish > 1./30.:
            t_publish = t
            
        # store for visualizations
        if i >= 0:
            TIME[i] = t
            
            #>>>>TODO: log information
            # com_pos = robot.tsid.comState().value()
            # com_vel = robot.tsid.comState().derivative()
            # try:
            #     com_acc = robot.tsid.comState().second_derivative()
            # except:
            #     com_acc = np.zeros(3)
            
            # COM_PIN_POS[i] = com_pos
            # COM_PIN_VEL[i] = com_vel  
            # COM_PIN_ACC[i] = com_acc
            
            # LF_POS_PIN[i] = robot.tsid.get_placement_LF().translation
            # RF_POS_PIN[i] = robot.tsid.get_placement_RF().translation
            
            # ZMP_VEC[i] = robot.zmp[:2]  # Only x,y components
            # DCM_VEC[i] = robot.dcm[:2]  # Only x,y components
            
    
    print("Simulation completed successfully!")
    
    ########################################################################
    # plot
    ########################################################################
    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create reference trajectories for plotting
    COM_POS_ref = np.zeros((N_sim, 3))
    COM_VEL_ref = np.zeros((N_sim, 3))
    ZMP_ref_plot = np.zeros((N_sim, 2))
    
    # Extract reference data from logged arrays
    for i in range(N_sim):
        if not np.isnan(COM_REF_POS_VEL[i, 0]):
            COM_POS_ref[i, 0] = COM_REF_POS_VEL[i, 0]  # x position
            COM_POS_ref[i, 1] = COM_REF_POS_VEL[i, 2]  # y position
            COM_POS_ref[i, 2] = conf.h + 0.85  # z position (constant)
            COM_VEL_ref[i, 0] = COM_REF_POS_VEL[i, 1]  # x velocity
            COM_VEL_ref[i, 1] = COM_REF_POS_VEL[i, 3]  # y velocity
        if not np.isnan(ZMP_REF_VEC[i, 0]):
            ZMP_ref_plot[i] = ZMP_REF_VEC[i]
    
    # plot everything
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # COM Position
    axes[0,0].plot(TIME, COM_POS_ref[:, 0], 'r-', label='X ref')
    axes[0,0].plot(TIME, COM_POS_ref[:, 1], 'g-', label='Y ref') 
    axes[0,0].plot(TIME, COM_POS_ref[:, 2], 'b-', label='Z ref')
    axes[0,0].plot(TIME, COM_PIN_POS[:, 0], 'r--', label='X actual')
    axes[0,0].plot(TIME, COM_PIN_POS[:, 1], 'g--', label='Y actual')
    axes[0,0].set_title('COM Position')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # COM Velocity
    axes[0,1].plot(TIME, COM_VEL_ref[:, 0], 'r-', label='X ref')
    axes[0,1].plot(TIME, COM_VEL_ref[:, 1], 'g-', label='Y ref')
    axes[0,1].plot(TIME, COM_PIN_VEL[:, 0], 'r--', label='X actual')
    axes[0,1].plot(TIME, COM_PIN_VEL[:, 1], 'g--', label='Y actual')
    axes[0,1].set_title('COM Velocity')
    axes[0,1].set_ylabel('Velocity (m/s)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # ZMP
    axes[1,0].plot(TIME, ZMP_ref_plot[:, 0], 'r-', label='X ref')
    axes[1,0].plot(TIME, ZMP_ref_plot[:, 1], 'g-', label='Y ref')
    axes[1,0].plot(TIME, ZMP_VEC[:, 0], 'r--', label='X actual')
    axes[1,0].plot(TIME, ZMP_VEC[:, 1], 'g--', label='Y actual')
    axes[1,0].set_title('ZMP')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Position (m)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # DCM
    axes[1,1].plot(TIME, DCM_VEC[:, 0], 'r-', label='X')
    axes[1,1].plot(TIME, DCM_VEC[:, 1], 'g-', label='Y')
    axes[1,1].set_title('DCM')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Position (m)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Forces
    axes[2,0].plot(TIME, LF_FORCE_PB, 'r-', label='Left Foot')
    axes[2,0].plot(TIME, RF_FORCE_PB, 'b-', label='Right Foot')
    axes[2,0].set_title('Normal Forces')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_ylabel('Force (N)')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # COM trajectory in XY plane
    axes[2,1].plot(COM_POS_ref[:, 0], COM_POS_ref[:, 1], 'b-', label='COM trajectory')
    axes[2,1].scatter(LF_POS_PIN[::100, 0], LF_POS_PIN[::100, 1], c='red', s=20, label='Left foot')
    axes[2,1].scatter(RF_POS_PIN[::100, 0], RF_POS_PIN[::100, 1], c='blue', s=20, label='Right foot')
    axes[2,1].set_title('Walking Pattern (Top View)')
    axes[2,1].set_xlabel('X (m)')
    axes[2,1].set_ylabel('Y (m)')
    axes[2,1].legend()
    axes[2,1].grid(True)
    axes[2,1].axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    # rclpy.init()
    main()