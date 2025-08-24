import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt 

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from ros_visuals.tsid_wrapper import TSIDWrapper
from ros_visuals.tsid_wrapper import create_sample
import ros_visuals.config as conf

# ROS
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped

# 画图
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适用于 Docker / 无 GUI
import matplotlib.pyplot as plt

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot): # Robot继承了Body；Talos继承了Robot
    # 构造函数 __init__ 初始化机器人模型、ROS节点、消息发布者和tf广播器
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False):
        # TODO call base class constructor
        # Fix: set proper base position to avoid ground collision
        super().__init__(simulator, urdf, model.model(), 
                        basePosition=np.array([0, 0, 1.15]), 
                        q=q, verbose=verbose, useFixedBase=useFixedBase)
        self._node = rclpy.create_node('t52_zmp_cmp_dcm_node')
        self._tsid_model = model  # Store TSID model separately
       
        # model 是 TSIDWrapper 提供的接口对象（robot.model()）
        # model.model() 才是标准的 pinocchio.Model
        # 而 pinocchio.centerOfMass()、framesForwardKinematics() 等函数需要你提供 (model, data)，你必须自己建一个 self._model 和 self._data
        self._model = model.model() # 从 TSIDWrapper 的 .robot 对象中拿到的 model 是一个 wrapper，而 model.model() 才是真正的 pinocchio.Model 对象
        self._data = self._model.createData()

        # Body 定义了 self._id；但 Robot 并没有把 self._id 存为 self._robot_id；applyForce() 依赖 self._robot_id
        self._robot_id = self.id()
        
        # TODO add publisher
        self._joint_state_pub = self._node.create_publisher(JointState, 'joint_states', 10)
        
        # TODO add tf broadcaster
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self._node)
        pass

    # 更新函数 update 用于更新机器人状态
    def update(self):
        # TODO update base class
        super().update()
        pass

    # publish(T_b_w) 负责将机器人关节状态通过 ROS 发布出去，并发布 base_link 相对于 world 的变换（tf）
    def publish(self, T_b_w):
        # TODO publish jointstate
        joint_msg = JointState()
        joint_msg.header.stamp = self._node.get_clock().now().to_msg()
        joint_msg.name = [name for name in self._model.names[1:]]  # Skip universe
        joint_msg.position = self.q()[7:].tolist()  # Skip floating base (7 DOF), call q as a method
        joint_msg.velocity = self.v()[6:].tolist() if hasattr(self, 'v') else [0.0] * len(joint_msg.position)
        joint_msg.effort = [0.0] * len(joint_msg.position)
        self._joint_state_pub.publish(joint_msg)
        
        # TODO broadcast transformation T_b_w
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self._node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"
        
        # T_b_w is (SE3_object, Motion_object) tuple
        se3_obj, motion_obj = T_b_w
        xyzquat = pin.SE3ToXYZQUAT(se3_obj)

        tf_msg.transform.translation.x = float(xyzquat[0])
        tf_msg.transform.translation.y = float(xyzquat[1])
        tf_msg.transform.translation.z = float(xyzquat[2])
        tf_msg.transform.rotation.x = float(xyzquat[3])
        tf_msg.transform.rotation.y = float(xyzquat[4])
        tf_msg.transform.rotation.z = float(xyzquat[5])
        tf_msg.transform.rotation.w = float(xyzquat[6])
        
        self._tf_broadcaster.sendTransform(tf_msg)

####################################
# Utility functions (ZMP, CMP, DCM) 
####################################
def compute_zmp_from_wrench(wrench: pin.Force, d: float = 0.1):
    fx, fy, fz = wrench.linear
    tx, ty, _ = wrench.angular
    if abs(fz) < 1e-5:
        return np.array([0.0, 0.0, 0.0])
    px = (-ty - fx * d) / fz
    py = ( tx - fy * d) / fz
    return np.array([px, py, 0.0])

def compute_cmp(com: np.ndarray, f: np.ndarray):
    fx, fy, fz = f
    if abs(fz) < 1e-5:
        return np.zeros(3)
    rx = com[0] - fx / fz * com[2]
    ry = com[1] - fy / fz * com[2]
    return np.array([rx, ry, 0.0])

def compute_dcm(com_pos: np.ndarray, com_vel: np.ndarray, g: float = 9.81):
    z = com_pos[2]
    if z < 1e-5:
        return np.zeros(3)
    omega = np.sqrt(g / z)
    xi = com_pos[:2] + com_vel[:2] / omega
    return np.array([xi[0], xi[1], 0.0])

# === Balance control strategies ===
# ##################################
# 定义Ankle Strategy
####################################
def ankle_strategy(robot, com_pos, com_vel, zmp_global, zmp_ref, com_ref, gains, tsid_wrapper):
    """
    踝关节策略：
    基于zmp & com的误差计算期望的踝关节力矩
    这是一种feedback Control策略，用于调整机器人在平衡状态下的质心速度
    核心思想：调整com的速度，使得ZMP跟踪参考ZMP，使得机器人平衡
    
    :param robot: 机器人对象
    :param com_pos: 当前 CoM 的位置
    :param com_vel: 当前 CoM 的速度
    :param zmp_global: 当前 ZMP 的位置
    :param zmp_ref: 参考 ZMP 的位置
    :param com_ref: 参考 CoM 的位置
    :param gains: 控制器增益 (Kx, Kp)
    """
    Kx, Kp = gains
    # 计算期望的 CoM 速度
    com_vel_desired = -Kx * (com_pos[:2] - com_ref[:2]) + Kp * (zmp_global[:2] - zmp_ref[:2])
    
    # 创建 TSID 的 CoM 参考样本
    com_ref_sample = create_sample(com_ref, vel=np.hstack((com_vel_desired, [0.0])))
    
    # 更新 TSID 的 CoM 任务
    tsid_wrapper.comTask.setReference(com_ref_sample)


####################################
# 定义Hip Strategy
####################################
def hip_strategy_control(cmp, cmp_ref, K_gamma=1.0):
    """
    髋关节策略控制器：根据 CMP 的误差计算期望的角动量。
    :param cmp: 当前 CMP 的位置
    :param cmp_ref: 参考 CMP 的位置
    :param K_gamma: 控制增益
    :return: 期望的角动量
    """
    cmp_error = cmp - cmp_ref # 计算 CMP 的误差
    Gamma_d = K_gamma * cmp_error # 根据误差计算期望的角动量
    return Gamma_d # 返回期望的角动量

###################
# 初始化时准备数据结构
###################
# 数据记录列表
log_time = []
log_zmp = []
log_cmp = []
log_dcm = []
log_com = []

################################################################################
# main
################################################################################
# main 函数是程序的入口点，负责初始化 ROS、创建机器人和模拟器实例，并运行主循环
def main(): 
    T_MAX = 14.0 # Max simulation time
    
    # node = rclpy.create_node('tutorial_4_standing_node')
    rclpy.init() # Initialize ROS
   
    # TODO init TSIDWrapper
    tsid_wrapper = TSIDWrapper(conf) # 初始化TSID控制器，传入配置文件
    
    # TODO init Simulator
    simulator = PybulletWrapper() # 初始化模拟器，使用PyBullet作为物理引擎
    
    # TODO init ROBOT
    robot = Talos(simulator, conf.urdf, tsid_wrapper.robot,  # 初始化机器人模型
                  q=conf.q_home, useFixedBase=False)
    
    ######################################
    # 初始化阶段添加传感器
    ######################################
    # 在 PyBullet 中启用第 6 个关节（在踝部上方 10cm）的力-力矩传感器
    pb.enableJointForceTorqueSensor(robot.id(), robot.jointNameIndexMap()['leg_right_6_joint'], True) # 启用右腿第六个关节的力矩传感器
    pb.enableJointForceTorqueSensor(robot.id(), robot.jointNameIndexMap()['leg_left_6_joint'], True) # 启用左腿第六个关节的力矩传感器

    ######################################
    # 状态机参数和跟踪变量
    ######################################
    f_push_mag = 40.0 # 推力大小
    t_push = 0.5 # 推力持续时间
    t_period = 4.0 # 推力周期

    push_directions = [
        np.array([1.0, 0.0, 0.0]),  # Push in right direction
        np.array([-1.0, 0.0, 0.0]), # Push in left direction
        np.array([0.0, -1.0, 0.0])  # Push in back direction
    ]

    last_push_time = -10.0 # 上次推力的时间
    push_active = False # 是否处于推力状态
    push_idx = -1 # 当前推力方向索引
    push_end_time = 0.0 # 推力结束时间
    line_id = -1  # 可视化箭头的id
    

    t_publish = 0.0 # 初始化发布间隔时间

    # 初始化虚拟状态
    q_tsid = np.copy(robot.q())  # 初始关节位置
    v_tsid = np.copy(robot.v())  # 初始关节速度

    n_update = 1  # 每次更新一步

    while rclpy.ok(): # Main loop
        ############################
        # Detmine control strategies
        ############################
        use_ankle_strategy = False  # 是否使用踝关节策略
        use_hip_strategy = False   # 是否使用髋关节策略

        global log_time, log_zmp, log_cmp, log_dcm, log_com
        
        # elaped time
        t = simulator.simTime()

        ##########################
        # 自动结束仿真
        ##########################
        if t > T_MAX:
            print(f"Simulation time has reached {T_MAX} s, stop and draw the pictures.")
            break

        # TODO: update the simulator and the robot
        simulator.step() # 更新模拟器状态
        robot.update() # 更新机器人状态

        # wrench 是 在踝部坐标系下 测量的，但是你要在世界坐标系下分析机器人，所以我们需要知道
        # 脚底 left_sole_link / right_sole_link 的位姿
        # 踝部 leg_left_6_joint / leg_right_6_joint 的位姿
        # 这些都是 SE(3) 变换，叫做 H_w_*，意思是：从世界坐标系 → 到该 frame 的变换
        
        ############################
        # 提取 ankle / sole 的世界位姿
        ############################

        # 创建一个空的数据结构，供 Pinocchio 存储 frame 位姿
        data = robot._model.createData() 
        # 利用当前 q 值（关节位置）做前向运动学，计算所有 frame 的位置
        pin.framesForwardKinematics(robot._model, data, robot.q())
        
        # 它们的类型是 pin.SE3，你可以通过
        # .translation 获取 3D 坐标；.rotation 获取旋转矩阵；.act(p) 将 p 从 local frame 转换到 world frame

        # 获取左脚底板位姿
        H_w_lsole = data.oMf[robot._model.getFrameId("left_sole_link")]
        # 获取右脚底板位姿
        H_w_rsole = data.oMf[robot._model.getFrameId("right_sole_link")]
        # 获取左踝（leg_left_6_joint）位姿
        H_w_lankle = data.oMf[robot._model.getFrameId("leg_left_6_joint")]
        # 获取右踝（leg_right_6_joint）位姿
        H_w_rankle = data.oMf[robot._model.getFrameId("leg_right_6_joint")]
        
        ###############################################
        # 主循环中读取 wrench（力 + 力矩）：在踝部坐标系下 测量
        ###############################################
        
        # 开启了传感器后, 获取传感器数据 wren = pb.getJointState(...)[2]
        # 把它转换为 pin.Force（用于后续计算 ZMP）
        # 分别处理左右踝部

        # 读取右踝的 wrench（力 + 力矩）
        # PyBullet 的 getJointState(...)[2] 返回的是：机器人“对地面施加”的力（也就是机器人脚踩地时向下压地的力）

        # 我们关心：地面对机器人施加的反作用力（用于计算 ZMP），方向相反，所以我们对每一项加了负号
        wren = pb.getJointState(robot.id(), robot.jointNameIndexMap()['leg_right_6_joint'])[2]
        # wren 顺序为：Fx, Fy, Fz, Tx, Ty, Tz
        wnp = np.array([
            -wren[0],  # Fx
            -wren[1],  # Fy
            -wren[2],  # Fz
            -wren[3],  # Tx
            -wren[4],  # Ty
            -wren[5],  # Tz
        ])
        wr_rankle = pin.Force(wnp) # 转为 Pinocchio 的 Force 对象

        # 读取左踝的 wrench（力 + 力矩）
        wren = pb.getJointState(robot.id(), robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        # wren 顺序为：Fx, Fy, Fz, Tx, Ty, Tz
        wnp = np.array([
            -wren[0],  # Fx
            -wren[1],  # Fy
            -wren[2],  # Fz
            -wren[3],  # Tx
            -wren[4],  # Ty
            -wren[5],  # Tz
        ])
        wl_lankle = pin.Force(wnp)

        #######################################
        # 主循环中调用compute_zmp_from_wrench
        #######################################
       
        # 单脚 ZMP：分别在 local frame 中计算
        zmp_l_local = compute_zmp_from_wrench(wl_lankle)
        zmp_r_local = compute_zmp_from_wrench(wr_rankle)

        # 然后从 ankle 坐标变换到世界坐标（使用 H_w_lankle.act()）
        zmp_l_world = H_w_lankle.act(zmp_l_local)
        zmp_r_world = H_w_rankle.act(zmp_r_local)

        # 全局 ZMP 合并（双足支持，来自式(4)-(6)）：
        fz_l = wl_lankle.linear[2]
        fz_r = wr_rankle.linear[2]
        total_fz = fz_l + fz_r
        
        if abs(total_fz) > 1e-5:
            zmp_global = (fz_r * zmp_r_world + fz_l * zmp_l_world) / total_fz
        else:
            zmp_global = np.zeros(3)

        #######################################
        # 主循环中调用compute_cmp
        #######################################

        # 当前 CoM 的位置（世界坐标系）
        com_pos = pin.centerOfMass(robot._model, robot._data, robot.q())  # 返回 np.ndarray(3,)

        # 地面对机器人总反作用力（合并左右脚）
        f_total = wr_rankle.linear + wl_lankle.linear  # 是 np.ndarray(3,)

        # CMP
        cmp_global = compute_cmp(com_pos, f_total)

        #######################################
        # 主循环中调用compute_dcm
        #######################################
        # 获取 CoM 速度（基于 current_q, current_v）
        com_vel = pin.centerOfMass(robot._model, robot._data, robot.q(), robot.v())

        # DCM 估计
        dcm_global = compute_dcm(com_pos, com_vel)

        # 记录每一帧的数据
        log_time.append(t)
        log_zmp.append(zmp_global[:2].copy())
        log_cmp.append(cmp_global[:2].copy())
        log_dcm.append(dcm_global[:2].copy())
        log_com.append(com_pos[:2].copy())

        ##########################################
        # 主循环中调用ankle_strategy 和 hip_strategy
        ##########################################
        # 调用踝关节策略
        if use_ankle_strategy:
            zmp_ref = np.array([0.0, 0.0, 0.0])  # 参考 ZMP
            com_ref = np.array([0.0, 0.0, 0.8])  # 参考 CoM，高度为 0.8 米
            ankle_gains = (4.0, 2.0)  # 调整增益 (Kx, Kp)
            
            # 调用踝关节策略
            ankle_strategy(robot, com_pos, com_vel, zmp_global, zmp_ref, com_ref, ankle_gains, tsid_wrapper)
     
       # 调用髋关节策略
        if use_hip_strategy:
            cmp_ref = np.array([0.0, 0.0, 0.0])  # 参考 CMP
            # 计算期望的角动量
            angular_momentum_desired = hip_strategy_control(cmp_global, cmp_ref, K_gamma=0.8)
            
            # 创建 TSID 的角动量参考样本
            am_ref = create_sample(angular_momentum_desired)
            
            # 更新 TSID 的角动量任务
            tsid_wrapper.amTask.setReference(am_ref)

        #######################################
        # 状态机逻辑（在 TSID 控制前执行）
        #######################################
        if t - last_push_time >= t_period and not push_active:
            # 如果当前时间超过上次推力时间加上周期，并且没有处于推力状态
            # 开始推力
            push_idx = (push_idx + 1) % len(push_directions) # 切换到下一个推力方向
            push_force = f_push_mag * push_directions[push_idx] # 计算推力向量
            robot.applyForce(push_force.tolist(), p_w=[0, 0, 1.0]) # 应用推力到机器人

            # 可视化箭头
            p1 = pb.getBasePositionAndOrientation(robot.id())[0] # 这是 PyBullet 的原始 API。你在 PybulletWrapper 和 Body 类中，实际调用底层 PyBullet 的 ID 来控制机器人
            p2 = p1 + 0.8 * push_force / np.linalg.norm(push_force) # 计算箭头终点位置
            line_id = simulator.addGlobalDebugLine(p1, p2, -1, color=[1, 0, 0]) # 添加全局调试线（箭头）到模拟器

            push_active = True # 设置推力状态为活动
            push_end_time = t + t_push # 推力结束时间为当前时间加上推力持续时间
            last_push_time = t # 更新上次推力时间

        elif push_active: # 如果当前处于推力状态
            if t < push_end_time: # 推力持续中
                # 计算当前推力向量
                # 根据需要调整推力大小或方向
                push_force = f_push_mag * push_directions[push_idx]
                robot.applyForce(push_force.tolist(), p_w=[0, 0, 1.0]) # 把推力施加在 base_link 高度 1.0 米处（接近上身腰部），与 TSID 控制的平衡区匹配
            else:
                # 推力结束
                robot.applyForce([0.0, 0.0, 0.0], p_w=[0, 0, 1.0]) # 清除推力
                simulator.removeDebugItem(line_id) # 移除可视化箭头
                push_active = False # 重置推力状态
        
        #######################################
        # TSID 控制器更新 + 设置力矩
        #######################################
        
        # setActuatedJointTorques 方法会将计算出的关节力矩应用到机器人上
        # 这将使机器人根据 TSID 控制器的输出进行运动
        current_q = np.asarray(robot.q(), dtype=np.float64)
        current_v = np.asarray(robot.v(), dtype=np.float64)
        tau_sol, dv_sol = tsid_wrapper.update(current_q, current_v, t)
        
        # integrate sol in virtual state
        # 这里的 q_tsid 和 v_tsid 是虚拟状态，代表 TSID 控制器的内部状态
        # 它们会在每次迭代中更新，以便 TSID 控制器可以在下一次迭代中使用

        # 从之前的力矩控制：直接控制关节的力矩，依赖于机器人硬件的精确力矩输出能力，容易受到外部扰动的影响。
        # 切换成位置控制：通过模拟动力学计算关节位置和速度，机器人硬件只需要跟踪目标位置，能够更好地应对外部扰动。
        # 任务将 TSID 计算的加速度积分为速度和位置：
        # 这种方法模拟了机器人动力学的自然行为，使得控制更加平滑和稳定
        q_tsid, v_tsid = tsid_wrapper.integrate_dv(q_tsid, v_tsid, dv_sol, n_update * simulator.stepTime())
        robot.setActuatedJointPositions(q_tsid, v_tsid) 

        # Publish to ROS 发布机器人状态到 ROS
        if t - t_publish > 1./30.: # 发布频率为 30Hz
            t_publish = t
            # 获取当前 base link 的变换 T_b_w
            # T_b_w 是一个 (SE3_object, Motion_object) 的元组
            # SE3_object 是一个 pinocchio 的 SE3 对象，表示位置和姿态
            # Motion_object 是一个 pinocchio 的 Motion 对象，表示速度
            
            T_b_w = tsid_wrapper.baseState() # 获取 base link 的状态
            
            # Publish the robot state 通过 ROS 发布机器人状态
            robot.publish(T_b_w) # 也可以在这里发布其他传感器数据或状态信息
            
            # 确保 ROS 节点处理消息
            # Process ROS callbacks
            rclpy.spin_once(robot._node, timeout_sec=0)

        print(f"Time: {t}")
        print(f"ZMP (global): {zmp_global}")
        print(f"CMP (global): {cmp_global}")
        print(f"DCM (global): {dcm_global}")
        print(f"CoM: {com_pos}")
        print(f"CoM height (z): {com_pos[2]}")
        print(f"Push force: {push_force}")
        print(f"Push direction index: {push_idx}")

    if DO_PLOT:
        log_time_np = np.array(log_time)
        log_zmp_np = np.array(log_zmp)
        log_cmp_np = np.array(log_cmp)
        log_dcm_np = np.array(log_dcm)
        log_com_np = np.array(log_com)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("X components over time")
        plt.plot(log_time_np, log_com_np[:, 0], label="CoM x")
        plt.plot(log_time_np, log_zmp_np[:, 0], label="ZMP x")
        plt.plot(log_time_np, log_cmp_np[:, 0], label="CMP x")
        plt.plot(log_time_np, log_dcm_np[:, 0], label="DCM x")
        plt.xlabel("Time [s]")
        plt.ylabel("X position [m]")
        plt.ylim([-0.1, 0.1])  # 设置 Y 轴范围
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Y components over time")
        plt.plot(log_time_np, log_com_np[:, 1], label="CoM y")
        plt.plot(log_time_np, log_zmp_np[:, 1], label="ZMP y")
        plt.plot(log_time_np, log_cmp_np[:, 1], label="CMP y")
        plt.plot(log_time_np, log_dcm_np[:, 1], label="DCM y")
        plt.xlabel("Time [s]")
        plt.ylabel("Y position [m]")
        plt.ylim([-0.1, 0.1])  # 设置 Y 轴范围
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        
        # 强制保存路径到源码文件夹 src/ros_visuals/ros_visuals/images/
        image_dir = "/workspaces/ros_ws/src/ros_visuals/ros_visuals/images"
        os.makedirs(image_dir, exist_ok=True) # 确保目录存在

        # 动态生成文件名并保存
        output_name = "no_control" if not use_ankle_strategy and not use_hip_strategy else \
                      "ankle_control" if use_ankle_strategy and not use_hip_strategy else \
                      "hip_control" if not use_ankle_strategy and use_hip_strategy else \
                      "both_control"
        dynamic_output_path = os.path.join(image_dir, f"t52_plot_{output_name}.png")
        plt.savefig(dynamic_output_path)
        print(f"The dynamic image is saved in {dynamic_output_path}")