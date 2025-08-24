import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 无GUI后端
import os

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from ros_visuals.tsid_wrapper import TSIDWrapper
import ros_visuals.config as conf

# ROS
import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped # for broadcasting transforms




################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot):
    # 构造函数 __init__ 初始化机器人模型、ROS节点、消息发布者和tf广播器
    def __init__(self, simulator, urdf, model, q=None, verbose=True, useFixedBase=False):
        # TODO call base class constructor
        # Fix: set proper base position to avoid ground collision
        super().__init__(simulator, urdf, model.model(), 
                        basePosition=np.array([0, 0, 1.15]), 
                        q=q, verbose=verbose, useFixedBase=useFixedBase)
        self._node = rclpy.create_node('tutorial_4_standing_node')
        self._tsid_model = model  # Store TSID model separately
        
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
        
        pass

################################################################################
# main
################################################################################
def main(): 
    MAX_SIM_TIME = 15.0

    rclpy.init()
    node = rclpy.create_node('tutorial_4_squating_node')
    
    # TODO init TSIDWrapper
    tsid_wrapper = TSIDWrapper(conf)
    # TODO init Simulator
    simulator = PybulletWrapper()
    # TODO init ROBOT
    robot = Talos(simulator, conf.urdf, tsid_wrapper.robot,  # 初始化机器人模型
                  q=conf.q_home, useFixedBase=False)
    
    t_publish = 0.0

    # Define a tiktok time 
    start_time = simulator.simTime()  # 记录开始时间，方便后续计时

    # Visualize the trajectory - 可视化采样点列表
    circle_center = np.array([0.4, -0.2, 1.1])  # 圆心位置
    radius = 0.2  # 圆的半径
    f_circuit = 0.1 # 圆的频率
    omega = 2 * np.pi * f_circuit  # 圆的角速度

    # 采样轨迹点用于可视化（100个点）
    N = 100
    theta_list = np.linspace(0, 2 * np.pi, N)
    X = [circle_center[0]] * N  # x不变
    Y = [circle_center[1] + radius * np.cos(theta) for theta in theta_list]
    Z = [circle_center[2] + radius * np.sin(theta) for theta in theta_list]

    # 添加可视化圆形轨迹线条
    simulator.addGlobalDebugTrajectory(X, Y, Z)


    # draw the picture & record the data 
    # 1） COM reference
    # 2） COM position TSID
    # 3） COM position Bullet
    # 4） COM velocity Bullet
    # 5） 每一帧的时间戳
    # 6） 每一帧的 COM 参考位置
    # 7） 每一帧的 COM TSID 位置
    # 8） 每一帧的 COM Bullet 位置
    # 9） 每一帧的 COM Bullet 速度

    # 包括 COM 的：
    # 位置（Position）
    # 速度（Velocity）
    # 加速度（Acceleration）
    # 记录时间戳
    log_t = []

    # 记录参考 COM 位置、速度和加速度
    # 参考 COM 位置
    log_com_ref = []
    log_com_ref_vel = []
    log_com_ref_acc = []

    # 记录 TSID 模型估计的 COM 位置、速度和加速度
    log_com_tsid = []
    log_com_tsid_vel = []
    log_com_tsid_acc = []
    
    # 记录 PyBullet 测量的 COM 位置和线速度
    log_com_bullet = []
    log_com_bullet_vel = []


    while rclpy.ok(): # Main loop

        # elaped time
        t = simulator.simTime()

        if t - start_time > MAX_SIM_TIME:
            break    # 仿真结束

        rh_task_activated = False  # 标志位：是否已激活右手任务

        # TODO: update the simulator and the robot
        simulator.step() # 更新模拟器状态
        robot.update() # 更新机器人状态

        # 每一帧都记录那些数据
        log_t.append(simulator.simTime())

        ref_sample = tsid_wrapper.comReference()        # TrajectorySample
        tsid_sample = tsid_wrapper.comState()           # TrajectorySample
        bullet_pos = robot.baseWorldPosition()          # np.array([x,y,z])
        bullet_vel = robot.baseWorldVelocity()[:3]      # np.array([vx,vy,vz])

        log_com_ref.append(ref_sample.value())
        log_com_ref_vel.append(ref_sample.derivative())
        log_com_ref_acc.append(ref_sample.second_derivative())

        log_com_tsid.append(tsid_wrapper.comState().value())
        log_com_tsid_vel.append(tsid_sample.derivative())
        log_com_tsid_acc.append(tsid_sample.second_derivative())


        log_com_bullet.append(bullet_pos)
        log_com_bullet_vel.append(bullet_vel)


        # TODO: update TSID controller
        current_q = np.asarray(robot.q(), dtype=np.float64)
        current_v = np.asarray(robot.v(), dtype=np.float64)
        tau_sol, dv_sol = tsid_wrapper.update(current_q, current_v, t)

        # command to the robot
        robot.setActuatedJointTorques(tau_sol)

        # update the COM, 参考到RF XY， 保持Z不变
        
        # get the current COM position
        p_com = tsid_wrapper.comState().value()
        #p_com = tsid_wrapper.comState().value  # 正确写法：先调用 .comState()，再取 .value 属性
        #print("DEBUG p_com =", p_com)
        #print("DEBUG type(p_com) =", type(p_com))
        
        # get the right foot position
        p_rf = tsid_wrapper.get_placement_RF().translation
        # construct a new COM position - X,Y取右脚，Z保持COM原高度
        new_com_ref = np.array([p_rf[0], p_rf[1], p_com[2]])
        # set the new COM reference
        tsid_wrapper.setComRefState(new_com_ref)
       
        # 2秒后移除左脚接触，抬起左脚
        if t - start_time > 2.0:
            # remove left foot contact
            tsid_wrapper.remove_contact_LF()
            # lift the left foot 获取左脚位置，Z轴抬高0.3米
            p_lf = tsid_wrapper.get_placement_LF()
            # p_lf[2] += 0.3 这里会报错，因为SE3不能这样用
            p_lf.translation[2] += 0.3 # 修改z轴高度

            # set the new left foot placement
            tsid_wrapper.set_LF_pose_ref(p_lf)

        # 4秒后开始用正弦波修改机器人的高度
        if t - start_time > 4.0:
            a = 0.05 # 振幅
            f = 0.5 # 频率
            elapsed_sin = t - start_time - 4.0  # 从第4秒开始计算正弦波时间
            
            # 以当前COM高度为基准振荡
            # 计算Z方向的参考位置、速度和加速度
            z_ref = p_com[2] + a * np.sin(2 * np.pi * f * elapsed_sin)
            v_z = a * 2 * np.pi * f * np.cos(2 * np.pi * f * elapsed_sin)
            a_z = - a * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * elapsed_sin)

            # 固定xy坐标为右脚位置 我们要让机器人整个身体的质心（COM）上下“蹲”，但始终保持在右脚正上方（XY对齐）。
            p_rf = tsid_wrapper.get_placement_RF().translation # .translation 提取右脚的 (x, y, z) 坐标
            p_com_ref = np.array([p_rf[0], p_rf[1], z_ref])  # 合成完整参考位置

            # 合成速度和加速度向量
            v_com_ref = np.array([0.0, 0.0, v_z])
            a_com_ref = np.array([0.0, 0.0, a_z])

            # 发给tsid控制器
            tsid_wrapper.setComRefState(p_com_ref, v_com_ref, a_com_ref)

        # 检查config信息：
        # 右手frame名：rh_frame_name = "contact_right_link"
        # 左手frame名：lh_frame_name = "contact_left_link"
        # 右脚frame名：rf_frame_name = "leg_right_sole_fix_joint
        # 左脚frame名：lf_frame_name = "leg_left_sole_fix_joint"
        # 机器人base frame名：base_frame_name = "base_link"
        # 手部任务权重：conf.w_hand = 1e-1
        # 手部控制器增益：conf.kp_hand = 10.0
        # TSIDWrapper 里已创建 rightHandTask 和 set_RH_pos_ref(...)

        # 在 8 秒后激活右手任务并让右手追踪你画的圆圈轨迹
        if t - start_time > 8.0:
            if not rh_task_activated:
                # 激活右手任务
                tsid_wrapper.formulation.addMotionTask(
                    tsid_wrapper.rightHandTask, # 右手任务
                    conf.w_hand, # 手部任务权重
                    1,               # priority level
                    0.0              # transition duration (no blending)
                )
                rh_task_activated = True

            # 每一帧更新右手的位置目标：在 YZ 平面画圆
            f = 0.1
            r = 0.2
            omega = 2 * np.pi * f
            elapsed_circle = t - start_time - 8.0 # 计算从第8秒开始的时间

            center = np.array([0.4, -0.2, 1.1]) # 圆心位置
            offset = np.array([
                0.0, 
                r * np.cos(omega * elapsed_circle),  # Y轴位置随时间变化
                r * np.sin(omega * elapsed_circle) # Z轴位置随时间变化
            ])
            pos_rh = center + offset # 右手参考位置

            # 右手参考速度
            vel_rh = np.array([
                0.0, 
                -r * omega * np.sin(omega * elapsed_circle),  # Y轴速度
                r * omega * np.cos(omega * elapsed_circle)   # Z轴速度
            ])

            # 右手参考加速度
            acc_rh = np.array([
                0.0, 
                -r * omega**2 * np.cos(omega * elapsed_circle),  # Y轴加速度
                -r * omega**2 * np.sin(omega * elapsed_circle)   # Z轴加速度
            ])
            # 将参考位置、速度和加速度设置到 TSID 控制器
            # 这个方法会：用 update_sample(...) 更新内部的 TrajectorySample；
            # 所以你必须提前构造好 TrajectorySample，而这个工作你已经在 wrapper 中通过 update_sample(...) 做好了，千万不要在 main() 中重复 setReference。
            tsid_wrapper.set_RH_pos_ref(pos_rh, vel_rh, acc_rh) # 不可以直接调用 set_RH_pos_ref(pos_rh)，因为它只设置位置，不更新速度和加速度。

        # publish to ros 发布机器人状态和ROS回调，保持不变
        if t - t_publish > 1./30.:
            t_publish = t
            # get current BASE Pose
            T_b_w = tsid_wrapper.baseState()
            robot.publish(T_b_w)
            rclpy.spin_once(robot._node, timeout_sec=0) # 处理ROS回调

    # 仿真结束后画图并保存
    # 转为 NumPy 数组
    t = np.array(log_t)
    com_ref = np.vstack(log_com_ref)
    vel_ref = np.vstack(log_com_ref_vel)
    acc_ref = np.vstack(log_com_ref_acc)

    com_tsid = np.vstack(log_com_tsid)
    vel_tsid = np.vstack(log_com_tsid_vel)
    acc_tsid = np.vstack(log_com_tsid_acc)

    com_bullet = np.vstack(log_com_bullet)
    vel_bullet = np.vstack(log_com_bullet_vel)

    # 设置输出路径
    output_dir = "/workspaces/ros_ws/src/ros_visuals/ros_visuals/images"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "T4_com_comparison_plot.png")

    # 绘图
    fig, axs = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    labels = ['x', 'y', 'z']

    for i in range(3):
        axs[0][i].plot(t, com_ref[:, i], 'r--', label='Ref')
        axs[0][i].plot(t, com_tsid[:, i], 'b-', label='TSID')
        axs[0][i].plot(t, com_bullet[:, i], 'g:', label='Bullet')
        axs[0][i].set_ylabel(f'pos {labels[i]} [m]')
        axs[0][i].legend()

        axs[1][i].plot(t, vel_ref[:, i], 'r--')
        axs[1][i].plot(t, vel_tsid[:, i], 'b-')
        axs[1][i].plot(t, vel_bullet[:, i], 'g:')
        axs[1][i].set_ylabel(f'vel {labels[i]} [m/s]')

        axs[2][i].plot(t, acc_ref[:, i], 'r--')
        axs[2][i].plot(t, acc_tsid[:, i], 'b-')
        axs[2][i].set_ylabel(f'acc {labels[i]} [m/s²]')

    axs[2][1].set_xlabel("Time [s]")
    plt.suptitle("T4 Exercise 4: COM Position / Velocity / Acceleration Comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"image has been saved in: {save_path}")
    print("log_com_tsid:", log_com_tsid)
    print("type(log_com_tsid[0]):", type(log_com_tsid[0]))
    print("tsid_wrapper.comState().value():", tsid_wrapper.comState().value())
    print("type(tsid_wrapper.comState().value()):", type(tsid_wrapper.comState().value()))
    print("simulator.simTime():", simulator.simTime())
    print("type(simulator.simTime()):", type(simulator.simTime()))

if __name__ == '__main__':
    main()
