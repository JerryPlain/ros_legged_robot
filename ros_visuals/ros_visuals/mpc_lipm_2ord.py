"""Task2: Linear inverted pendulum MPC

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12 but this time as a model predictive controller (MPC).

In this case we will solve the trajectory planning multiple times over 
a shorter horizon of just 2 steps (receding horizon).
Time between two MPC updates is called T_MPC.

In between MPC updates we simulate the Linear inverted pendulum at a smaller
step time T_SIM, with the lates MPC control ouput u.

Our state & control is the same as before
x = [cx, vx, cy, vy]
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Solve

import matplotlib.animation as animation

################################################################################
# settings
################################################################################

NO_STEPS                = 8         # total number of foot steps
STEP_TIME               = 0.8       # time needed for every step

# Robot Parameters:
# --------------
h                       = 0.80      # fixed CoM height (assuming walking on a flat terrain)
g                       = 9.81      # norm of the gravity vector
foot_length             = 0.10      # foot size in the x-direction
foot_width              = 0.06      # foot size in the y-direciton


# MPC Parameters:
# --------------
T_MPC                   = 0.1                                               # sampling time interval of the MPC
NO_MPC_SAMPLES_PER_STEP = int(round(STEP_TIME/T_MPC))                       # number of mpc updates per step

NO_STEPS_PER_HORIZON  = 2                                                   # how many steps in the horizon
T_HORIZON = NO_STEPS_PER_HORIZON*STEP_TIME                                  # duration of future horizon
NO_MPC_SAMPLES_HORIZON = int(round(NO_STEPS_PER_HORIZON*STEP_TIME/T_MPC))   # number of mpc updates per horizon

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                  # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                  # CoM velocity error squared cost weight (= smoothing cost)

# Simulation Parameters:
# --------------
T_SIM                   = 0.005                         # 200 Hz simulation time

NO_SIM_SAMPLES_PER_MPC = int(round(T_MPC/T_SIM))        # NO SIM samples between MPC updates
NO_MPC_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_MPC))   # Total number of MPC samples
NO_SIM_SAMPLES = int(round(NO_STEPS*STEP_TIME/T_SIM))   # Total number of Simulator samples

################################################################################
# Helper fnc
################################################################################

# 生成x方向的足迹，给定起始足迹位置 foot_step_0 和步长 step_size_x
# 以及步数 no_steps，返回一个包含所有足迹位置的列表
def generate_foot_steps(foot_step_0, step_size_x, no_steps):
    """Write a function that generates footstep of step size = step_size_x in the 
    x direction starting from foot_step_0 located at (x0, y0).
    
    Args:
        foot_step_0 (_type_): first footstep position (x0, y0)
        step_size_x (_type_): step size in x direction
        no_steps (_type_): number of steps to take

    Returns:
        list: 步态位置列表 [(x1, y1), (x2, y2), ...]
    """

    #>>>>TODO: generate the foot step plan with no_steps
    #>>>>Hint: Check the pdf Fig.3 for inspiration
    foot_steps = []
    x, y = foot_step_0
    for i in range(no_steps):
        foot_steps.append((x, y))
        # Alternate between left and right foot steps
        x += step_size_x
        # Alternate y position for left and right foot
        y = -y if i % 2 == 0 else abs(y) # i % 2 == 0 means left foot, so we use -y to keep y negative；abs(y) to keep y positive
    return foot_steps


def plot_foot_steps(foot_steps, XY_foot_print, ax):
    """
    绘制步态矩形。
    
    Args:
        foot_steps (list): 步态位置列表 [(x1, y1), (x2, y2), ...]
        XY_foot_print (tuple): 足迹矩形的尺寸 (length, width)
        ax (matplotlib.axes._axes.Axes): 用于绘制的 Matplotlib 轴对象
    """
    for i, step in enumerate(foot_steps):  # 遍历每个步态位置，返回索引 i 和位置 step
        # 矩形的左下角坐标
        x0 = step[0] - XY_foot_print[0] / 2 # 将步态位置的 x 坐标减去矩形长度的一半，得到矩形左下角的 x 坐标
        y0 = step[1] - XY_foot_print[1] / 2 # 将步态位置的 y 坐标减去矩形宽度的一半，得到矩形左下角的 y 坐标
        # 设置颜色：红色表示左脚，绿色表示右脚
        color = 'red' if i % 2 == 0 else 'green'
        # 绘制矩形
        # 创建一个矩形对象，左下角坐标为 (x0, y0)，宽度为 XY_foot_print[0]，高度为 XY_foot_print[1]，颜色为 color，透明度为 0.5
        # XY_foot_print[0] 是矩形的长度，XY_foot_print[1] 是矩形的宽度
        # 注意：矩形的坐标系是左下角为原点，x 轴向右，y 轴向上
        rect = plt.Rectangle((x0, y0), XY_foot_print[0], XY_foot_print[1], color=color, alpha=0.5)
        ax.add_patch(rect) # 添加矩形到轴对象中
    ax.set_xlabel("Position X (m)") # 设置 x 轴标签
    ax.set_ylabel("Position Y (m)") # 设置 y 轴标签
    ax.set_title("Footsteps") # 设置图表标题
    ax.axis("equal")  # 设置坐标轴比例相等
    ax.grid()  # 添加网格线


def generate_zmp_reference(foot_steps, no_samples_per_step):
    # 生成zmp的参考轨迹函数，确保在每个步态的持续时间内，ZMP参考应该位于该步态的中心位置
    # zmp需要在每一个步态的持续时间内保持不变
    # 返回一个大小为 (TOTAL_NO_SAMPLES, 2) 的向量
    """generate a function that computes a referecne trajecotry for the ZMP
    (We need this for the tracking cost in the cost function of eq. 12)
    Remember: Our goal is to keep the ZMP at the footstep center within each step.
    So for the # of samples a step is active the zmp_ref should be at that step.
    
    Returns a vector of size (TOTAL_NO_SAMPLES, 2)

    Args:
        foot_steps (list): 步态位置列表 [(x1, y1), (x2, y2), ...]
        no_samples_per_step (int): 每步的时间步数
    """
    #>>>>TODO: Generate the ZMP reference based on given foot_steps
    zmp_ref = []
    for step in foot_steps:
        zmp_ref.extend([step] * no_samples_per_step)  # 每个步态位置重复 no_samples_per_step 次
    zmp_ref = np.array(zmp_ref)  # 转换为 NumPy 数
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################
def continious_LIP_dynamics():
    omega2 = g / h
    A = np.array([
        [0, 1, 0, 0],
        [omega2, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, omega2, 0]
    ])
    B = np.array([
        [0, 0],
        [-omega2, 0],
        [0, 0],
        [0, -omega2]
    ])
    return A, B

from scipy.linalg import expm, solve_continuous_lyapunov

def discrete_LIP_dynamics(dt):
    A, B = continious_LIP_dynamics()
    # 构造 augmented 矩阵
    M = np.block([
        [A, B],
        [np.zeros((2, 6))]
    ])  # 6x6 matrix

    M_exp = expm(M * dt)
    Ad = M_exp[:4, :4]
    Bd = M_exp[:4, 4:]

    return Ad, Bd

################################################################################
# Simulation
################################################################################

class Simulator:
    """Simulates the Linear inverted pendulum continous dynamics
    Uses simple euler integration to simulate LIP at sample time dt
    """
    def __init__(self, x_inital, dt):
        self.dt = dt
        self.x = x_inital
        
        self.A, self.B = continious_LIP_dynamics()
        self.D = np.array([[0, 0], [1, 0], [0, 0], [0, 1]]) # 这个是扰动矩阵 D
        # D is used to add disturbance to the system, e.g., external pushes
        
    def simulate(self, u, d=np.zeros(2)):
        """updates the LIP state x using based on command u
        
        Optionally: Takes a disturbance acceleration d to simulate effect
        of external pushes on the LIP.
        """

        #>>>>TODO: Compute x_dot and use euler integration to approximate
        # the state at t+dt
        #>>>>TODO: The disturbance is added in x_dot as self.D@d
        x_dot = self.A @ self.x + self.B @ u + self.D @ d 
        self.x = self.x + x_dot * self.dt
        # return the new state
        # Note: self.x is a 4D vector [px, vx, py, vy]
        #       where px, py are the positions and vx, vy are the velocities
        #       self.x[0] = px, self.x[1] = vx,
        #       self.x[2] = py, self.x[3] = vy
        #       self.x[0:2] = [px, vx], self.x[2:4] = [py, vy]
        return self.x    

################################################################################
# MPC
################################################################################

class MPC:
    """MPC for the Linear inverted pendulum
    """
    def __init__(self, dt, T_horizon):
        self.dt = dt                                        # mpc dt
        self.T_horizon = T_horizon                          # time of horizon
        # 计算在整个预测范围中将有多少个采样点（即控制节点）
        self.no_samples = int(round(T_horizon/self.dt))     # mpc samples in horizon (nodes)
        # 使用函数 discrete_LIP_dynamics(dt) 获得了线性倒立摆的离散时间动力学矩阵 Ad,BdAd​,Bd
        self.Ad, self.Bd = discrete_LIP_dynamics(dt)
        
        self.X_k = None                                     # state over current horizon
        self.U_k = None                                     # control over current horizon
        self.ZMP_ref_k = None                               # ZMP reference over current horizon
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        # 输入参数:
        # x_k: 当前状态 [cx, vx, cy, vy]
        # ZMP_ref_k: 当前时间步的ZMP参考轨迹，形状=(no_samples, 2)
        # terminal_idx: 终端约束的索引
        # 输出参数:
        # 返回当前时间步的控制命令 U_k[0]，即 U_k[0] 是当前时间步的控制命令     
        """ build the MathematicalProgram that solves the mpc problem and 
        returns the first command of U_k

        Args:
            x_k (_type_): the current state of the lip when starting the mpc
            ZMP_ref_k (_type_): the reference over the current horizon, shape=(no_samples, 2)
            terminal_idx (_type_): index of the terminal constraint within horizon (or bigger than horizon if no constraint)
            
        """
        
        # variables
        nx = 4 #>>>>TODO: State dimension = ? # state = [cx, vx, cy, vy]
        nu = 2 #>>>>TODO: control dimension = ? # state = [cx, vx, cy, vy]
        prog = MathematicalProgram() # 通过 MathematicalProgram() 实例化了一个优化器，表示 MPC 优化问题的数学形式
        
        # 预测的 CoM 状态（包含位置和速度）
        state = prog.NewContinuousVariables(self.no_samples, nx, 'state')
        # 预测的 CoM 状态（包含位置和速度）
        control = prog.NewContinuousVariables(self.no_samples, nu, 'control')
        
        # 1. intial constraint
        #>>>>TODO: Add inital state constraint, Hint: x_k
        # 确保优化的第一个状态与当前观测状态一致（MPC的基本逻辑：从当前实际状态出发优化未来行为）
        # Note: state[0, i] is the initial state at time step 0, i.e., state at t=0
        for i in range(nx):
            prog.AddConstraint(state[0, i] == x_k[i])  # x_k is a 4D vector [cx, vx, cy, vy]

        # 2. at each time step: respect the LIP descretized dynamics
        #>>>>TODO: Enforce the dynamics at every time step
        # 用k和i来遍历每个时间步和状态维度：k 是时间步，i 是状态的维度
        # 用离散模型 xk+1=Adxk+Bdu 对每个时间步都施加了动力学约束，确保系统状态更新过程符合物理规律。
        for k in range(self.no_samples - 1):
            for i in range(nx):
                prog.AddConstraint(
                    state[k+1, i] == self.Ad[i, :] @ state[k, :] + self.Bd[i, 0] * control[k, 0] + self.Bd[i, 1] * control[k, 1]
        )

        # 3. at each time step: keep the ZMP within the foot sole (use the footprint and planned step position)
        #>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
        #Hint: first compute upper and lower bound based on zmp_ref then add constraints.
        #Hint: Add constraints at every time step
        # 添加zmp的上界和下界约束，确保控制（ZMP）在每个足迹内
        # ZMP_ref_k[k] is the reference ZMP at time step k
        # footprint is the size of the foot in x and y direction
        # footprint = [foot_length, foot_width]
        for k in range(self.no_samples): # 遍历每个时间步
            zmp_ub = ZMP_ref_k[k] + footprint / 2
            zmp_lb = ZMP_ref_k[k] - footprint / 2
            prog.AddBoundingBoxConstraint(zmp_lb, zmp_ub, control[k, :])

        # 4. if terminal_idx < self.no_samples than we have the terminal state within
        # the current horizon. In this case create the terminal state (foot step pos + zero vel)
        # and apply the state constraint to all states >= terminal_idx within the horizon
        #>>>>TODO: Add the terminal constraint if requires
        # Hint: If you are unsure, you can start testing without this first!
        # 当预测区间内覆盖了最后一个 ZMP 参考点时，你添加了“期望机器人最终静止在最后一步足印上”的终端约束：位置 = 脚印中心，速度 = 0
        if terminal_idx < self.no_samples:
            terminal_state = np.array([ZMP_ref_k[terminal_idx, 0], 0, ZMP_ref_k[terminal_idx, 1], 0])
            for i in range(nx):
                prog.AddConstraint(state[terminal_idx, i] == terminal_state[i]) # terminal_state is a 4D vector [cx, vx, cy, vy]

        # setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
        #>>>>TODO: add the cost at each timestep, hint: prog.AddCost
        # 添加目标函数（代价函数）
        # 目标函数是每个时间步的 ZMP 误差（跟踪成本）和 CoM 速度（平滑成本）的加权和
        # 其中 alpha 和 gamma 是权重系数，控制 ZMP 误差和 CoM 速度在目标函数中的相对重要性
        for k in range(self.no_samples):
            prog.AddCost(alpha * ((control[k, 0] - ZMP_ref_k[k, 0])**2 + (control[k, 1] - ZMP_ref_k[k, 1])**2)) # ZMP error cost, we want to minimize the distance between the control (ZMP) and the reference ZMP
            prog.AddCost(gamma * (state[k, 1]**2 + state[k, 3]**2)) # CoM velocity cost, we want to minimize the velocity of the CoM

        # solve 求解器会返回一个完整的未来轨迹
        result = Solve(prog)
        if not result.is_success:
            print("failure")
        
        # 保存整个 X_k 和 U_k，同时返回第一个控制输入 U_k[0]，这是当前时刻要使用的命令
        self.X_k = result.GetSolution(state)
        self.U_k = result.GetSolution(control)
        if np.isnan(self.X_k).any():
            print("failure")
        
        self.ZMP_ref_k = ZMP_ref_k
        return self.U_k[0] # 返回当前时间步的控制命令 U_k[0]，即 U_k[0] 是当前时间步的控制命令
    
################################################################################
# run the simulation
################################################################################
# 这一段代码是模拟线性倒立摆（LIP）的 MPC 控制器的主要部分
# 通过设置初始状态、足迹、ZMP 参考轨迹等，来模拟 LIP 的运动
# 它是 整个 MPC 控制器的测试验证主程序（main loop）。没有这部分，你只写了控制器，但不知道它到底效果好不好、机器人走得稳不稳、有没追踪目标

# 线性倒立摆模型的状态 x=[px,p˙x,py,p˙y]∈R4
# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# 创建步伐计划 + ZMP 参考轨迹
# footprint
footprint = np.array([foot_length, foot_width]) # size of the foot in x and y direction

# generate the footsteps
step_size = 0.2

#>>>>TODO: 1. generate the foot step plan using generate_foot_steps
# 设置第一个足迹位置 foot_step_0
foot_step_0 = np.array([0.0, -0.09])  # 注意：这就是 footstep0
foot_steps = generate_foot_steps(foot_step_0, step_size, NO_STEPS) # 生成足迹列表

# reapeat the last two foot steps (so the mpc horizon never exceeds the plan!)
foot_steps = np.vstack([
    foot_steps, foot_steps[-1], foot_steps[-1]])

# zmp reference trajecotry
# 生成一条“ZMP 目标轨迹”，MPC 要去跟踪它
#>>>>TODO: 2. generate the complete ZMP reference using generate_zmp_reference
ZMP_ref = generate_zmp_reference(foot_steps, NO_MPC_SAMPLES_PER_STEP)

# generate mpc
mpc = MPC(T_MPC, T_HORIZON) # 实例化 MPC 控制器，传入采样时间 T_MPC 和预测区间 T_HORIZON

# generate the pendulum simulator
state_0 = np.concatenate([x_0, y_0])
sim = Simulator(state_0, T_SIM) # 实例化模拟器，传入初始状态 state_0 和采样时间 T_SIM

# setup some vectors for plotting stuff
TIME_VEC = np.nan*np.ones(NO_SIM_SAMPLES)
STATE_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 4])
ZMP_REF_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])
ZMP_VEC = np.nan*np.ones([NO_SIM_SAMPLES, 2])

# time to add some disturbance
t_push = 3.2

# 执行主循环（最核心部分）
# execution loop
k = 0   # the number of mpc update
for i in range(NO_SIM_SAMPLES):
    
    # simulation time
    t = i*T_SIM
        
    if i % NO_SIM_SAMPLES_PER_MPC == 0: #  每 T_MPC 秒更新一次 MPC 控制器：
        # time to update the mpc
        # 从当前状态出发
        # 取未来一段的参考轨迹
        # 生成一组最优控制 uk​，只执行第一个
        # 这体现了 MPC 的核心思想：滚动优化 + 只执行第一步
        
        # current state
        #>>>>TODO: get current state from the simulator
        x_k = sim.x.copy()  # x_k is the current state of the LIP
    
        #>>>>TODO: extract the current horizon from the complete reference trajecotry ZMP_ref
        ZMP_ref_k = ZMP_ref[k : k + mpc.no_samples] # 这个是当前时间步的ZMP参考轨迹，形状=(no_samples, 2)
        
        # 如果 k + mpc.no_samples > len(ZMP_ref)，则会报错，所以需要确保 k + mpc.no_samples <= len(ZMP_ref)
        # 若超出长度，用最后一帧填充
        if ZMP_ref_k.shape[0] < mpc.no_samples:
            last_ref = ZMP_ref_k[-1]
            pad_len = mpc.no_samples - ZMP_ref_k.shape[0]
            ZMP_ref_k = np.vstack([ZMP_ref_k, np.tile(last_ref, (pad_len, 1))])
        
        # check if we have terminal constraint
        idx_terminal_k = NO_MPC_SAMPLES - k

        #>>>>TODO: Update the mpc, get new command
        u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, idx_terminal_k) 
        
        k += 1
    
    # simulate a push for 0.05 sec with 1.0 m/s^2 acceleration 
    x_ddot_ext = np.array([0, 0])
    
    #>>>>TODO: when you got everything working try adding a small disturbance
    # if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
    #    x_ddot_ext = np.array([0, 1.0])
    if i > int(t_push/T_SIM) and i < int((t_push + 0.05)/T_SIM):
        x_ddot_ext = np.array([0.0, 1.0])
    else:
        x_ddot_ext = np.zeros(2)

    #>>>>TODO: Update the simulation using the current command
    x_k = sim.simulate(u_k, x_ddot_ext) # 执行当前控制命令，仿真更新状态
    
    # save some stuff
    TIME_VEC[i] = t
    STATE_VEC[i] = x_k
    ZMP_VEC[i] = u_k
    ZMP_REF_VEC[i] = mpc.ZMP_ref_k[0]
    
ZMP_LB_VEC = ZMP_REF_VEC - footprint[None,:]
ZMP_UB_VEC = ZMP_REF_VEC + footprint[None,:]

#>>>>TODO: Use the recodings in STATE_VEC and ZMP_VEC to compute the 
# LIP acceleration
#>>>>Hint: Use the continious dynamic matrices
# Continuous dynamics
A_cont, B_cont = continious_LIP_dynamics()
STATE_DOT_VEC = STATE_VEC @ A_cont.T + ZMP_VEC @ B_cont.T

################################################################################
# plot something

#>>>>TODO: plot everything in x-axis
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(TIME_VEC, STATE_VEC[:, 0], label="CoM x")
plt.plot(TIME_VEC, ZMP_REF_VEC[:, 0], '--', label="ZMP ref x")
plt.plot(TIME_VEC, ZMP_VEC[:, 0], label="ZMP x")
plt.ylabel("Position [m]")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(TIME_VEC, STATE_VEC[:, 1], label="CoM vx")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(TIME_VEC, STATE_DOT_VEC[:, 1], label="CoM ax")
plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")
plt.legend()
plt.grid()
plt.suptitle("X-Direction")
plt.tight_layout()

#>>>>TODO: plot everything in y-axis
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(TIME_VEC, STATE_VEC[:, 2], label="CoM y")
plt.plot(TIME_VEC, ZMP_REF_VEC[:, 1], '--', label="ZMP ref y")
plt.plot(TIME_VEC, ZMP_VEC[:, 1], label="ZMP y")
plt.ylabel("Position [m]")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(TIME_VEC, STATE_VEC[:, 3], label="CoM vy")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(TIME_VEC, STATE_DOT_VEC[:, 3], label="CoM ay")
plt.ylabel("Acceleration [m/s²]")
plt.xlabel("Time [s]")
plt.legend()
plt.grid()
plt.suptitle("Y-Direction")
plt.tight_layout()

#>>>>TODO: plot everything in xy-plane
fig, ax = plt.subplots()
ax.plot(STATE_VEC[:, 0], STATE_VEC[:, 2], label='CoM trajectory')
ax.plot(ZMP_VEC[:, 0], ZMP_VEC[:, 1], label='ZMP trajectory')
ax.plot(ZMP_REF_VEC[:, 0], ZMP_REF_VEC[:, 1], '--', label='ZMP ref')

# 添加足迹框
plot_foot_steps(foot_steps, footprint, ax)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Plan View: CoM and ZMP')
ax.grid()
ax.legend()
plt.axis('equal')
plt.show()
