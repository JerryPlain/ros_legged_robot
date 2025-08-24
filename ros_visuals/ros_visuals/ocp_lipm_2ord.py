"""Task2: Linear inverted pendulum Trajectory planning

The goal of this file is to formulate the optimal control problem (OCP)
in equation 12. 

In this case we will solve the trajectory planning over the entire footstep plan
(= horizon) in one go.

Our state will be the position and velocity of the pendulum in the 2d plane.
x = [cx, vx, cy, vy]
And your control the ZMP position in the 2d plane
u = [px, py]

You will need to fill in the TODO to solve the task.
"""

import numpy as np

from pydrake.all import MathematicalProgram, Solve

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

################################################################################
# settings
################################################################################

# Robot Parameters:
# --------------

h           = 0.80   # fixed CoM height (assuming walking on a flat terrain)
g           = 9.81   # norm of the gravity vector
foot_length = 0.10   # foot size in the x-direction
foot_width  = 0.06   # foot size in the y-direciton

# OCP Parameters:
# --------------
T                     = 0.1                                # fixed sampling time interval of computing the ocp in [s]
STEP_TIME             = 0.8                                # fixed time needed for every foot step [s]

NO_SAMPLES_PER_STEP   = int(round(STEP_TIME/T))            # number of ocp samples per step

NO_STEPS              = 10                                 # total number of foot steps in the plan
TOTAL_NO_SAMPLES      = NO_SAMPLES_PER_STEP*NO_STEPS       # total number of ocp samples over the complete plan (= Horizon)

# Cost Parameters:
# ---------------
alpha       = 10**(-1)                                      # ZMP error squared cost weight (= tracking cost)
gamma       = 10**(-3)                                      # CoM velocity error squared cost weight (= smoothing cost)

################################################################################
# helper function for visualization and dynamics
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
    zmp_ref = zmp_ref[:TOTAL_NO_SAMPLES]  # 确保长度为 TOTAL_NO_SAMPLES
    return zmp_ref

################################################################################
# Dynamics of the simplified walking model
################################################################################

def continious_LIP_dynamics(g, h):
    """
    返回线性倒立摆模型的连续动力学矩阵 A 和 B。
    
    Args:
        g (float): 重力加速度
        h (float): 质心高度
    
    Returns:
        np.array: 矩阵 A, B
    """
    omega2 = g / h  # 倒立摆的自然频率的平方
    A = np.array([[0, 1], [omega2, 0]])
    B = np.array([[0], [-omega2]])
    return A, B


def discrete_LIP_dynamics(delta_t, g, h):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics

    Args:
        delta_t (_type_): discretization steps
        g (_type_): gravity
        h (_type_): height

    Returns:
        _type_: _description_
    """
    #>>>>TODO: Generate Ad, Bd for the discretized linear inverted pendulum
    # 计算离散化矩阵
    from scipy.linalg import expm  # 导入矩阵指数函数

    # zero-order hold discretization
    A, B = continious_LIP_dynamics(g, h)
    n = A.shape[0] # 获取状态维度

    # 构造扩展矩阵 M = [[A, B], [0, 0]]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, -1:] = B
    # 计算 e^{M dt}
    M_d = expm(M * delta_t)
    A_d = M_d[:n, :n]
    B_d = M_d[:n, -1]
    return A_d, B_d

################################################################################
# setup the plan references and system matrices
################################################################################

# inital state in x0 = [px0, vx0]
x_0 = np.array([0.0, 0.0])
# inital state in y0 = [py0, vy0]
y_0 = np.array([-0.09, 0.0])

# footprint
footprint = np.array([foot_length, foot_width])

# generate the footsteps
step_size = 0.2
#>>>>TODO: 1. generate the foot step plan using generate_foot_steps
foot_steps = generate_foot_steps((0.0, -0.09), step_size, NO_STEPS)

# zmp reference trajecotry
#>>>>TODO: 2. generate the ZMP reference using generate_zmp_reference
zmp_ref = generate_zmp_reference(foot_steps, NO_SAMPLES_PER_STEP)

#>>>>Note: At this point you can already start plotting things to see if they
# really make sense!

# discrete LIP dynamics
#>>>>TODO: get the static dynamic matrix Ad, Bd
A_d, B_d = discrete_LIP_dynamics(delta_t=T, g=g, h=h)
# continous LIP dynamics
#>>>>TODO: get the static dynamic matrix A, B
A, B = continious_LIP_dynamics(g=g, h=h)

################################################################################
# problem definition
################################################################################

# Define an instance of MathematicalProgram 
prog = MathematicalProgram() 

################################################################################
# variables
nx = 2 #>>>>TODO: State dimension = ? 状态维度 [质心位置, 质心速度]
nu = 2 #>>>>TODO: control dimension = ? 控制维度 [ZMP位置]

state = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nx, 'state')
control = prog.NewContinuousVariables(TOTAL_NO_SAMPLES, nu, 'control')

# intial state
#>>>>TODO: inital state if based on first footstep (+ zero velo)
# 质心位置为第一步的中心，速度为零
state_initial = np.array([foot_steps[0][0], 0.0])  # 初始质心位置和速度

# terminal state
#>>>>TODO: terminal state if based on last footstep (+ zero velo)
# 质心位置为最后一步的中心，速度为零
state_terminal = np.array([foot_steps[-1][0], 0.0])  # 终止质心位置和速度

################################################################################
# constraints

# 1. intial constraint
#>>>>TODO: Add inital state constrain, Hint: prog.AddConstraint
# 确保优化问题的初始状态与 state_initial 一致。
for i in range(state_initial.shape[0]):
    prog.AddConstraint(state[0, i] == state_initial[i])

# 2. terminal constraint
#>>>>TODO: Add terminal state constrain, Hint: prog.AddConstraint
for i in range(state_terminal.shape[0]):
    prog.AddConstraint(state[-1, i] == state_terminal[i])

# 3. at each step: respect the LIP descretized dynamics
#>>>>TODO: Enforce the dynamics at every time step
for k in range(TOTAL_NO_SAMPLES - 1):
    next_state = A_d @ state[k, :] + B_d * control[k, 0]
    for i in range(next_state.shape[0]):
        prog.AddConstraint(next_state[i] == state[k + 1, i])

# 4. at each step: keep the ZMP within the foot sole (use the footprint and planned step position)
#>>>>TODO: Add ZMP upper and lower bound to keep the control (ZMP) within each footprints
#Hint: first compute upper and lower bound based on zmp_ref then add constraints.
#Hint: Add constraints at every time step
for k in range(TOTAL_NO_SAMPLES):
    zmp_upper_bound = zmp_ref[k] + footprint / 2
    zmp_lower_bound = zmp_ref[k] - footprint / 2
    prog.AddBoundingBoxConstraint(zmp_lower_bound, zmp_upper_bound, control[k, :])

################################################################################
# stepwise cost, note that the cost function is scalar!

# setup our cost: minimize zmp error (tracking), minimize CoM velocity (smoothing)
#>>>>TODO: add the cost at each timestep, hint: prog.AddCost
for k in range(TOTAL_NO_SAMPLES):
    # ZMP误差代价
    prog.AddCost(alpha * np.sum((control[k, :] - zmp_ref[k])**2))
    # 质心速度误差代价
    prog.AddCost(gamma * np.sum(state[k, 1]**2))

################################################################################
# solve

result = Solve(prog)
if not result.is_success:
    print("failure")
print("solved")

# extract the solution
#>>>>TODO: extract your variables from the result object
t = T*np.arange(0, TOTAL_NO_SAMPLES)
# 提取优化结果
state_opt = result.GetSolution(state)  # 状态变量 [质心位置, 质心速度]
control_opt = result.GetSolution(control)  # 控制变量 [ZMP位置]

# compute the acceleration
#>>>>TODO: compute the acceleration of the COM
# 计算质心加速度
acceleration_opt = np.gradient(state_opt[:, 1], t)  # 对速度进行时间梯度计算

################################################################################
# plot something
#>>>>TODO: plot everything in x-axis
#>>>>TODO: plot everything in y-axis
#>>>>TODO: plot everything in xy-plane

# 绘制 x 方向的结果
plt.figure()
plt.plot(t, state_opt[:, 0], label="CoM Position")
plt.plot(t, state_opt[:, 1], label="CoM Velocity")
plt.plot(t, acceleration_opt, label="CoM Acceleration")
plt.plot(t, control_opt[:, 0], label="ZMP Position")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("X-axis Values")
plt.title("X-axis Results")
plt.grid()

# 绘制 y 方向的结果
plt.figure()
plt.plot(t, control_opt[:, 1], label="ZMP Position (Y)")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Y-axis Values")
plt.title("Y-axis Results")
plt.grid()

# 绘制 xy 平面的轨迹
plt.figure()
plt.plot(state_opt[:, 0], control_opt[:, 1], label="CoM Trajectory")
plt.scatter(zmp_ref[:, 0], zmp_ref[:, 1], label="ZMP Reference", color='red')
plt.legend()
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("XY-plane Trajectory")
plt.grid()
plt.show()