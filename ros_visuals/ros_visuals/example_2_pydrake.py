"""
Example 2

The goal of this example is to use trajectory optimization on the standard
pendulum swingup problem (= getting the pendulum into an upward position)

This will show you how to use one of pydrakes solvers to find the trajectory x[]
through a system's statespace and the coresponding control signals u[]
over a horizon of N steps (also called nodes)

The problem (like always) is composed of a cost funciton C(x[],u[]) and a
series of constraints that have to be fullfiled (eigher at each step 
or at the start/end = intial/terminal constraint)

min_{x[], u[]} sum_k C_k(x_k,u_k) 
    s.t. ...

Here we will use the pendulum model as an example
State:      x = [q, q_dot] \in R^2   (angle and angle velocity)
Control:    u = tau \in R            (torque of a motor)

Please ckeck the pdf and TODOs in this file.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')

from pydrake.all import (MathematicalProgram, Solve, SnoptSolver) # for optimization
# SnoptSolver is a solver that can solve nonlinear programming problems
import pydrake.symbolic as sym # for symbolic math

################################################################################
# settings
################################################################################

# Pendulum Parameters:
# --------------
# m, g, l, b = 质量、重力、摆长、摩擦系数
m = 1.0         # mass of the pendulum
g = 9.81        # gravity
l = 0.8         # length of the pendulum
b = 0.1         # viscous friction of the pendulm

# Solver Parameters:
# --------------
N       = 350   # number of steps / our problem horizon 控制步数，也叫轨迹优化的 horizon（N 个阶段）

# The minimum and maximum Timeteps limits h 每一步时间间隔范围（限制最小最大步长）
# 总共要走 N 步
# 每步允许的时间范围在 h_min ~ h_max 之间；
h_min = .002    # 500 Hz
h_max = .05     # 100 Hz

################################################################################
# helper function for visualization and dynamics
################################################################################

class Visualizer(): # 这是一个类，用来在模拟过程中画出摆的实时位置动画
    """Visualize the pendulum
    """
    def __init__(self, ax, length): 
        """init visualizer

        Args:
            ax (axis): axis of a figure
            length (float): length of the pendulum in meters
        """
        self.ax = ax # ax 是 matplotlib 的坐标轴
        self.ax.set_aspect("equal") # set_aspect("equal")：x、y 比例一样
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.link = np.vstack((0.025 * np.array([1, -1, -1, 1, 1]),
                        length * np.array([0, 0, -1, -1, 0])))

        self.link_fill = ax.fill(
                self.link[0, :], self.link[1, :], zorder=1, edgecolor="k", facecolor=[.6, .6, .6])

    def draw(self, time, q): # 动画更新函数
        # Draw the pendulum at angle q at time t
        R = np.array([[np.cos(q), np.sin(q)], [-np.sin(q), np.cos(q)]]) # 构建一个二维旋转矩阵用来把摆在“竖直方向”的 link，旋转到当前角度 qq
        p = np.dot(R, self.link)
        self.link_fill[0].get_path().vertices[:, 0] = p[0, :] # 把刚才画好的矩形轮廓更新成新的位置（旋转后的）
        self.link_fill[0].get_path().vertices[:, 1] = p[1, :]
        self.ax.set_title("t = {:.1f}".format(time))


def pendulum_continous_dynamics(x, u):
    # 用于描述连续时间系统动力学的函数
    """continous dynamics x_dot = f(x, u) for a simple one dimensional 
    pendulum. (See eq. 13)

    Args:
        x (_type_): two dimensional state x=[q, q_dot]
        u (_type_): one dimensional control input u=torque

    Returns:
        _type_: the two dimensional time derivative x_dot=[q_dot, q_ddot]
    """
    
    # use object d to get math functions such as (d.sin, d.log, etc.)
    # 为了兼容符号优化器（symbolic）和数值计算（numpy）；
    # if x.dtype == object, use symbolic math, else use numpy
    # 这里的 x.dtype 是 numpy 的数据类型（dtype），如果是 object，
    # 则使用符号计算，否则使用 numpy 数值计算
    # 这样可以兼容符号优化器和数值计算
    d = sym if x.dtype == object else np # 所以 d.sin() 会自动切换用 sym.sin() 或 np.sin()

    #>>>>TODO: add continous state space equation and return x_dot
    # x_dot = [q_dot, q_ddot] 创建一个 2维的零向量
    x_dot = np.zeros(2, dtype=object if x.dtype == object else float)
    x_dot[0] = x[1] # q_dot = dq/dt = x[1]
    # q_ddot = d²q/dt² = -m * g * l * sin(q) - b * dq + u / (m * l²)
    # 这里的 m * g * l * sin(q) 是重力对摆的影响，
    # b * dq 是摩擦力对摆的影响，u 是施加的力矩
    # 注意：这里的 x[0] 是角度 q，x[1] 是角速度 q_dot
    # 这里的 m * g * l * sin(q) 是重力对摆的影响，
    # b * x[1] 是摩擦力对摆的影响，u 是施加的力矩
    # 注意：这里的 x[0] 是角度 q，x[1] 是角速度 q_dot
    # 这里的 m * l² 是摆的惯性矩（moment of inertia）
    x_dot[1] = (-m * g * l * d.sin(x[0]) - b * x[1] + u) / (m * l ** 2)
    return x_dot

def pendulum_discretized_dynamics(x, u, x_next, dt):
    # 这个函数实现离散化，并返回残差（residual），用于写成优化约束
    # 用欧拉法近似一阶微分系统
    # 构造优化约束：保证系统在每一时刻的状态转移与动力学模型一致
    # 为什么要用欧拉法？因为它简单且易于实现，适合初学者理解离散化过程。
    # 欧拉法的基本思想是：在当前状态 xk​ 处，计算出导数 f(xk​,uk​)，
    # 然后用这个导数来预测下一个状态 xk+1​
    # 也就是：xk+1​≈xk​+hk​⋅f(xk​,uk​)
    # 这里的 hk 是时间步长
    # 这个函数的目的是将连续时间系统的动力学方程离散化为离散时间系统的形式，
    # 以便在优化过程中使用。我们希望通过优化器找到一系列状态 x[k] 和控制 u[k]
    # 所以我们希望优化器满足下面的约束：
    # xk+hkf(xk,uk)−xk+1=0  ⇒  residual=0


    """descritization of the continous dynamics.
    First, compute the derivative at the current state x(t) using the know
    dynamics of the pendulum. Then, use euler integration to compute the next
    state at t + dt. 
    Finally, return the residual between the euler integration 
    and x_next. The solver needs to make this zero to repect the dynamics.

    Args:
        x (_type_): two dimensional state x=[q, q_dot] at time t
        x_next (_type_): two dimensional state x_next=[q, q_dot] at time t + dt
        delta_t (_type_): time discretization step

    Returns:
        _type_: the residual between euler integration and x_next
    """

    #>>>>TODO: compute x_dot integrated it using x and dt. Return the
    # residual to x_next
    x_dot = pendulum_continous_dynamics(x, u) # get the time derivative at x
    # use euler integration to compute the next state
    x_integrated = x + dt * x_dot # x_next = x + dt * x_dot
    # compute the residual between the integrated state and the next state
    residuals = x_integrated - x_next # residual = x_next - x_integrated
    return residuals # 优化器将 residuals == 0 当成一个约束，强制使
    # residuals 为零，
    # 也就是强制使 x_next = x + dt * x_dot 成立

################################################################################
# problem definition
################################################################################

# the important dimension in this problem
nx = 2  # dimension of our state x=[q, q_dot]
nu = 1  # dimension of our control u=tau

# Define our inital state, the pendulum is in its lowest energy state,
# (angle q=0, velocity q_dot=0)
x_intial = np.array([0.0, 0.0])

# Define our goal (terminal) state, the pendulum is in its upward position
# and should have zero acceleration
x_final = np.array([np.pi, 0.0])

# Define an instance of MathematicalProgram 创建优化器对象 & 决策变量
# 这是 PyDrake 的标准入口，用来：
# 定义决策变量；
# 添加代价函数；
# 添加各种约束。
# 可以把它想成一个“空白优化问题”。
prog = MathematicalProgram() 

################################################################################
# variables

# At any of our N timesteps we want to find the state x[k] and the control u[k]

# 添加决策变量（变量 = 系统轨迹）
# 状态：N+1 个点，每点是 x=[q, dq]
state = prog.NewContinuousVariables(N+1, nx, 'state')

# 添加控制输入变量（控制 = 力矩）
# 这里的控制输入是一个 torque（力矩），
# 也就是在每个时间步 k=0,1,...,N-1
# 我们要找到一个 torque u[k]，使得系统在每个时间步 k 的状态 x[k] 能够满足动力学约束
control = prog.NewContinuousVariables(N, 'control')

# 添加时间变量（时间 = 每个时间步的时间长）
# 这里的时间变量 h[k] 是每个时间步 k 的时间长度，
# 也就是在每个时间步 k=0,1,...,N-1
# 我们要找到一个时间 h[k]，使得系统在每个时间步 k 的状态 x[k] 能够满足动力学约束
h = prog.NewContinuousVariables(N, name='h') 

################################################################################
# constraints

# 1. we want our pendulum to start with the inital state x_init
# For this we can add an equality constraint to the first time step k=0
for i in range(nx):
    prog.AddConstraint(state[0,i] == x_intial[i])

# 2. we want our pendulum to end with the final state k=N
# For this we can add an equality to the last time step k=N
# 因为我们有 N 步控制，但 N+1 个状态；
# 第 N 步之后不会再用控制，所以终点通常设在 state[N-1]
for i in range(nx):
    prog.AddConstraint(state[N-1,i] == x_final[i])

# 3. add any timestep we want our solution to respect the dynamics of the pendulum
# That means the next state x_k+1 should be the integral of the prev. state x_k
# 动力学一致性约束
# 在每个时间步 k=0,1,...,N-1
# 我们希望优化器找到一个控制 u[k]，使得系统在每个时间步 k 的状态 x[k] 能够满足动力学约束
# 也就是
# x[k+1] = x[k] + h[k] * f(x[k], u[k])
# 这里的 f(x[k], u[k]) 是 pendulum_continous_dynamics(x[k], u[k])
# 也就是 pendulum_continous_dynamics(x[k], control[k])
for k in range(N-1):
    residuals = pendulum_discretized_dynamics(state[k], control[k], state[k+1], h[k])
    for i in range(nx):
        prog.AddConstraint(residuals[i] == 0)

prog.AddBoundingBoxConstraint([h_min]*N, [h_max]*N, h) # 时间步长限制（Box 约束）

# 4. add a constrain on the control torque
#>>>>TODO: After, you simulated the unconstraint case. 
#>>>>TODO: Add some limits on the control torque between some min and max value
# 给每个控制输入 uk=control[k]添加幅值限制：
# Set control bounds
u_min = -8.0   # 最小扭矩
u_max = +8.0   # 最大扭矩

# 给 control[k] 加上 box constraint
prog.AddBoundingBoxConstraint([u_min]*N, [u_max]*N, control)



################################################################################
# cost function

# in this example there are three costs:
# 1) minimize the control effort: u*R*u
# 2) get closer to the goal: (x - x_goal)^T*Q*(x - x_goal)
# 3) ####TODO: What is the meaning of the term S*sum(h) ?

Q = np.array([[500, 0],[0, 500]])
R = 10
S = 100 

for k in range(N):
    prog.AddCost(control[k]*R*control[k]) # 控制 effort cost（减少用力）
    prog.AddCost((state[k] - x_final).dot(Q.dot(state[k] - x_final))) # 状态跟踪 cost（接近目标）
prog.AddCost(S*sum(h)) # 时间 cost（鼓励快点完成）

################################################################################
# solve
# 设置初始猜测（给 h）
# 优化器是“从一个初始猜测开始”逐步迭代的；
# 这里我们假设每步时间 hk≈hmax
# 相当于先给了一个“最慢”的轨迹作为起点
h_guess = h_max
prog.SetInitialGuess(h, [h_guess]*N)

# finally lets start the solver, if we want we could provide an inital guess
result = Solve(prog) # Solve() 会根据问题的线性/非线性结构自动调用适合的优化器（例如 SNOPT）

if not result.is_success:
    print("failure")
print("solved")

# extract the solution
# 提取最优解
state_opt = result.GetSolution(state)
control_opt = result.GetSolution(control)
h_opt = result.GetSolution(h)

# seperate into variables
t_opt = np.cumsum(h_opt) # 将时间步转成累计时间点
q_opt = state_opt[:-1,0]
q_dot_opt = state_opt[:-1,1]
torque_opt = control_opt

################################################################################
# plot some stuff

fig, ax = plt.subplots(1,1)
vis = Visualizer(ax, length=l)
for k in range(N):
    if h_opt[k] > h_min:
        vis.draw(t_opt[k], q_opt[k])
        plt.pause(h_opt[k])

# x-axis pos, vel
fig, ax = plt.subplots(3,1, figsize=(12, 10))
ax[0].plot(t_opt, q_opt, linewidth=2, label="Position")
ax[0].grid();ax[0].legend();ax[0].set_ylabel("Postion [rad]")
ax[1].plot(t_opt, q_dot_opt, linewidth=2, label="Velocity")
ax[1].grid();ax[1].legend();ax[1].set_ylabel("Velocity [rad/s]")
ax[2].plot(t_opt, torque_opt, linewidth=2, label="Torque")
ax[2].grid();ax[2].legend();ax[2].set_ylabel("Torque [Nm]");ax[2].set_xlabel("Time [s]")

plt.show()