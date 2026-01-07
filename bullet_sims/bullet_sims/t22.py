import pybullet as pb
import numpy as np
import pinocchio as pin
import time

from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot


def main():
    """
    Tutorial 2 – Exercise 3
    目标不再是固定的 q_desired，而是一个随时间变化的轨迹（spline/interpolation）。
    为什么要插值、插值在 Pinocchio 里意味着什么、以及控制器如何跟踪这个轨迹？
    ----------------------
    Home posture controller for a floating-base humanoid robot (Talos).

    The controller performs:
    1) Time-based interpolation (spline) between an initial configuration
       and a desired home posture using Pinocchio interpolation.
    2) Joint-space PD control with nonlinear (gravity + Coriolis) compensation.

    Control law:
        tau = b_joints(q, v) + Kp (q_des - q) - Kd qdot

    let Talos robot
    - start from current posure q_ini
    - interpolate to home posture q_home over T_total seconds
    - use joint-space PD control with nonlinear compensation to track the trajectory
    - Final: stand and maintain the home posture
    
    Interpolation 是在已知起点和终点的情况下，构造一条路径，使得路径在这些点上“准确经过”。
    q_desired = pin.interpolate(model, q_ini, q_home, alpha)
    其中 alpha ∈ [0, 1] 是插值因子，表示从起点到终点的进度。
    - 当 alpha = 0 时，q_desired = q_ini（起点）
    - 当 alpha = 1 时，q_desired = q_home（终点）
    - 当 0 < alpha < 1 时，q_desired 位于 q_ini 和 q_home 之间的某个位置，具体位置由 alpha 决定。
    通过调整 alpha 的值，可以让机器人沿着这条路径平滑地从起点移动到终点。

    Spline 是一类“分段多项式轨迹”，通过额外约束（速度、加速度连续性）来保证运动平滑。
    例子： 三项式样条（三次多项式）在每个区间内使用三次多项式来插值，并确保在节点处位置、速度和加速度连续。
    这种方法可以生成平滑的轨迹，适合机器人运动控制。
    q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    其中系数 a0, a1, a2, a3 通过边界条件（位置、速度、加速度）确定。
    通过使用 Pinocchio 的插值函数，可以确保插值结果在正确的流形上，特别是对于包含四元数的配置空间。
    这对于机器人控制非常重要，因为错误的插值可能导致不自然的运动或数值不稳定。
    通过这种方式，机器人可以沿着预定的轨迹平滑地移动，同时保持物理正确性和数值稳定性。
    综上所述，插值和样条在机器人控制中起着关键作用，确保机器人能够实现平滑、自然的运动。
    """

    # ============================================================
    # Robot description
    # ============================================================
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    # ============================================================
    # Initial configuration
    # ============================================================
    z_init = 1.15  # initial floating-base height

    # Talos has 32 actuated joints
    q_actuated_home = np.zeros(32)

    # Explicit grouping for readability
    q_actuated_home[0:6] = 0.0      # left leg
    q_actuated_home[6:12] = 0.0     # right leg
    q_actuated_home[14:22] = 0.0    # left arm
    q_actuated_home[22:30] = 0.0    # right arm

    # Full Pinocchio configuration vector:
    # q = [base_position (3), base_quaternion (4), actuated_joints (32)]
    q_home_init = np.hstack([
        np.array([0.0, 0.0, z_init, 0.0, 0.0, 0.0, 1.0]),
        q_actuated_home
    ])

    # ============================================================
    # Build Pinocchio model (floating base)
    # ============================================================
    model_wrap = pin.RobotWrapper.BuildFromURDF(
        urdf,
        path_meshes,
        pin.JointModelFreeFlyer(),
        True,
        None
    )
    model = model_wrap.model
    data = model.createData()

    # ============================================================
    # Joint-space PD gains
    # ============================================================
    # Number of actuated joints
    n = model.nv - 6  # remove floating-base velocities

    Kp_diag = np.ones(n)
    Kd_diag = np.ones(n)

    # Joint indexing (Talos):
    # 0–11  : legs
    # 12–13 : torso
    # 14–21 : left arm
    # 22–29 : right arm
    # 30–31 : head

    # Legs (load-bearing)
    Kp_diag[0:12] = 800.0
    Kd_diag[0:12] = 60.0

    # Torso
    Kp_diag[12:14] = 150.0
    Kd_diag[12:14] = 20.0

    # Arms
    Kp_diag[14:30] = 2.0
    Kd_diag[14:30] = 0.2

    # Head
    Kp_diag[30:32] = 2.0
    Kd_diag[30:32] = 0.2

    Kp = np.diag(Kp_diag)
    Kd = np.diag(Kd_diag)

    # ============================================================
    # Initialize simulator and robot
    # ============================================================
    simulator = PybulletWrapper(sim_rate=1000)

    robot = Robot(
        simulator,
        urdf,
        model,
        [0.0, 0.0, z_init],
        [0.0, 0.0, 0.0, 1.0],
        q=q_home_init,
        useFixedBase=False,
        verbose=True
    )

    simulator.addLinkDebugFrame(-1, -1)

    # One step to initialize internal states
    simulator.step()
    robot.update()

    # ============================================================
    # Initial and target configurations (Pinocchio space)
    # ============================================================
    # Initial configuration
    q_ini = robot.q().copy()

    # Target home posture (full configuration)
    q_home = pin.neutral(model)
    q_home[2] = z_init                  # base height
    q_home[3:7] = [0, 0, 0, 1]           # base orientation

    # Legs
    q_home[7+0:7+6]   = [0, 0, -0.44, 0.9, -0.45, 0]   # left leg
    q_home[7+6:7+12]  = [0, 0, -0.44, 0.9, -0.45, 0]   # right leg

    # Torso
    q_home[7+12:7+14] = [0.0, 0.0]

    # Arms
    q_home[7+14:7+21] = [0, -0.24, 0, -1, 0, 0, 0]     # left arm
    q_home[7+21:7+28] = [0, -0.24, 0, -1, 0, 0, 0]     # right arm

    # Head
    q_home[7+28:7+30] = [0.0, 0.0]

    # ============================================================
    # Time-based interpolation setup
    # ============================================================
    T_total = 0.5  # total interpolation duration (seconds)
    t_start = time.time()

    # ============================================================
    # Visualization
    # ============================================================
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8]
    )

    # ============================================================
    # Main control loop
    # ============================================================
    while True:
        simulator.step()
        simulator.debug()
        robot.update()

        # --------------------------------------------------------
        # Compute interpolation factor
        # --------------------------------------------------------
        """
        project time elapsed to [0, 1] over T_total seconds
        0   → at t = 0, start of interpolation
        1   → at t = T_total, end of interpolation
        >1  → after T_total, hold the final posture
        """
        elapsed = time.time() - t_start
        alpha = min(elapsed / T_total, 1.0)

        """
        use pin.interpolate to compute the interpolated configuration instead of manual linear interpolation
        因为 q 的前 7 维包含 四元数（属于 Lie group / manifold），不能随便线性插值，否则会出现：
        - 插值结果不是单位四元数
        - 插值路径不在 SO(3) 上
        Pinocchio 的 interpolate 函数会正确处理四元数插值（使用 Slerp），确保插值结果在正确的流形上
        这样可以保证插值的物理正确性和数值稳定性
        参考资料：
        - Slerp: https://en.wikipedia.org/wiki/Slerp
        - Pinocchio interpolate doc: https://pinocchio.readthedocs.io/en/latest/api/generated/pinocchio.interpolate.html    
        """
        q_desired = pin.interpolate(model, q_ini, q_home, alpha)

        # --------------------------------------------------------
        # Read actuated joint states
        # --------------------------------------------------------
        q = robot.q()[7:]      # joint positions
        v = robot.v()[6:]      # joint velocities

        # --------------------------------------------------------
        # Nonlinear effects (gravity + Coriolis)
        # --------------------------------------------------------
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())
        b_actuated = b[6:]

        # --------------------------------------------------------
        # Joint-space PD control with feedforward compensation
        # --------------------------------------------------------
        tau = (
            b_actuated
            + Kp @ (q_desired[7:] - q)
            - Kd @ v
        )

        # Torque saturation for stability
        tau = np.clip(tau, -200.0, 200.0)

        robot.setActuatedJointTorques(tau)


if __name__ == "__main__":
    main()
