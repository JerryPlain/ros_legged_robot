import pybullet as pb
import numpy as np
import pinocchio as pin

from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot


def main():
    """
    Tutorial 2 – Exercise 2
    ----------------------
    Exercise 1 we have set the tau = 0, now we implement
    "closed-loop PD control with partial nonlinear compensation"

    - Joint-space PD control of a floating-base humanoid robot (Talos) with partial nonlinear compensation (gravity + Coriolis).

    Control law implemented:
        tau = b_joints(q, v) + Kp (q_des - q) - Kd qdot

    where:
        - q, qdot are actuated joint states only
        - b_joints is the actuated part of Pinocchio nonLinearEffects
    
    因为机器人关节是“转动系统”，你算出来的不是“力”，而是“让关节转起来的量”——力矩（torque）。
    线性世界（推箱子）里，我们说“力等于质量乘以加速度”（F=ma）。
    转动世界（转关节）里，我们说“力矩等于转动惯量乘以角加速度”（τ=Iα）。
    机器人控制里，我们通常说“给关节施加力矩”（apply torque to joints），
    意思是“让关节转起来”（make the joints move）。
    施加力矩的结果是关节会产生角加速度，从而带动关节运动起来。
    施加力矩的单位是牛·米（N·m），表示让关节转动的力量大小。
    1 N·m 的力矩表示在距离转动轴 1 米的位置施加 1 牛顿的力。
    施加力矩的方向决定了关节转动的方向（顺时针或逆时针）。
    施加力矩的大小决定了关节转动的加速度（转得快还是慢）。
    通过控制施加在关节上的力矩，我们可以精确地控制机器人的运动和姿态。
    综上所述，力矩是让机器人关节转动的“推动力”，
    是机器人控制中非常重要的概念。
    通过合理设计力矩控制策略，可以实现机器人的平稳运动和精确控制。
    """

    # we still need q_home, model, data, robot, simulator from previous exercises
    # q_home is not the desired joint configuration for regulation
    # we will set q_desired to be the current joint configuration at the beginning

    # ============================================================
    # Robot description (Talos)
    # ============================================================
    urdf = "src/talos_description/robots/talos_reduced.urdf"
    path_meshes = "src/talos_description/meshes/../.."

    # ============================================================
    # Initial configuration
    # ============================================================
    # Initial height of the floating base above the ground
    z_init = 1.15

    # Actuated joint configuration (Talos has 32 actuated joints)
    # All joints are initialized at zero position
    q_actuated_home = np.zeros(32)

    # (Optional explicit grouping for clarity)
    q_actuated_home[0:6] = 0.0     # left leg
    q_actuated_home[6:12] = 0.0    # right leg
    q_actuated_home[14:22] = 0.0   # left arm
    q_actuated_home[22:30] = 0.0   # right arm

    # ============================================================
    # Full Pinocchio configuration vector
    # ============================================================
    # Pinocchio convention for floating-base systems:
    #
    #   q = [ p_base (3),
    #         quaternion_base (4),
    #         actuated_joints (n) ]
    #
    # Quaternion is ordered as (x, y, z, w)

    q_home = np.hstack([
        np.array([0.0, 0.0, z_init, 0.0, 0.0, 0.0, 1.0]),
        q_actuated_home
    ])

    # ============================================================
    # Build Pinocchio model
    # ============================================================
    # JointModelFreeFlyer specifies a 6-DoF floating base
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
    # PD gains (joint-space)
    # ============================================================
    # Number of actuated joints:
    # model.nv = 6 (floating base velocity) + 32 (joint velocities)
    n = model.nv - 6

    Kp_diag = np.zeros(n)
    Kd_diag = np.zeros(n)

    """
    - leg is load-bearing, use high gains
    - torso moderate gains
    - arms and head low gains (not load-bearing)
    """

    # Joint indexing (Talos):
    # 0–11   : legs (left + right)
    # 12–13  : torso
    # 14–21  : left arm
    # 22–29  : right arm
    # 30–31  : head

    # Legs (high gains: load-bearing)
    Kp_diag[0:12] = 800.0
    Kd_diag[0:12] = 60.0

    # Torso
    Kp_diag[12:14] = 150.0
    Kd_diag[12:14] = 20.0

    # Left arm
    Kp_diag[14:22] = 5.0
    Kd_diag[14:22] = 0.5

    # Right arm
    Kp_diag[22:30] = 5.0
    Kd_diag[22:30] = 0.5

    # Head
    Kp_diag[30:32] = 2.0
    Kd_diag[30:32] = 0.2

    # Diagonal gain matrices
    Kp = np.diag(Kp_diag)
    Kd = np.diag(Kd_diag)

    # ============================================================
    # Initialize PyBullet simulator
    # ============================================================
    simulator = PybulletWrapper(sim_rate=1000)  # 1 kHz simulation

    # ============================================================
    # Create Robot wrapper (Pinocchio ↔ PyBullet bridge)
    # ============================================================
    robot = Robot(
        simulator=simulator,
        urdf=urdf,
        model=model,
        base_pos=[0.0, 0.0, z_init],
        base_quat=[0.0, 0.0, 0.0, 1.0],
        q=q_home,
        useFixedBase=False,
        verbose=True
    )

    # One step to initialize internal simulator states
    simulator.step()
    robot.update()

    # ============================================================
    # Desired joint configuration
    # ============================================================
    # .copy means we create a separate copy in memory. this is a frozen snapshot of the current joint configuration
    # control task: maintain the position at the starting configuration
    q_desired = robot.q()[7:].copy() # for testing we set desired to current joint configuration

    # ============================================================
    # Visualization
    # ============================================================
    simulator.addLinkDebugFrame(-1, -1)

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

        # Update robot state from PyBullet
        robot.update()

        # --------------------------------------------------------
        # Read actuated joint states (Pinocchio convention)
        # --------------------------------------------------------
        """
        q: [p_b(3), quat_b(4), joints(32)] → 7+32=39
        v: [v_b(3), ω_b(3), qdot(32)] → 6+32=38
        Joint-space PD Control vectors only:
        q: joints(32)
        v: qdot(32)
        """
        q = robot.q()[7:]     # joint positions
        v = robot.v()[6:]     # joint velocities

        # --------------------------------------------------------
        # Nonlinear effects (gravity + Coriolis)
        # --------------------------------------------------------
        # Full vector includes floating base + joints
        b = pin.nonLinearEffects(model, data, robot.q(), robot.v())

        # Extract actuated joint part only
        b_actuated = b[6:] # only take joint-related nonlinear effects

        # --------------------------------------------------------
        # Joint-space PD control with nonlinear compensation
        # --------------------------------------------------------
        """
        tau is the torque command we will send to the robot
        using the control law:
        tau = (
            b_actuated
            + Kp @ (q_desired - q) 
            - Kd @ v
        )
        where:
            - b_actuated: nonlinear effects (gravity + Coriolis) for actuated joints
            - Kp, Kd: joint-space PD gain matrices
            - q_desired: desired joint positions
            - q: current joint positions
            - v: current joint velocities

        Kp: the more the error in position, the more torque we apply to correct it
        Kd: the more the velocity, the more torque we apply to damp it
        𝜏: the torque command we send to the robot in the simulator
        """
        tau = (
            b_actuated
            + Kp @ (q_desired - q)
            - Kd @ v
        )

        # Torque saturation for numerical stability
        tau = np.clip(tau, -200.0, 200.0)

        # Apply torques to the simulator
        robot.setActuatedJointTorques(tau)

if __name__ == "__main__":
    main()