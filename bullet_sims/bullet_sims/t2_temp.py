import pybullet as pb
import numpy as np
import time

import pinocchio as pin # Pinocchio library for rigid body dynamics
from simulator.pybullet_wrapper import PybulletWrapper #  PyBullet simulator wrapper
from simulator.robot import Robot # Bridge: Pinocchio (q/v) <-> PyBullet (joint/base state transfer)


def main():
    """
    Tutorial 2 – Exercise 1
    -----------------------
    This script demonstrates how to:
    1) Use Pybullet to load Talos robot model
    2) Use Pinocchio to set up the robot model and compute dynamics quantities
    3) Bridge the PyBullet simulation and Pinocchio model using a Robot wrapper (use RobotWrapper to transfer the Pybullet state to pinocchio standard (q, v) format)
    4) Read the robot state and compute the inertia matrix M(q) and nonlinear effects b(q, v)
    """

    # ============================================================
    # Robot description (URDF + meshes)
    # ============================================================
    # NOTE:
    # - The same URDF is used by both PyBullet (simulation)
    #   and Pinocchio (analytical dynamics model).
    # - path_meshes is required by Pinocchio to locate visual/collision meshes.

    urdf = "src/talos_description/robots/talos_reduced.urdf" # URDF is structure/kinematics/dynamics description of the robot
    path_meshes = "src/talos_description/meshes/../.."

    # ============================================================
    # Actuated joint indexing (Talos)
    # ============================================================
    """
    Talos actuated joints ordering:

    0–5     : left leg
    6–11    : right leg
    12–13   : torso              # torso is 2 DoF, pitch and roll
    14–21   : left arm
    22–29   : right arm
    30–31   : head
    """

    # ============================================================
    # Initial configuration
    # ============================================================
    # Initial base height above the ground
    z_init = 1.15

    # Actuated joint positions (32 DoF for Talos)
    # All joints are initialized to zero position
    q_actuated_home = np.zeros(32)  # 32 actuated joints, not including floating base

    # ============================================================
    # Full Pinocchio configuration vector (floating base + joints)
    # ============================================================
    # Pinocchio convention for floating-base systems:
    # q = [ p_base (3), # base position
    #       quaternion_base (4), # base orientation x, y, z, w
    #       actuated_joints (n) ] # actuated joint positions
    #
    # Quaternion is given as (x, y, z, w)

    # because the base is floating, we need to add 7 DoF for the base
    """
    Floating base configuration vector structure: (in pinocchio)
    - 3 DoF for base position (x, y, z)
    - 4 DoF for base orientation (quaternion: x, y, z, w)
    - use np.array to ensure the first part is an array,
      so that np.hstack can concatenate with q_actuated_home properly.
    - write base pose like this means the Floating base of Talos is at origin with height z_init and no rotation (in world coordinates SE(3))
    """
    # if not using np.array, it would be a list
    q_home = np.hstack([ # hstack anticipated array-like input and stacks them in sequence horizontally (column wise)
        np.array([0.0, 0.0, z_init, 0.0, 0.0, 0.0, 1.0]), # base position and orientation, means at origin with height z_init without rotation
        q_actuated_home # actuated joints
    ])

    # ============================================================
    # Build Pinocchio model and data
    # ============================================================
    """
    JointModelFreeFlyer(): Tell Pinocchio "this is a floating base robot"
    - velocity vector structure:
      v = [ v_base_linear (3),  # base linear velocity
            v_base_angular (3), # base angular velocity
            qdot_joints (n) ]   # actuated joint velocities
    - position vector structure:
      q = [ p_base (3),        # base position
            quaternion_base (4), # base orientation
            q_joints (n) ]      # actuated joint positions

    - model: contains the robot structure, kinematics, dynamics parameters
    - data: contains temporary variables to perform computations
    """
    model_wrapper = pin.RobotWrapper.BuildFromURDF(
        urdf,
        path_meshes,
        pin.JointModelFreeFlyer(), # floating base
        True, # verbose
        None # package directories (None if not using ROS packages)
    )
    model = model_wrapper.model # Pinocchio model
    data = model.createData() # Pinocchio data

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

    # One simulation step to initialize internal states
    simulator.step()
    robot.update()

    # ============================================================
    # Read robot state in Pinocchio format
    # ============================================================
    """
    q: [p_b(3), quat_b(4), joints(32)] → 7+32=39
    v: [v_b(3), ω_b(3), qdot(32)] → 6+32=38
    """
    q = robot.q()   # Generalized position vector
    v = robot.v()   # Generalized velocity vector

    # after transferring the state from PyBullet to Pinocchio, we can compute dynamics quantities

    # ============================================================
    # Compute dynamics quantities using Pinocchio
    # ============================================================
    # Joint-space inertia matrix M(q)
    M = pin.crba(model, data, q)

    # Nonlinear effects: Coriolis + centrifugal + gravity
    b = pin.nonLinearEffects(model, data, q, v)

    print("Inertia Matrix M(q):\n", M)
    print("Nonlinear Effects Vector b(q, v):\n", b)

    # ============================================================
    # Visualization setup
    # ============================================================
    simulator.addLinkDebugFrame(-1, -1)

    pb.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=90,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.8]
    )

    # ============================================================
    # Zero torque command (robot will fall under gravity)
    # ============================================================
    # all actuated joints zero torque -> for testing if the robot falls under gravity
    tau = np.zeros_like(q_actuated_home)

    # ============================================================
    # Main simulation loop
    # ============================================================
    start_time = time.time()
    done = False

    while not done:
        simulator.step() # step the simulation
        simulator.debug() # debug visualization

        # Update robot state from PyBullet
        robot.update()

        # Apply actuator torques
        # take the calculated torque command tau and send it to the robot wrapper to apply in PyBullet (really applying in Robot in PyBullet)
        robot.setActuatedJointTorques(tau) # Pinocchio‘ τ  →  Robot wrapper  →  PyBullet

        # Stop after 10 seconds (let the simulation run for a while)
        if time.time() - start_time > 10.0:
            done = True

# conduct main function only if the script is executed directly
# if the script is imported as a module, the main function will not be executed
# because the meaning of import is to use the functions/classes defined in the script without running the script itself
if __name__ == "__main__":
    main()