# Simulation-based Modeling and Control of Humanoid Robot

A research-driven ROS2 workspace for building and evaluating humanoid locomotion pipelines on Talos, from rigid-body geometry to full walking simulation.

## Highlights

- End-to-end tutorial chain: **T1 -> T7**
- Tight integration of **ROS2 + PyBullet + Pinocchio + TSID + Drake/Pydrake**
- Reusable simulation bridge between physics (`PyBullet`) and model-based control (`Pinocchio`/`TSID`)
- Balance analysis with **ZMP/CMP/DCM** and disturbance-recovery strategy comparisons
- Reproducible scripts for standing, one-leg support, squat, push recovery, MPC, and walking

## System Stack

| Layer | Main Tools |
|---|---|
| Robotics middleware | ROS2 (`rclpy`, TF, RViz) |
| Physics | PyBullet |
| Kinematics/Dynamics | Pinocchio |
| Whole-body control | TSID |
| OCP/MPC learning tasks | Drake / Pydrake |

## Repository Layout

```text
ros_legged_robot/
├── bullet_sims/         # T2/T3: dynamics and control executables
├── ros_visuals/         # T1/T4/T5/T6/T7: visualization and algorithm scripts
├── simulator/           # Shared simulation wrappers
├── talos_description/   # Talos URDF + meshes
├── reemc_description/   # Additional robot description assets
├── QA/                  # Structured answer documents for submissions
└── docs/                # Project conclusions and supporting notes
```

## Package Map

| Package | Role | Entrypoints |
|---|---|---|
| `bullet_sims` | Dynamics and low-level control exercises | `t2_temp`, `t21`, `t22`, `t23`, `t3_main` |
| `ros_visuals` | TF demos, TSID tasks, balance experiments, walking pipeline | `t11`, `t12`, `t13`, `t4_standing`, `one_leg_stand`, `squating`, `t51`, `t52`, `teleop_marker`, `interactive` |
| `simulator` | Utilities for simulation stepping, robot abstraction, and state/control bridging | Library package (no CLI entrypoint) |

## Tutorial Roadmap (T1-T7)

| Tutorial | Focus | Key Scripts |
|---|---|---|
| T1 | SE(3), twist, wrench transforms + TF broadcasting | `t11.py`, `t12.py`, `t13.py` |
| T2 | Floating-base dynamics and joint-space control | `t2_temp.py`, `t21.py`, `t22.py`, `t23.py` |
| T3 | Two-stage control (joint-space -> Cartesian-space) | `t3_main.py` |
| T4 | TSID standing / one-leg support / squat | `t4_standing.py`, `one_leg_stand.py`, `squating.py` |
| T5 | ZMP/CMP/DCM and push-recovery strategy comparison | `t51.py`, `t52.py` |
| T6 | OCP/MPC foundations | `example_2_pydrake.py`, `ocp_lipm_2ord.py`, `mpc_lipm_2ord.py` |
| T7 | Footstep planning + swing-foot trajectory + walking integration | `footstep_planner.py`, `foot_trajectory.py`, `lip_mpc.py`, `walking.py` |

## Environment

- Ubuntu + ROS2
- Python 3.10+
- `colcon`
- Python deps: `pybullet`, `pinocchio`, `tsid`, `numpy`, `scipy`, `matplotlib`
- Optional (T6/T7): `pydrake`

> Dependency versions are currently not pinned. For reproducibility, use a local lockfile or environment export.

## Quick Start

### 1. Build Workspace

```bash
cd ros_legged_robot
colcon build --symlink-install
source install/setup.bash
```

### 2. Run Experiments

```bash
# T1
ros2 launch ros_visuals launch_t11.py
ros2 launch ros_visuals launch_t12.py
ros2 launch ros_visuals launch_t13.py

# T2
ros2 run bullet_sims t2_temp
ros2 run bullet_sims t21
ros2 run bullet_sims t22
ros2 run bullet_sims t23
ros2 launch ros_visuals talos_rviz.launch.py

# T3
ros2 run bullet_sims t3_main

# T4
ros2 run ros_visuals t4_standing
ros2 run ros_visuals one_leg_stand
ros2 run ros_visuals squating

# T5
ros2 run ros_visuals t51
ros2 run ros_visuals t52

# T6
source ~/drake_env/bin/activate
python3 ros_visuals/ros_visuals/example_2_pydrake.py
python3 ros_visuals/ros_visuals/ocp_lipm_2ord.py
python3 ros_visuals/ros_visuals/mpc_lipm_2ord.py

# T7
python3 ros_visuals/ros_visuals/foot_trajectory.py
python3 ros_visuals/ros_visuals/footstep_planner.py
python3 ros_visuals/ros_visuals/walking.py
```

Optional interaction tools:

```bash
ros2 run ros_visuals teleop_marker
ros2 run ros_visuals interactive
rviz2
```

## Balance Strategy Switches (T5)

In `t51.py` and `t52.py`:

- `use_ankle_strategy`
- `use_hip_strategy`

Suggested comparison modes:

- no strategy
- ankle only
- hip only
- ankle + hip

## Outputs

Generated figures are saved in:

- `ros_legged_robot/ros_visuals/ros_visuals/images/`

Typical artifacts:

- `T4_com_comparison_plot.png`
- `t51_plot_*.png`
- `t52_plot_*.png`

## QA Documents

- `ros_legged_robot/QA/submission_01_tutorials_1_2.md`
- `ros_legged_robot/QA/submission_02_tutorial_5.md`

## Reproducibility Notes

- Use `--symlink-install` for fast iteration on Python packages
- Re-source setup after each rebuild:

```bash
source ros_legged_robot/install/setup.bash
```

- Some launch scripts assume standard ROS workspace layout
- Run push-recovery strategies separately, then compare saved plots
