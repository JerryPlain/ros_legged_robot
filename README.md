# legged_ros: A ROS2 Humanoid Locomotion and Control Playground

A research-oriented ROS2 workspace for humanoid robotics, focused on **Talos simulation**, **whole-body control**, and **walking generation**.
The repository integrates rigid-body geometry, dynamics, optimization, and visualization workflows using:

- **ROS2** (`rclpy`, TF, RViz)
- **PyBullet** (physics simulation)
- **Pinocchio** (kinematics/dynamics)
- **TSID** (task-space inverse dynamics)
- **Drake/Pydrake** (OCP/MPC exercises)

## Highlights

- End-to-end progression from SE(3) fundamentals to dynamic walking.
- Clear tutorial-style task split: **T1 → T7**.
- Practical balance experiments with **ZMP/CMP/DCM** and disturbance rejection.
- Reusable simulation bridge between PyBullet and Pinocchio.
- ROS-native visualization pipeline (`joint_states`, TF, RViz).

## Repository Structure

```text
.
├── ros_legged_robot/
│   ├── bullet_sims/         # Tutorial 2/3 simulation and control scripts
│   ├── ros_visuals/         # Tutorial 1/4/5/6/7 scripts and RViz configs
│   ├── simulator/           # PyBullet-Pinocchio wrapper utilities
│   ├── talos_description/   # Talos URDF and meshes
│   └── reemc_description/   # Additional robot description assets
├── 项目总结_中文.md          # Chinese project summary
└── DIVERSITY_METRICS_README.md  # Additional technical notes
```

## Core Modules

### `simulator`
Provides reusable wrappers:

- `PybulletWrapper`: simulation stepping, debug drawing, utility methods
- `Body`: PyBullet body abstraction
- `Robot`: state/control bridge between PyBullet and Pinocchio spaces

### `bullet_sims`
Main executables:

- `t2_temp`: model loading + `M(q)` and `b(q, v)` inspection
- `t21`: joint-space PD with nonlinear compensation
- `t22`: interpolation tracking to home posture
- `t23`: ROS2 `joint_states` publishing for RViz
- `t3_main`: two-phase control (joint-space → Cartesian-space)

### `ros_visuals`
Main executables:

- `t11`, `t12`, `t13`: SE(3), twist, wrench transformations + TF publishing
- `t4_standing`, `one_leg_stand`, `squating`: TSID standing and balance tasks
- `t51`, `t52`: ZMP/CMP/DCM estimation + ankle/hip balance strategies under pushes
- `walking`: integrated footstep planning + LIPMPC + swing-foot trajectory

## Prerequisites

Recommended environment:

- Ubuntu + ROS2 (Humble/Foxy-compatible Python workflows)
- Python 3.10+
- `colcon`
- `pybullet`, `pinocchio`, `tsid`, `numpy`, `scipy`, `matplotlib`
- Optional for T6/T7 optimization exercises: `pydrake`

> Note: exact package versions are not pinned in this repository.

## Quick Start

### 1. Build workspace

```bash
cd ros_legged_robot
colcon build --symlink-install
source install/setup.bash
```

### 2. Run selected tasks

### Tutorial 1 (SE(3), Twist, Wrench)

```bash
ros2 launch ros_visuals launch_t11.py
ros2 launch ros_visuals launch_t12.py
ros2 launch ros_visuals launch_t13.py
```

### Tutorial 2 (Dynamics and Joint Control)

```bash
ros2 run bullet_sims t2_temp
ros2 run bullet_sims t21
ros2 run bullet_sims t22
ros2 run bullet_sims t23
ros2 launch ros_visuals talos_rviz.launch.py
```

### Tutorial 3 (Two-Phase Control)

```bash
ros2 run bullet_sims t3_main
```

Optional marker tools:

```bash
ros2 run ros_visuals teleop_marker
ros2 run ros_visuals interactive
rviz2
```

### Tutorial 4 (TSID Standing / One-Leg / Squat)

```bash
ros2 run ros_visuals t4_standing
ros2 run ros_visuals one_leg_stand
ros2 run ros_visuals squating
```

### Tutorial 5 (Balance Metrics and Push Recovery)

```bash
ros2 run ros_visuals t51
ros2 run ros_visuals t52
```

Control strategy toggles are exposed in `t51.py` and `t52.py`:

- `use_ankle_strategy`
- `use_hip_strategy`

This enables controlled comparisons:

- no strategy
- ankle only
- hip only
- ankle + hip

### Tutorial 6 (OCP/MPC Foundations)

```bash
source ~/drake_env/bin/activate
python3 ros_visuals/ros_visuals/example_2_pydrake.py
python3 ros_visuals/ros_visuals/ocp_lipm_2ord.py
python3 ros_visuals/ros_visuals/mpc_lipm_2ord.py
```

### Tutorial 7 (Walking Pipeline)

```bash
python3 ros_visuals/ros_visuals/foot_trajectory.py
python3 ros_visuals/ros_visuals/footstep_planner.py
python3 ros_visuals/ros_visuals/walking.py
```

## Outputs and Artifacts

Generated plots are saved in:

- `ros_legged_robot/ros_visuals/ros_visuals/images/`

Notable artifacts include balance comparison plots for T4/T5 (e.g. `T4_com_comparison_plot.png`, `t51_plot_*.png`, `t52_plot_*.png`).

## Reproducibility Notes

- Build with `--symlink-install` to iterate quickly on Python sources.
- Re-source the workspace after each build:

```bash
source ros_legged_robot/install/setup.bash
```

- Some launch files assume ROS workspace-style paths.
- For push-recovery experiments, run each strategy setting separately and compare generated plots.