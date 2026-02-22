# legged_ros 项目总结（中文）

## 1. 项目概况
本项目是一个基于 **ROS2 + PyBullet + Pinocchio + TSID + Drake** 的双足人形机器人学习/实践仓库，核心对象是 Talos 模型。你把教程任务按阶段拆分为可运行脚本，覆盖了：
- 刚体位姿与坐标变换（SE(3) / twist / wrench）
- 浮动基机器人动力学建模与控制（PD、逆动力学）
- ROS2 可视化链路（`joint_states`、TF、RViz）
- 全身控制（TSID）与平衡策略（ZMP/CMP/DCM）
- LIPMPC + 足步规划 + 摆动脚轨迹 + 行走仿真

## 2. 你完成了什么（按交付阶段）

## 2.1 Delivery 1（基础建模、仿真与ROS可视化）
你完成了从“运动学表示”到“动力学控制”再到“ROS可视化”的全链路打通。

### T1：SE(3) 框架与坐标变换可视化
- 在 `ros_visuals/ros_visuals/t11.py` 构建 8 个立方体角点 frame，使用 `pin.exp6` 按固定 twist 积分中心位姿。
- 在 `ros_visuals/ros_visuals/t12.py` 实现 twist 坐标变换，发布 `twist_w` 与 `twist_o6`。
- 在 `ros_visuals/ros_visuals/t13.py` 实现 wrench 坐标变换，发布 `wrench_w` 与 `wrench_o6`，并用矩阵法与 `actInv` 做一致性校验。
- 配套完成 launch 与 RViz 配置（`launch_t11.py`、`launch_t12.py`、`launch_t13.py`）。

### T2：PyBullet-Pinocchio 桥接与关节控制
- 在 `bullet_sims/t2_temp.py` 完成浮动基状态读取与 `M(q)、b(q,v)` 计算。
- 在 `bullet_sims/t21.py` 实现关节空间 PD + 非线性补偿控制。
- 在 `bullet_sims/t22.py` 实现从初始位姿到 home 位姿的插值跟踪控制。
- 在 `bullet_sims/t23.py` 增加 ROS2 `joint_states` 发布，实现 PyBullet 状态在 RViz 中可视化。

### T3：两阶段控制（关节空间 -> 笛卡尔空间）
- 在 `bullet_sims/t3_main.py` 设计状态机：先关节样条过渡，再切换到末端笛卡尔控制。
- 完成关节空间逆动力学跟踪控制器 + 笛卡尔空间阻尼伪逆控制器。
- 关键更新点体现在最近提交描述：`update joint-space and cartesion space pd control`。

## 2.2 Delivery 2（TSID站立、单腿平衡、抗扰实验）
你把控制重心从单关节控制提升到全身任务级控制，并完成了平衡评估闭环。

### T4：TSID 站立与单腿支撑
- `ros_visuals/ros_visuals/t4_standing.py`：完成双脚支撑站立，发布 TF + `joint_states`。
- `ros_visuals/ros_visuals/one_leg_stand.py`：先移动 CoM 到支撑脚，再切换单脚支撑并抬脚。
- `ros_visuals/ros_visuals/squating.py`：加入周期性 CoM 高度调制（下蹲），并记录/绘制参考与实测轨迹。

### T5：ZMP/CMP/DCM 与抗扰平衡控制
- `ros_visuals/ros_visuals/t51.py`：
  - 读取双踝力矩传感器，估计单脚/全局 ZMP；
  - 计算 CMP、DCM；
  - 接入踝策略与髋策略；
  - 实现周期外力扰动注入与可视化；
  - 输出多组对比图（无控制/踝/髋/联合）。
- `ros_visuals/ros_visuals/t52.py`：
  - 将外力级别提升到 40N 级；
  - 引入“TSID虚拟状态积分 + 位置接口”模式，与 `t51` 的力矩接口形成对照实验；
  - 继续输出策略对比图。

## 2.3 Delivery 3（MPC与行走）
你完成了从步态规划到在线 MPC 的行走主循环搭建。

### T6：最优控制基础
- `example_2_pydrake.py`、`ocp_lipm_2ord.py`、`mpc_lipm_2ord.py`：实现摆系统与 LIP 的 OCP/MPC 练习。

### T7：足步规划 + LIPMPC + 摆动脚轨迹 + 行走集成
- `footstep_planner.py`：实现左右脚交替、起止并脚的直线足步生成与可视化。
- `foot_trajectory.py`：实现摆动脚轨迹插值框架。
- `lip_mpc.py`：实现 LIP 连续/离散模型、MPC优化问题、ZMP参考生成。
- `walking.py`：集成 TSID + 足步计划 + MPC + 摆动脚轨迹，形成完整行走仿真主循环与日志输出。

## 3. 你搭建的工程化能力

### 3.1 可复用底层仿真接口
- `simulator/` 包（`pybullet_wrapper.py`、`body.py`、`robot.py`）构建了 PyBullet 与 Pinocchio 的统一桥接层。
- 支持浮动基状态映射、关节控制模式切换、调试绘图、外力施加、传感器读数访问。

### 3.2 可执行入口与运行组织
- `bullet_sims/setup.py` 暴露 `t2_temp/t21/t22/t23/t3_main`。
- `ros_visuals/setup.py` 暴露 `t11/t12/t13/t4_standing/one_leg_stand/squating/t51/t52/teleop_marker/interactive`。
- `ros_legged_robot/README.md` 按 Delivery 提供运行命令，形成可复现实验手册。

### 3.3 结果资产沉淀
- 生成并保存了 T4/T5 多组对比图（`ros_visuals/ros_visuals/images/`）。
- 提交了阶段答题材料（`submission_01_tutorials_1_2.md`、`submission_02_tutorial_5.md`）。

## 4. 当前仓库结构结论
- ROS工作主体在 `ros_legged_robot/`，包含 5 个 ROS 包：
  - `ros_visuals`
  - `bullet_sims`
  - `simulator`
  - `talos_description`
  - `reemc_description`
- 代码规模上，控制与算法脚本主要集中在：
  - `ros_visuals/ros_visuals/*.py`（24个）
  - `bullet_sims/bullet_sims/*.py`（6个）
  - `simulator/simulator/*.py`（6个）

## 5. 从提交记录看你的工作轨迹
- `2025-11-19`：`init legged_ros`，完成仓库基础搭建与教程代码导入。
- `2026-02-11`：`update joint-space and cartesion space pd control`，继续强化 T3 中关节空间/笛卡尔空间控制实现。

## 6. 总结（一句话）
你把一个“教程型双足机器人项目”从数学建模、控制实现、ROS接口、可视化到行走集成完整跑通，并且在平衡控制（ZMP/CMP/DCM + 踝/髋策略）上做了成体系的实验与对比产出。
