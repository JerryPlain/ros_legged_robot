## How to Run

################################################################################
# Delivery 1
################################################################################

### T1
colcon build --symlink-install
source install/setup.bash
ros2 launch ros_visuals launch_t11.py

colcon build --symlink-install
source install/setup.bash
ros2 launch ros_visuals launch_t12.py

colcon build --symlink-install
source install/setup.bash
ros2 launch ros_visuals launch_t13.py

### T2
colcon build
source install/setup.bash
ros2 run bullet_sims t2_temp

colcon build
source install/setup.bash
ros2 run bullet_sims t21

colcon build
source install/setup.bash
ros2 run bullet_sims t22

colcon build
source install/setup.bash
ros2 run bullet_sims t23

colcon build
source install/setup.bash
ros2 launch ros_visuals talos_rviz.launch.py

### T3
colcon build --packages-select bullet_sims
source install/setup.bash
ros2 run bullet_sims t3_main

colcon build
source install/setup.bash
ros2 run ros_visuals teleop_marker

"solely for the visualization of 6-DOF cubic" (Because I cannot find a way to build the connection between t3_main.py & teleoperation.py):

colcon build
source install/setup.bash
ros2 run ros_visuals interactive

Open another Terminal:
rviz2

### Answers for the questions in the tutorial 1,2,3:
See Answers_1.txt


################################################################################
# Delivery 2
################################################################################

### T4
EX1：
colcon build
source install/setup.bash
ros2 run ros_visuals t4_standing

EX2：
colcon build
source install/setup.bash
ros2 run ros_visuals one_leg_stand

EX3 and EX4: the simulation time is about 15s, after that you can (or you already can) find the graph （T4_com_comparison_plot.png） in /workspaces/ros_ws/src/ros_visuals/ros_visuals/images
colcon build
source install/setup.bash
ros2 run ros_visuals squating

### T5
EX1 and EX2:
colcon build
source install/setup.bash
ros2 run ros_visuals t51

EX3: Before beginning, please Change the f_push_mag in line 220 of t51.py from 10N to 18N for better understanding
colcon build
source install/setup.bash
ros2 run ros_visuals t51

you can do the following change in t51 for 4 different situations of balance control:
    while rclpy.ok(): # Main loop
        ############################
        # Detmine control strategies
        ############################
        use_ankle_strategy = False  # YOU CAN CHANGE IT TO "True" HERE
        use_hip_strategy = False

1. when the ankle_strategy and hip_strategy are False, the robot will fall
2. when only ankle_strategy is true, it will stand but hard
3. when only hip_strategy is true, it will stand but still hard
4. when we use both strategy, it will stand easier.

see the graphs in src/ros_visuals/ros_visuals/images

EX4：I have changed the F_push_mag to 40N
colcon build
source install/setup.bash
ros2 run ros_visuals t52

    while rclpy.ok(): # Main loop
        ############################
        # Detmine control strategies
        ############################
        use_ankle_strategy = False  # YOU CAN CHANGE IT TO "True" HERE
        use_hip_strategy = False

see the graphs in src/ros_visuals/ros_visuals/images


################################################################################
# Delivery 3
################################################################################

### T6
EX1：
source ~/drake_env/bin/activate 
python3 src/ros_visuals/ros_visuals/example_2_pydrake.py

EX2：
source ~/drake_env/bin/activate
python3 src/ros_visuals/ros_visuals/ocp_lipm_2ord.py

EX3：
source ~/drake_env/bin/activate
python3 src/ros_visuals/ros_visuals/mpc_lipm_2ord.py

### T7 Team Member:
Shijie Zhou
Yufei Hua
Yuhan Chen

EX1.1: Foot Trajectory
colcon build
source install/setup.bash
python3 src/ros_visuals/ros_visuals/foot_trajectory.py

EX1.2: Footstep Planner
colcon build
source install/setup.bash
python3 src/ros_visuals/ros_visuals/footstep_planner.py

EX1.3 & EX1.4: Walking Check
source ~/drake_env/bin/activate
colcon build
source install/setup.bash
python3 src/ros_visuals/ros_visuals/walking.py
