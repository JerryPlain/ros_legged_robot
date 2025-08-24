import numpy as np
import pinocchio as pin # Pinocchio for calculating poses(SE3), Transforms, dynamics
import time
from simulator.pybullet_wrapper import PybulletWrapper
from enum import Enum # enumeration for foot identification

################################################
# 枚举类 `Side` 表示左右脚，创建足步时标明是左脚还是右脚
################################################
class Side(Enum):
    """Side
    Describes which foot to use
    """
    LEFT=0
    RIGHT=1

def other_foot_id(id): # switch the foot id
    if id == Side.LEFT:
        return Side.RIGHT
    else:
        return Side.LEFT

#####################################################
# 类 `FootStep` 表示单个足步，包含位姿、脚底形状和脚的侧面信息
#####################################################   
class FootStep:
    """FootStep
    Holds all information describing a single footstep
    """
    def __init__(self, pose, footprint, side=Side.LEFT):
        """inti FootStep

        Args:
            pose (pin.SE3): the pose of the footstep
            footprint (np.array): 3 by n matrix of foot vertices
            side (_type_, optional): Foot identifier. Defaults to Side.LEFT.
        """
        self.pose = pose
        self.footprint = footprint
        self.side = side
        
    def poseInWorld(self):
        return self.pose # get the pose of the footstep in world coordinates
        
    def plot(self, simulation):
    
        # Calculate corners of the foot rectangle from footprint
        # footprint format: [front_x, back_x, outer_y, inner_y]
        p1 = np.array([self.footprint[0], self.footprint[2], 0.002])  # front outer
        p2 = np.array([self.footprint[0], self.footprint[3], 0.002])  # front inner
        p3 = np.array([self.footprint[1], self.footprint[3], 0.002])  # back inner
        p4 = np.array([self.footprint[1], self.footprint[2], 0.002])  # back outer
        
        # Transform points to world frame, because the pose is in world coordinates
        p1_w = self.pose.act(p1) # self.pose.act applies the transformation to the point
        p2_w = self.pose.act(p2)
        p3_w = self.pose.act(p3)
        p4_w = self.pose.act(p4)
        
        # Set color based on foot side
        color = [1, 0, 0] if self.side == Side.LEFT else [0, 0, 1]
        
        #  使用 PyBullet API 显示矩形框
        import pybullet as pb
        pb.addUserDebugLine(p1_w, p2_w, color)
        pb.addUserDebugLine(p2_w, p3_w, color)
        pb.addUserDebugLine(p3_w, p4_w, color)
        pb.addUserDebugLine(p4_w, p1_w, color)
        
        # Display side text，在脚中心正上方（+0.02m）显示 “LEFT” 或 “RIGHT” 文本
        text_pos = self.pose.translation + np.array([0, 0, 0.02])
        side_text = "LEFT" if self.side == Side.LEFT else "RIGHT"
        pb.addUserDebugText(side_text, text_pos, color)
    
        return None

class FootStepPlanner:
    """FootStepPlanner
    Creates footstep plans (list of right and left steps)
    """
    
    def __init__(self, conf):
        self.conf = conf
        self.steps = []
        
    def planLine(self, T_0_w, side, no_steps):
        """
        生成一条直线上的步伐序列，左右脚交替，起止都双脚并排。

        Args:
            T_0_w (pin.SE3): 机器人初始世界位姿
            side (Side): 起始迈步的脚（左脚或右脚）
            no_steps (int): 需要迈的步数（不包括起止并排的两步）

        Returns:
            list: 步伐序列 每个元素是FootStep对象
        """
        
        # the displacement between steps in x and y direction
        dx = self.conf.step_size_x # the forward step size
        dy = 2 * self.conf.step_size_y # the distance between the feet
        
        # the footprint of the robot
        lfxp, lfxn = self.conf.lfxp, self.conf.lfxn
        lfyp, lfyn = self.conf.lfyp, self.conf.lfyn
        
        # Create left and right footprints
        left_footprint = np.array([lfxp, lfxn, lfyp, lfyn])
        right_footprint = np.array([lfxp, lfxn, -lfyn, -lfyp])  # Mirrored y-coordinates
    
        steps = []
    
        # Add first step at initial pose
        first_footprint = left_footprint if side == Side.LEFT else right_footprint
        first_pose = T_0_w.copy()
        if side == Side.LEFT:
            first_pose.translation[1] += dy/2
        else:
            first_pose.translation[1] -= dy/2
        first_step = FootStep(first_pose, first_footprint, side)
        steps.append(first_step)

        # Add second step parallel to first (robot starts with both feet)
        second_side = other_foot_id(side)
        second_footprint = left_footprint if second_side == Side.LEFT else right_footprint

        second_pose = pin.SE3(T_0_w.rotation, T_0_w.translation.copy())
        if second_side == Side.LEFT:
            second_pose.translation[1] += dy/2
        else:
            second_pose.translation[1] -= dy/2
        second_step = FootStep(second_pose, second_footprint, second_side)
        steps.append(second_step)
    
        # Generate remaining steps
        current_side = second_side
        current_pose = second_pose
    
        for i in range(no_steps - 2):  # -2 because we already added two steps
            current_side = other_foot_id(current_side)
            current_footprint = left_footprint if current_side == Side.LEFT else right_footprint
        
            # Create new pose
            next_pose = pin.SE3(current_pose.rotation, current_pose.translation.copy())
            next_pose.translation[0] += dx  # Move forward
        
            # Set y position based on side
            if current_side == Side.LEFT:
                next_pose.translation[1] = T_0_w.translation[1] + dy/2
            else:
                next_pose.translation[1] = T_0_w.translation[1] - dy/2
            
            next_step = FootStep(next_pose, current_footprint, current_side)
            steps.append(next_step)
            current_pose = next_pose
    
        # Add final step parallel to last one (robot ends with both feet)
        final_side = other_foot_id(current_side)
        final_footprint = left_footprint if final_side == Side.LEFT else right_footprint
    
        final_pose = pin.SE3(current_pose.rotation, current_pose.translation.copy())
        # Keep same x position as last step
        if final_side == Side.LEFT:
            final_pose.translation[1] = T_0_w.translation[1] + dy/2
        else:
            final_pose.translation[1] = T_0_w.translation[1] - dy/2
        
        final_step = FootStep(final_pose, final_footprint, final_side)
        steps.append(final_step)
    
        self.steps = steps
        return steps

    
    def plot(self, simulation):
        for step in self.steps:
            step.plot(simulation)

            
if __name__=='__main__':
    """ Test footstep planner
    """ 
    # Define configuration class
    class Config:
        def __init__(self):
            # Step parameters
            self.step_size_x = 0.2  # Forward step size
            self.step_size_y = 0.1  # Half distance between feet
            
            # Foot dimensions
            self.lfxp = 0.12  # Front edge
            self.lfxn = -0.08  # Back edge
            self.lfyp = 0.05  # Outer edge
            self.lfyn = -0.05  # Inner edge
    
    # Create simulator
    sim = PybulletWrapper() # Initialize the simulator
    sim.resetSimulation() # Reset simulation environment
    
    # Ensure floor is visible by creating it explicitly
    import pybullet as pb
    pb.loadURDF("plane.urdf", [0, 0, 0]) # Load a simple plane as the ground
    
    # Create planner and configuration
    conf = Config() # Create configuration object with step parameters
    planner = FootStepPlanner(conf)
    
    # Create initial pose
    initial_pose = pin.SE3.Identity()
    initial_pose.translation = np.array([0, 0, 0.01])  # Slightly above ground
    
    # Generate steps
    num_steps = 8
    steps = planner.planLine(initial_pose, Side.LEFT, num_steps)
    
    # Print step information
    print(f"Generated {len(steps)} steps:")
    for i, step in enumerate(steps):
        side_text = "LEFT" if step.side == Side.LEFT else "RIGHT"
        pos = step.pose.translation
        print(f"Step {i}: {side_text} at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    # Visualize steps
    planner.plot(sim)
    
    # Keep simulation running for visualization
    print("Press Ctrl+C to exit...")
    try:
        while True:
            sim.step()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Simulation stopped")