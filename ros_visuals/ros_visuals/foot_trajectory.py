import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

# import ndcurves, scipy, numpy, etc... to do your splines

class SwingFootTrajectory:
    """SwingFootTrajectory
    Interpolate Foot trajectory between SE3 T0 and T1
    """
    def __init__(self, T0, T1, duration, height=0.05):
        """initialize SwingFootTrajectory

        Args:
            T0 (pin.SE3): Inital foot pose
            T1 (pin.SE3): Final foot pose
            duration (float): step duration
            height (float, optional): setp height. Defaults to 0.05.
        """
        self._height = height
        self._t_elapsed = 0.0
        self._duration = duration
        self.reset(T0, T1)

    def reset(self, T0, T1):
        '''reset back to zero, update poses
        '''
        #>>>>TODO: plan the spline
        self._t_elapsed = 0.0
        self._T0 = T0
        self._T1 = T1
        
        # Extract positions
        self._p0 = T0.translation
        self._p1 = T1.translation
        
        # Compute polynomial coefficients
        # We need 5th order polynomial to satisfy position, velocity and acceleration constraints
        # For X and Y, we use: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        # Store delta position for efficiency
        self._delta_p = self._p1 - self._p0
        
        # For Z direction, we need to ensure we reach height at the middle point
        self._z_ground = max(self._p0[2], self._p1[2])
        self._z_apex = self._z_ground + self._height

    def isDone(self):
        return self._t_elapsed >= self._duration 
    
    def evaluate(self, t):
        """evaluate at time t
        """
        #>>>>TODO: evaluate the spline at time t, return pose, velocity, acceleration
        # Clamp time to valid range
        t = min(max(t, 0.0), self._duration)
        self._t_elapsed = t
        
        # Normalized time [0,1]
        s = t / self._duration if self._duration > 0 else 0.0
        
        # Minimum jerk trajectory (5th order polynomial)
        # s(t) = 10*t^3 - 15*t^4 + 6*t^5
        # This satisfies s(0)=0, s(1)=1, s'(0)=0, s'(1)=0, s''(0)=0, s''(1)=0
        s_t = 10 * s**3 - 15 * s**4 + 6 * s**5
        ds_t = (30 * s**2 - 60 * s**3 + 30 * s**4) / self._duration
        dds_t = (60 * s - 180 * s**2 + 120 * s**3) / (self._duration**2)
        
        # Calculate position for X and Y
        pos_xy = self._p0[:2] + self._delta_p[:2] * s_t
        
        # For Z, we use a function that goes up in the middle and back down
        # We use a function that's 0 at s=0 and s=1, and 1 at s=0.5
        # f(s) = 4*s*(1-s) satisfies this
        height_factor = 4 * s * (1 - s)
        dheight_factor = 4 * (1 - 2*s) / self._duration
        ddheight_factor = -8 / (self._duration**2)
        
        # Calculate Z position
        z_base = self._p0[2] + (self._p1[2] - self._p0[2]) * s_t
        z_height = (self._z_apex - self._z_ground) * height_factor
        pos_z = z_base + z_height
        
        # Calculate velocity
        vel_xy = self._delta_p[:2] * ds_t
        vel_z = (self._p1[2] - self._p0[2]) * ds_t + (self._z_apex - self._z_ground) * dheight_factor
        
        # Calculate acceleration
        acc_xy = self._delta_p[:2] * dds_t
        acc_z = (self._p1[2] - self._p0[2]) * dds_t + (self._z_apex - self._z_ground) * ddheight_factor
        
        # Combine results
        position = np.array([pos_xy[0], pos_xy[1], pos_z])
        velocity = np.array([vel_xy[0], vel_xy[1], vel_z])
        acceleration = np.array([acc_xy[0], acc_xy[1], acc_z])
        
        # Create pose (keeping orientation from initial pose)
        pose = pin.SE3(self._T0.rotation, position)
        
        return pose, velocity, acceleration

if __name__=="__main__":
    T0 = pin.SE3(np.eye(3), np.array([0, 0, 0]))
    T1 = pin.SE3(np.eye(3), np.array([0.2, 0, 0]))

    #>>>>TODO: plot to make sure everything is correct
    # Create trajectory with 1 second duration and 0.05m height
    duration = 1.0
    height = 0.05
    traj = SwingFootTrajectory(T0, T1, duration, height)
    
    # Sample trajectory at multiple time points
    num_samples = 100
    time_points = np.linspace(0, duration, num_samples)
    
    # Arrays to store results
    positions = np.zeros((num_samples, 3))
    velocities = np.zeros((num_samples, 3))
    accelerations = np.zeros((num_samples, 3))
    
    # Evaluate trajectory at each time point
    for i, t in enumerate(time_points):
        pose, velocity, acceleration = traj.evaluate(t)
        positions[i] = pose.translation
        velocities[i] = velocity
        accelerations[i] = acceleration
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Position plot
    axs[0].plot(time_points, positions[:, 0], 'r-', label='X')
    axs[0].plot(time_points, positions[:, 1], 'g-', label='Y')
    axs[0].plot(time_points, positions[:, 2], 'b-', label='Z')
    axs[0].set_title('Position')
    axs[0].set_ylabel('Position (m)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Velocity plot
    axs[1].plot(time_points, velocities[:, 0], 'r-', label='X')
    axs[1].plot(time_points, velocities[:, 1], 'g-', label='Y')
    axs[1].plot(time_points, velocities[:, 2], 'b-', label='Z')
    axs[1].set_title('Velocity')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Acceleration plot
    axs[2].plot(time_points, accelerations[:, 0], 'r-', label='X')
    axs[2].plot(time_points, accelerations[:, 1], 'g-', label='Y')
    axs[2].plot(time_points, accelerations[:, 2], 'b-', label='Z')
    axs[2].set_title('Acceleration')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration (m/s²)')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()