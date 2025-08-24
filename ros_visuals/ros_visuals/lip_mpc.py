import numpy as np

from pydrake.all import MathematicalProgram, Solve

################################################################################
# Helper fnc
################################################################################

def continious_LIP_dynamics(g, h):
    """returns the static matrices A,B of the continious LIP dynamics
    """
    #>>>>TODO: Compute
    # Linear Inverted Pendulum dynamics: x_dot = A*x + B*u
    # State: [px, vx, py, vy] (position and velocity in x,y directions)
    # Control: [px_zmp, py_zmp] (ZMP position)
    
    omega_sq = g / h  # omega^2 = g/h
    
    # A matrix for 2D LIP (4x4)
    A = np.array([
        [0, 1, 0, 0],        # px_dot = vx
        [omega_sq, 0, 0, 0], # vx_dot = omega^2 * px - omega^2 * px_zmp
        [0, 0, 0, 1],        # py_dot = vy
        [0, 0, omega_sq, 0]  # vy_dot = omega^2 * py - omega^2 * py_zmp
    ])
    
    # B matrix (4x2)
    B = np.array([
        [0, 0],           # px_dot doesn't depend on ZMP
        [-omega_sq, 0],   # vx_dot = -omega^2 * px_zmp
        [0, 0],           # py_dot doesn't depend on ZMP
        [0, -omega_sq]    # vy_dot = -omega^2 * py_zmp
    ])
    
    return A, B

def discrete_LIP_dynamics(g, h, dt):
    """returns the matrices static Ad,Bd of the discretized LIP dynamics
    """
    #>>>>TODO: Compute
    # Get continuous dynamics
    A, B = continious_LIP_dynamics(g, h)
    
    # Simple Euler discretization
    # x[k+1] = x[k] + dt * (A*x[k] + B*u[k])
    # x[k+1] = (I + dt*A)*x[k] + (dt*B)*u[k]
    
    n = A.shape[0]
    I = np.eye(n)
    
    A_d = I + dt * A
    B_d = dt * B
    
    return A_d, B_d

################################################################################
# LIPInterpolator
################################################################################

class LIPInterpolator:
    """Integrates the linear inverted pendulum model using the 
    continous dynamics. To interpolate the solution to hight 
    """
    def __init__(self, x_inital, conf):
        self.conf = conf
        self.dt = conf.dt
        self.x = x_inital.copy()  # Current state [px, vx, py, vy]
        #>>>>TODO: Finish
        self.g = getattr(conf, 'g', 9.81)
        self.h = getattr(conf, 'h', 0.8)
        self.A, self.B = continious_LIP_dynamics(self.g, self.h)
        
    def integrate(self, u):
        #>>>>TODO: integrate with dt
        # Euler integration: x_new = x + dt * x_dot
        # where x_dot = A*x + B*u
        x_dot = self.A @ self.x + self.B @ u
        self.x = self.x + self.dt * x_dot
        return self.x
    
    def comState(self):
        #>>>>TODO: return the center of mass state
        # that is position \in R3, velocity \in R3, acceleration \in R3
        px, vx, py, vy = self.x
        
        # Position in 3D (constant height h)
        c = np.array([px, py, self.h])
        
        # Velocity in 3D
        c_dot = np.array([vx, vy, 0.0])
        
        # Acceleration from LIP dynamics (without ZMP input)
        omega_sq = self.g / self.h
        ax = omega_sq * px
        ay = omega_sq * py
        c_ddot = np.array([ax, ay, 0.0])
        
        return c, c_dot, c_ddot
    
    def dcm(self):
        #>>>>TODO: return the computed dcm
        # Divergent Component of Motion: xi = p + v/omega
        omega = np.sqrt(self.g / self.h)
        px, vx, py, vy = self.x
        
        dcm_x = px + vx / omega
        dcm_y = py + vy / omega
        dcm = np.array([dcm_x, dcm_y, self.h])
        
        return dcm
    
    def zmp(self):
        #>>>>TODO: return the zmp
        # ZMP from current state (assuming no external ZMP input)
        # For LIP: p_zmp = p - a/omega^2, but here we return center position
        px, vx, py, vy = self.x
        zmp = np.array([px, py, 0.0])
        
        return zmp
        
    
################################################################################
# LIPMPC
################################################################################

class LIPMPC:
    def __init__(self, conf):
        self.conf = conf
        self.dt = conf.dt        
        self.no_samples = conf.no_mpc_samples_per_horizon
        
        # Get discrete dynamics
        self.g = getattr(conf, 'g', 9.81)
        self.h = getattr(conf, 'h', 0.8)
        self.A_d, self.B_d = discrete_LIP_dynamics(self.g, self.h, self.dt)
        
        # solution and references over the horizon
        self.X_k = None
        self.U_k = None
        self.ZMP_ref_k = None
        
    def buildSolveOCP(self, x_k, ZMP_ref_k, terminal_idx):
        """build and solve ocp

        Args:
            x_k (_type_): inital mpc state
            ZMP_ref_k (_type_): zmp reference over horizon
            terminal_idx (_type_): index within horizon to apply terminal constraint

        Returns:
            _type_: control
        """
        
        #>>>>TODO: build and solve the ocp
        #>>>>Note: start without terminal constraints
        
        # Check for NaN values in inputs
        if np.any(np.isnan(x_k)):
            print("Warning: NaN detected in initial state x_k")
            return np.zeros(2)
            
        if np.any(np.isnan(ZMP_ref_k)):
            print("Warning: NaN detected in ZMP reference")
            return np.zeros(2)
        
        # Create optimization problem
        prog = MathematicalProgram()
        
        # Problem dimensions
        n_states = self.A_d.shape[0]  # 4
        n_controls = self.B_d.shape[1]  # 2
        N = min(self.no_samples, len(ZMP_ref_k))
        
        if N <= 0:
            print("Warning: Empty ZMP reference")
            return np.zeros(2)
        
        # Decision variables
        X = prog.NewContinuousVariables(N + 1, n_states, "state")
        U = prog.NewContinuousVariables(N, n_controls, "control")
        
        # Initial condition constraint
        for i in range(n_states):
            prog.AddConstraint(X[0, i] == x_k[i])
        
        # Dynamics constraints
        for k in range(N):
            # x-direction dynamics
            prog.AddConstraint(X[k+1, 0] == self.A_d[0,0]*X[k, 0] + self.A_d[0,1]*X[k, 1] + self.B_d[0,0]*U[k, 0])
            prog.AddConstraint(X[k+1, 1] == self.A_d[1,0]*X[k, 0] + self.A_d[1,1]*X[k, 1] + self.B_d[1,0]*U[k, 0])
            # y-direction dynamics
            prog.AddConstraint(X[k+1, 2] == self.A_d[0,0]*X[k, 2] + self.A_d[0,1]*X[k, 3] + self.B_d[0,0]*U[k, 1])
            prog.AddConstraint(X[k+1, 3] == self.A_d[1,0]*X[k, 2] + self.A_d[1,1]*X[k, 3] + self.B_d[1,0]*U[k, 1])
        
        # Cost function - track ZMP reference
        Q_zmp = 500.0  # ZMP tracking weight
        R_u = 2.0       # Control regularization
        
        for k in range(N):
            # Check for valid ZMP reference
            if k < len(ZMP_ref_k) and not np.any(np.isnan(ZMP_ref_k[k])):
                zmp_ref = ZMP_ref_k[k][:2]  # Take only x,y components
                prog.AddCost(Q_zmp * (U[k, 0] - zmp_ref[0])**2)
                prog.AddCost(Q_zmp * (U[k, 1] - zmp_ref[1])**2)
            else:
                # Use zero reference if no valid reference available
                prog.AddCost(Q_zmp * U[k, 0]**2)
                prog.AddCost(Q_zmp * U[k, 1]**2)
            
            # Control regularization
            prog.AddCost(R_u * U[k, 0]**2)
            prog.AddCost(R_u * U[k, 1]**2)
        
        # Solve the optimization problem
        result = Solve(prog)
        
        if result.is_success():
            # Store solution
            X_opt = result.GetSolution(X)
            U_opt = result.GetSolution(U)
            
            # Check for NaN in solution
            if np.any(np.isnan(U_opt[0])):
                print("Warning: NaN detected in optimization solution")
                return np.zeros(n_controls)
            
            return U_opt[0]
        else:
            print("MPC optimization failed!")
            return np.zeros(n_controls)
    
    def lip_mpc_loop(self, k, ZMP_ref, x_k):
        """
        Wrapper for buildSolveOCP, fetches the correct horizon slice.
        """
        horizon = self.no_samples
        # 取当前步长后的参考
        ZMP_ref_k = ZMP_ref[k:k+horizon]
        # 如果参考长度不够，补零
        if len(ZMP_ref_k) < horizon:
            pad = np.zeros((horizon - len(ZMP_ref_k), 3))
            ZMP_ref_k = np.vstack([ZMP_ref_k, pad])
        # 调用OCP
        u_k = self.buildSolveOCP(x_k, ZMP_ref_k, terminal_idx=None)
        return u_k
    
    def generate_zmp_reference(self, plan):
        return generate_zmp_reference(plan, self.conf.no_mpc_samples_per_step)
        

def generate_zmp_reference(foot_steps, no_samples_per_step):
    """generate the zmp reference given a sequence of footsteps
    """

    #>>>>TODO: use the previously footstep type to build the reference 
    # trajectory for the zmp

    if not foot_steps or len(foot_steps) < 2:
        return np.array([])
    
    zmp_ref = []
    for i in range(len(foot_steps) - 1):
        p0 = foot_steps[i].pose.translation
        p1 = foot_steps[i+1].pose.translation
        for k in range(no_samples_per_step):
            alpha = k / no_samples_per_step
            zmp = (1 - alpha) * p0 + alpha * p1
            zmp_ref.append(np.array([zmp[0], zmp[1], 0.0]))
    # 最后一步保持在最后一个脚
    for _ in range(no_samples_per_step):
        zmp_ref.append(np.array([foot_steps[-1].pose.translation[0], foot_steps[-1].pose.translation[1], 0.0]))
    zmp_ref = np.array(zmp_ref)
    return zmp_ref