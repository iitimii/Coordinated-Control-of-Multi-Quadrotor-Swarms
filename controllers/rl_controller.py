# import math
# import numpy as np
# import pybullet as p
# from scipy.spatial.transform import Rotation
# from gym_pybullet_drones.control.BaseControl import BaseControl
# from gym_pybullet_drones.utils.enums import DroneModel
# import control as ct


# class RLControl(BaseControl):
#     """Model Based RL Controller class for Crazyflies."""

#     def __init__(self, drone_model: DroneModel, g: float = 9.8):
#         super().__init__(drone_model=drone_model, g=g)
#         if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
#             print("[ERROR] MRAC requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
#             exit()
#         self.Ixx = self._getURDFParameter("ixx")
#         self.Iyy = self._getURDFParameter("iyy")
#         self.Izz = self._getURDFParameter("izz")
#         self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
#         self.mass = self._getURDFParameter("m")
#         self.l = self._getURDFParameter("arm")
#         self.g = g
#         self.PWM2RPM_SCALE = 0.2685
#         self.PWM2RPM_CONST = 4070.3
#         self.MIN_PWM = 20000
#         self.MAX_PWM = 65535
#         self.Ka = self.KF
#         self.Km = self.KM

#         if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
#             self.MIXER_MATRIX = np.array([ 
#                                     [-.5, -.5, -1],
#                                     [-.5,  .5,  1],
#                                     [.5, .5, -1],
#                                     [.5, -.5,  1]
#                                     ])
#         elif self.DRONE_MODEL == DroneModel.CF2P:
#             self.MIXER_MATRIX = np.array([
#                                     [0, -1,  -1],
#                                     [+1, 0, 1],
#                                     [0,  1,  -1],
#                                     [-1, 0, 1]
#                                     ])
  
#         self.reset()


#     def reset(self):
#         super().reset()

#     def forward(self):
#         pass

#     def backward(self):
#         pass

#     def computeControl(self,
#                        control_timestep,
#                        cur_pos,
#                        cur_quat,
#                        cur_vel,
#                        cur_ang_vel,
#                        target_pos,
#                        target_rpy=np.zeros(3),
#                        target_vel=np.zeros(3),
#                        target_rpy_rates=np.zeros(3)):
        
#         cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
#         cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame

#         action = self.forward()
#         self.backward()

#         thrust, tx, ty, tz = action
#         thrust = np.maximum(0, thrust)
#         target_torques = np.hstack((tx, ty, tz))
#         target_torques = np.clip(target_torques, -3200, 3200)

#         thrust = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
#         pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
#         pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
#         rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

#         return rpm

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
import control as ct

class RLControl(BaseControl):
    """Model Based RL Controller class for Crazyflies."""
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] MRAC requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
            exit()
        self.Ixx = self._getURDFParameter("ixx")
        self.Iyy = self._getURDFParameter("iyy")
        self.Izz = self._getURDFParameter("izz")
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.mass = self._getURDFParameter("m")
        self.l = self._getURDFParameter("arm")
        self.g = g
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.Ka = self.KF
        self.Km = self.KM
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        
        # Initialize model parameters
        self.init_model_parameters()
        
        # Learning rate for model updates
        self.model_lr = 0.01
        self.policy_lr = 0.005
        
        # Memory buffer for model learning
        self.buffer_size = 1000
        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        
        # Current and previous states for dynamics learning
        self.prev_state = None
        self.prev_action = None
        self.current_state = None
        
        self.reset()

    def init_model_parameters(self):
        """Initialize model parameters for the dynamics model."""
        # Initial model parameters - these will be learned
        # Using linear dynamics model: s_{t+1} = A*s_t + B*a_t + c
        
        # State dimensions: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        state_dim = 12
        # Action dimensions: [thrust, tx, ty, tz]
        action_dim = 4
        
        # Initialize dynamics matrices with reasonable values
        # A: state transition matrix
        self.A = np.eye(state_dim)  # Start with identity
        # Add basic physics to state transition (position += velocity * dt)
        dt = 0.01  # Assuming a nominal timestep of 10ms
        self.A[0:3, 3:6] = np.eye(3) * dt
        
        # B: control matrix (effect of actions on state)
        self.B = np.zeros((state_dim, action_dim))
        # Thrust effect on vertical acceleration
        self.B[5, 0] = 1.0 / self.mass  
        # Torque effects on angular acceleration
        self.B[9, 1] = 1.0 / self.Ixx   # roll
        self.B[10, 2] = 1.0 / self.Iyy  # pitch
        self.B[11, 3] = 1.0 / self.Izz  # yaw
        
        # c: constant offset (e.g., gravity)
        self.c = np.zeros(state_dim)
        self.c[5] = -self.g  # gravity affects z-acceleration
        
        # Initialize policy parameters (LQR-based)
        self.Q = np.eye(state_dim)  # State cost matrix
        self.R = np.eye(action_dim)  # Action cost matrix
        
        # Heighten position and attitude costs
        self.Q[0:3, 0:3] *= 10.0  # Position cost
        self.Q[6:9, 6:9] *= 5.0   # Attitude cost
        
        # Initialize optimal gain K (will be computed in forward)
        self.K = None
        
        # Model uncertainty estimates
        self.A_var = np.ones_like(self.A) * 0.1
        self.B_var = np.ones_like(self.B) * 0.1
        self.c_var = np.ones_like(self.c) * 0.1

    def state_to_vector(self, pos, rpy, vel, ang_vel):
        """Convert state components to state vector."""
        return np.concatenate([pos, vel, rpy, ang_vel])
    
    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_action = None
        self.current_state = None
        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []

    def forward(self):
        """
        Forward pass: use the learned model to compute optimal control actions.
        This implements a model-based policy using LQR control.
        """
        if self.current_state is None:
            # Default safe action if no state available
            return np.array([self.mass * self.g, 0, 0, 0])
        
        # Get current state error (difference from target state)
        target_state = np.zeros(12)  # Initialize target state vector
        target_state[0:3] = self.target_pos
        target_state[3:6] = self.target_vel
        target_state[6:9] = self.target_rpy
        target_state[9:12] = self.target_rpy_rates
        
        state_error = target_state - self.current_state
        
        # Compute LQR gains if not already computed or if model updated
        if self.K is None:
            self._compute_lqr_gain()
        
        # Compute optimal action using LQR policy
        optimal_action = np.dot(self.K, state_error)
        
        # Baseline thrust (hover thrust)
        hover_thrust = self.mass * self.g
        
        # Final action: [thrust, tx, ty, tz]
        action = np.array([
            hover_thrust + optimal_action[0],  # Add delta thrust to hover thrust
            optimal_action[1],                # Roll torque
            optimal_action[2],                # Pitch torque
            optimal_action[3]                 # Yaw torque
        ])
        
        # Store current action for learning
        self.prev_action = action.copy()
        
        return action
    
    def _compute_lqr_gain(self):
        """Compute optimal LQR gain based on current dynamics model."""
        try:
            # Create continuous-time system
            sys = ct.ss(self.A, self.B, np.eye(self.A.shape[0]), np.zeros((self.A.shape[0], self.B.shape[1])))
            
            # Solve Riccati equation
            X, _, _ = ct.lqr(sys, self.Q, self.R)
            
            # Compute optimal gain K
            self.K = np.dot(np.linalg.inv(self.R), np.dot(self.B.T, X))
        except np.linalg.LinAlgError:
            # Fallback to robust default gain if LQR fails
            state_dim = self.A.shape[0]
            action_dim = self.B.shape[1]
            self.K = np.zeros((action_dim, state_dim))
            
            # Basic PD gains for position and attitude
            # Mapping: [x,y,z] -> thrust, [roll,pitch,yaw] -> [tx,ty,tz]
            self.K[0, 0:3] = np.array([0.5, 0.5, 5.0])  # Position -> thrust
            self.K[1:4, 6:9] = np.diag([10.0, 10.0, 2.0])  # Attitude -> torques
            
            # Add derivative gains
            self.K[0, 3:6] = np.array([0.1, 0.1, 1.0])  # Velocity -> thrust
            self.K[1:4, 9:12] = np.diag([0.5, 0.5, 0.5])  # Angular velocity -> torques

    def backward(self):
        """
        Backward pass: update the dynamics model based on observed transitions.
        This implements the model learning aspect of MBRL.
        """
        if self.prev_state is not None and self.prev_action is not None and self.current_state is not None:
            # Store transition in buffer
            self.state_buffer.append(self.prev_state.copy())
            self.action_buffer.append(self.prev_action.copy())
            self.next_state_buffer.append(self.current_state.copy())
            
            # Maintain buffer size
            if len(self.state_buffer) > self.buffer_size:
                self.state_buffer.pop(0)
                self.action_buffer.pop(0)
                self.next_state_buffer.pop(0)
            
            # Update model if enough data available
            if len(self.state_buffer) > 10:  # Need some minimum data
                self._update_dynamics_model()
                
                # Recompute LQR gain with updated model
                self.K = None  # This will trigger recomputation in next forward pass
        
        # Update previous state
        self.prev_state = self.current_state.copy() if self.current_state is not None else None

    def _update_dynamics_model(self):
        """Update the dynamics model parameters using the collected data."""
        # Convert buffer to numpy arrays for efficient computation
        states = np.array(self.state_buffer)
        actions = np.array(self.action_buffer)
        next_states = np.array(self.next_state_buffer)
        
        # Simple batch least squares update for linear dynamics
        # Solve for A, B, c in: s_{t+1} = A*s_t + B*a_t + c
        
        # Construct data matrices
        n_samples = states.shape[0]
        state_dim = states.shape[1]
        action_dim = actions.shape[1]
        
        # X = [s_t, a_t, 1] (concatenated)
        X = np.zeros((n_samples, state_dim + action_dim + 1))
        X[:, :state_dim] = states
        X[:, state_dim:state_dim+action_dim] = actions
        X[:, -1] = 1.0  # Constant term
        
        # Y = s_{t+1}
        Y = next_states
        
        # Solve least squares for each state dimension
        for i in range(state_dim):
            # Add regularization to avoid overfitting
            # Solve (X^T X + lambda*I)^{-1} X^T Y
            lambda_reg = 0.1
            reg_matrix = lambda_reg * np.eye(X.shape[1])
            
            try:
                # Try solving the regularized least squares problem
                beta = np.linalg.solve(X.T @ X + reg_matrix, X.T @ Y[:, i])
                
                # Update model parameters with learning rate for stability
                self.A[i, :] = (1 - self.model_lr) * self.A[i, :] + self.model_lr * beta[:state_dim]
                self.B[i, :] = (1 - self.model_lr) * self.B[i, :] + self.model_lr * beta[state_dim:state_dim+action_dim]
                self.c[i] = (1 - self.model_lr) * self.c[i] + self.model_lr * beta[-1]
                
                # Compute prediction errors for uncertainty estimation
                preds = X @ beta
                errors = Y[:, i] - preds
                variance = np.mean(errors**2)
                
                # Update uncertainty estimates
                self.A_var[i, :] = (1 - self.model_lr) * self.A_var[i, :] + self.model_lr * variance
                self.B_var[i, :] = (1 - self.model_lr) * self.B_var[i, :] + self.model_lr * variance
                self.c_var[i] = (1 - self.model_lr) * self.c_var[i] + self.model_lr * variance
                
            except np.linalg.LinAlgError:
                # If solving fails, make a smaller update based on prediction error
                print("Linear solver failed, falling back to gradient update")
                for j in range(n_samples):
                    # Predicted next state
                    pred = np.dot(self.A[i, :], states[j]) + np.dot(self.B[i, :], actions[j]) + self.c[i]
                    # Error
                    error = next_states[j, i] - pred
                    # Gradient updates
                    self.A[i, :] += self.model_lr * error * states[j] / n_samples
                    self.B[i, :] += self.model_lr * error * actions[j] / n_samples
                    self.c[i] += self.model_lr * error / n_samples
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        
        # Get Euler angles from quaternion
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        
        # Convert angular velocity to body frame
        cur_ang_vel_body = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel)
        
        # Store target values for controller use
        self.target_pos = np.array(target_pos)
        self.target_rpy = np.array(target_rpy)
        self.target_vel = np.array(target_vel)
        self.target_rpy_rates = np.array(target_rpy_rates)
        
        # Update current state
        self.current_state = self.state_to_vector(
            np.array(cur_pos),
            cur_rpy,
            np.array(cur_vel),
            cur_ang_vel_body
        )
        
        # Run forward pass to get control action
        action = self.forward()
        
        # Run backward pass to update model
        self.backward()
        
        # Unpack action
        thrust, tx, ty, tz = action
        
        # Ensure positive thrust
        thrust = np.maximum(0, thrust)
        
        # Clip torques to reasonable values
        target_torques = np.array([tx, ty, tz])
        target_torques = np.clip(target_torques, -3200, 3200)
        
        # Convert thrust to PWM
        thrust_pwm = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Apply mixer matrix to get motor commands
        pwm = thrust_pwm + np.dot(self.MIXER_MATRIX, target_torques)
        
        # Clip PWM values
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        
        # Convert PWM to RPM
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        
        return rpm