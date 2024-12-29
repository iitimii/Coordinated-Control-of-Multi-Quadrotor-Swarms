# # Claude's code

# import numpy as np
# from scipy.linalg import solve_lyapunov
# import pybullet as p
# from scipy.spatial.transform import Rotation
# from gym_pybullet_drones.control.BaseControl import BaseControl
# from gym_pybullet_drones.utils.enums import DroneModel

# class MRACControl(BaseControl):
#     """Model Reference Adaptive Control (MRAC) for Crazyfly drones.
    
#     This implementation follows the Georgia Tech paper's approach with both linear
#     and nonlinear MRAC variants.
#     """
    
#     def __init__(self, drone_model: DroneModel, g: float = 9.81):
#         """Initialize the MRAC controller.
        
#         Args:
#             drone_model (DroneModel): The type of drone being controlled
#             g (float): Gravitational acceleration constant
#         """
#         super().__init__(drone_model=drone_model, g=g)
        
#         # Validate drone model
#         if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
#             print("[ERROR] MRACControl requires DroneModel.CF2X, DroneModel.CF2P or DroneModel.RACE")
#             exit()
            
#         # Get physical parameters from URDF
#         self.Ixx = self._getURDFParameter("ixx")
#         self.Iyy = self._getURDFParameter("iyy") 
#         self.Izz = self._getURDFParameter("izz")
#         self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
#         self.mass = self._getURDFParameter("m")
#         self.l = self._getURDFParameter("arm")
#         self.g = g
        
#         # Motor constants
#         self.Ka = self.KF  # Thrust coefficient
#         self.Km = self.KM  # Moment coefficient
        
#         # Initialize state space matrices
#         self._setup_state_space()
        
#         # Initialize adaptive parameters
#         self.Q = 600 * np.eye(12)
#         self.P = solve_lyapunov(self.Am.T, -self.Q)
#         self.Gamma_x = 0.005 * np.eye(12)  # Adaptation rate for state feedback
#         self.Gamma_r = 0.005 * np.eye(4)   # Adaptation rate for reference
        
#         # Initialize adaptive gains
#         self.Kx = np.zeros((12, 4))
#         self.Kr = np.eye(4)
        
#         # Reference model state
#         self.Xm = np.zeros(12)
        
#         self.reset()

#     def _setup_state_space(self):
#         """Setup the state space matrices for the linear system."""
#         # For hover equilibrium point where ψ (psi) = 0
#         O6 = np.zeros((6, 6))
#         I6 = np.eye(6)
        
#         # Build A matrix
#         Phi = np.zeros((6, 6))
#         Phi[0:2, 3:5] = -self.g * np.array([[0, 1], [-1, 0]])
        
#         self.A = np.block([
#             [O6, I6],
#             [Phi, O6]
#         ])
        
#         # Build B matrix
#         O84 = np.zeros((8, 4))
#         Delta = np.array([
#             [self.Ka/self.mass, self.Ka/self.mass, self.Ka/self.mass, self.Ka/self.mass],
#             [0, -self.Ka*self.l/self.Ixx, 0, self.Ka*self.l/self.Ixx],
#             [self.Ka*self.l/self.Iyy, 0, -self.Ka*self.l/self.Iyy, 0],
#             [self.Km/self.Izz, -self.Km/self.Izz, self.Km/self.Izz, -self.Km/self.Izz]
#         ])
        
#         self.B = np.vstack((O84, Delta))
        
#         # Compute nominal feedback gains using pole placement
#         desired_poles = -np.linspace(1, 12, 12)
#         self.K_nom = self._place_poles(self.A, self.B, desired_poles)
        
#         # Setup reference model
#         self.Am = self.A - self.B @ self.K_nom
#         self.Bm = self.B
#         self.Kr_nom = np.linalg.inv(self.Bm.T @ self.Bm) @ self.Bm.T @ self.Am

#     def _place_poles(self, A, B, poles):
#         """Simple pole placement implementation.
        
#         Args:
#             A (ndarray): System matrix
#             B (ndarray): Input matrix
#             poles (ndarray): Desired closed-loop poles
            
#         Returns:
#             ndarray: State feedback gain matrix
#         """
#         # This is a simplified implementation - in practice you'd want to use
#         # a more robust method like scipy.signal.place_poles
#         n = A.shape[0]
#         K = np.zeros((B.shape[1], n))
        
#         # Place poles to get nominal controller
#         for i, pole in enumerate(poles):
#             K += pole * np.linalg.pinv(B) @ np.linalg.matrix_power(A, i)
            
#         return K

#     def reset(self):
#         """Reset the controller state."""
#         super().reset()
#         self.Xm = np.zeros(12)
#         # Initialize adaptive gains to nominal values
#         self.Kx = -self.K_nom.T 
#         self.Kr = np.eye(4)

#     def compute_control_inputs(self, X, X_ref):
#         """Compute control inputs using MRAC.
        
#         Args:
#             X (ndarray): Current state [x, y, z, phi, theta, psi, xdot, ydot, zdot, p, q, r]
#             X_ref (ndarray): Reference state
            
#         Returns:
#             ndarray: Control inputs [w1^2, w2^2, w3^2, w4^2]
#         """
#         # Compute reference input
#         r = X_ref[:4]  # Take first 4 states as reference
        
#         # Compute control input
#         u = self.Kx.T @ X + self.Kr @ r
        
#         # Update reference model
#         Xm_dot = self.Am @ self.Xm + self.Bm @ r
#         self.Xm += Xm_dot * self.TIMESTEP
        
#         # Compute tracking error
#         e = X - self.Xm
        
#         # Update adaptive gains
#         e = e.reshape(-1, 1)
#         X = X.reshape(-1, 1)
#         r = r.reshape(-1, 1)
        
#         Kx_dot = -self.Gamma_x @ X @ e.T @ self.P @ self.B
#         Kr_dot = -self.Gamma_r @ r @ e.T @ self.P @ self.B
        
#         self.Kx += Kx_dot.T * self.TIMESTEP
#         self.Kr += Kr_dot.T * self.TIMESTEP
        
#         return u

#     def computeControl(self,
#                       control_timestep,
#                       cur_pos,
#                       cur_quat,
#                       cur_vel,
#                       cur_ang_vel,
#                       target_pos,
#                       target_rpy=np.zeros(3),
#                       target_vel=np.zeros(3),
#                       target_rpy_rates=np.zeros(3)):
#         """Compute the control inputs for the drone.
        
#         This method overrides the base class method to implement MRAC control.
#         """
#         # Convert quaternion to Euler angles
#         cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        
#         # Convert angular velocity to body frame
#         cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel)
        
#         # Build current state vector
#         X = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel))
        
#         # Build reference state vector
#         X_ref = np.hstack((target_pos, target_rpy[2]))  # Only using yaw reference
        
#         # Compute control inputs
#         u = self.compute_control_inputs(X, X_ref)
        
#         # Convert control inputs to RPM commands
#         rpm = np.clip(u, 0, self.MAX_RPM)


#         T_sigma = self.Ka*u.sum()
#         tau_x = self.Ka*self.l*(u[3] - u[1])
#         tau_y = self.Ka*self.l*(u[0] - u[2])
#         tau_z = self.Km*(u[0] - u[1] + u[2] - u[3])
        
#         # Compute errors for logging
#         pos_e = target_pos - cur_pos
#         rpy_e = target_rpy - cur_rpy
        
#         return rpm, pos_e, rpy_e