import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import control as ct


# class LQRControl(BaseControl):
#     """LQR class for Crazyflies"""

#     def __init__(self, drone_model: DroneModel, g: float = 9.8):
#         super().__init__(drone_model=drone_model, g=g)
#         if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
#             print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
#             exit()

#         self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5]) # for cf2x.urdf
#         self.mass = 0.027
#         self.l = 0.0397
#         self.g = g

#         self.K_att, self.K_pos = self._compute_lqr_gains()

#         self.PWM2RPM_SCALE = 0.2685
#         self.PWM2RPM_CONST = 4070.3
#         self.MIN_PWM = 20000
#         self.MAX_PWM = 65535
#         self.MIXER_MATRIX = np.array([ 
#                                     [-.5, -.5, -1],
#                                     [-.5,  .5,  1],
#                                     [.5, .5, -1],
#                                     [.5, -.5,  1]
#                                     ])

#         self.reset()

#     def _compute_lqr_gains(self):
#         # Attitude subsystem matrices
#         A_att = np.block([
#             [np.zeros((3, 3)), np.eye(3)],
#             [np.zeros((3, 3)), np.zeros((3, 3))]
#         ])

#         B_att = np.block([
#             [np.zeros((3, 3))],
#             [np.eye(3)]
#         ])
        
#         # Position subsystem matrices
#         A_pos = np.block([
#             [np.zeros((3, 3)), np.eye(3)],
#             [np.zeros((3, 3)), np.zeros((3, 3))]
#         ])

#         B_pos = np.block([
#             [np.zeros((3, 3))],
#             [np.eye(3)]
#         ])
        
#         # Cost matrices for LQR
#         Q_att = np.eye(6)  # State cost matrix
#         R_att = np.eye(3)  # Control input cost matrix
#         Q_pos = np.eye(6)
#         R_pos = np.eye(3)

#         K_att, _, _ = ct.lqr(A_att, B_att, Q_att, R_att)
#         K_pos, _, _ = ct.lqr(A_pos, B_pos, Q_pos, R_pos)

#         return K_att, K_pos
        
#     def reset(self):
#         return super().reset()
    
#     def computeControlFromState(self,
#                                 control_timestep,
#                                 state,
#                                 target_pos,
#                                 target_yaw=np.zeros(1),
#                                 target_vel=np.zeros(3),
#                                 target_yaw_rate=np.zeros(1)
#                                 ):
#         """Interface method using `computeControl`."""
#         return self.computeControl(control_timestep=control_timestep,
#                                    cur_pos=state[0:3],
#                                    cur_quat=state[3:7],
#                                    cur_vel=state[10:13],
#                                    cur_ang_vel=state[13:16],
#                                    target_pos=target_pos,
#                                    target_yaw=target_yaw,
#                                    target_vel=target_vel,
#                                    target_yaw_rate=target_yaw_rate
#                                    )
    
#     def computeControl(self,
#                        control_timestep,
#                        cur_pos,
#                        cur_quat,
#                        cur_vel,
#                        cur_ang_vel,
#                        target_pos,
#                        target_yaw=np.zeros(0),
#                        target_vel=np.zeros(3),
#                        target_yaw_rate=np.zeros(0)
#                        ):
        
#         self.control_counter += 1

#         # Position control to determine thrust and target attitude
#         total_thrust, target_quat = self._position_control(
#             control_timestep, cur_pos, cur_quat, cur_vel, 
#             cur_ang_vel, target_pos, target_yaw, 
#             target_vel, target_yaw_rate
#         )

#         # Attitude control to determine torques
#         target_torques = self._attitude_control(
#             control_timestep, cur_quat, cur_ang_vel, 
#             target_quat, np.zeros(3)  # Assuming zero target angular velocity
#         )

#         # Combine thrust and torques into motor commands
#         pwm = total_thrust + np.dot(self.MIXER_MATRIX, target_torques)
#         pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
#         rpm =  self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

#         return rpm
    
#     def _position_control(self,
#                           control_timestep,
#                           cur_pos,
#                           cur_quat,
#                           cur_vel,
#                           cur_ang_vel,
#                           target_pos,
#                           target_yaw,
#                           target_vel,
#                           target_yaw_rate
#                           ):
        
#         # Create state vector for position subsystem
#         x_pos = np.concatenate([cur_pos, cur_vel])
#         x_pos_des = np.concatenate([target_pos, target_vel])

#         # Compute control input using LQR gain
#         u_pos = -self.K_pos @ (x_pos - x_pos_des)

#         # Total thrust computation (based on paper's equation)
#         total_thrust = self.mass * (np.linalg.norm(u_pos) + self.g)

#         # Compute target quaternion 
#         # Follows the paper's approach of creating a rotation that aligns thrust with desired direction
#         b = np.array([0, 0, 1])  # Body-fixed z-axis 
#         u_thrust = u_pos / np.linalg.norm(u_pos)
        
#         # Compute rotation between current z-axis and desired thrust direction
#         v = np.cross(b, u_thrust)
#         s = np.linalg.norm(v)
#         c = np.dot(b, u_thrust)
        
#         # Quaternion from axis-angle representation
#         target_quat = np.array([
#             1 + c,  # Scalar part
#             v[0],   # Vector part x
#             v[1],   # Vector part y
#             v[2]    # Vector part z
#         ])
        
#         # Normalize the quaternion
#         target_quat /= np.linalg.norm(target_quat)
        
#         return total_thrust, target_quat
    
#     def _attitude_control(self,
#                           control_timestep,
#                           cur_quat,
#                           cur_ang_vel,
#                           target_quat,
#                           target_ang_vel):
        
#         # Compute quaternion error
#         q_error = self._quaternion_error(cur_quat, target_quat)

#         # Compute attitude control input
#         # Similar to the attitude control law in the paper
#         x_att = np.concatenate([q_error, cur_ang_vel])
#         x_att_des = np.zeros_like(x_att)  # x_att_des = np.concatenate([np.zeros(3), target_yaw_rate])
        
#         target_torques = -self.K_att @ (x_att - x_att_des)
        
#         return target_torques
    
#     def _quaternion_error(self, q1, q2):
#         """
#         Compute quaternion error as in the paper
        
#         Args:
#             q1 (np.ndarray): First quaternion
#             q2 (np.ndarray): Second quaternion
        
#         Returns:
#             np.ndarray: Quaternion error
#         """
#         # Quaternion error as defined in the paper
#         return np.array([q1[0]*q2[0] + np.dot(q1[1:], q2[1:]),
#                          q1[0]*q2[1:] - q2[0]*q1[1:] + np.cross(q1[1:], q2[1:])])
    

    

    

# GPT 40
class LQRControl(BaseControl):
    """LQR Controller class for Crazyflies."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])  # Inertia matrix for CF2X
        self.mass = 0.027
        self.l = 0.0397
        self.g = g

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])

        self.K_att, self.K_pos = self._compute_lqr_gains()

        self.reset()

    def _compute_lqr_gains(self):
        # Attitude control system matrices
        A_att = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        B_att = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])

        # Position control system matrices
        A_pos = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        B_pos = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])

        # Cost matrices
        Q_att = np.eye(6) * 1.0  # Adjust weights for attitude stabilization
        R_att = np.eye(3) * 0.1
        Q_pos = np.eye(6) * 1.0  # Adjust weights for position stabilization
        R_pos = np.eye(3) * 0.1

        # Calculate LQR gains
        K_att, _, _ = ct.lqr(A_att, B_att, Q_att, R_att)
        K_pos, _, _ = ct.lqr(A_pos, B_pos, Q_pos, R_pos)

        return K_att, K_pos

    def reset(self):
        self.control_counter = 0

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_yaw, target_vel, target_yaw_rate):
        """Compute control outputs."""
        thrust, target_quat = self._position_control(control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_yaw, target_vel, target_yaw_rate)
        torques = self._attitude_control(control_timestep, cur_quat, cur_ang_vel, target_quat)

        pwm = thrust + np.dot(self.MIXER_MATRIX, torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm =  self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm
    
    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_yaw=np.zeros(1),
                                target_vel=np.zeros(3),
                                target_yaw_rate=np.zeros(1)
                                ):
        
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_yaw=target_yaw,
                                   target_vel=target_vel,
                                   target_yaw_rate=target_yaw_rate
                                   )

    def _position_control(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_yaw, target_vel, target_yaw_rate=0):
        x_pos = np.concatenate([cur_pos, cur_vel])
        x_pos_des = np.concatenate([target_pos, target_vel])
        u_pos = -self.K_pos @ (x_pos - x_pos_des)

        target_quat = p.getQuaternionFromEuler((u_pos[1], u_pos[2], target_yaw))

        return u_pos[0], target_quat

    def _attitude_control(self, control_timestep, cur_quat, cur_ang_vel, target_quat, target_ang_vel=0):
        cur_theta = self._quaternion_ln(cur_quat)
        target_theta = self._quaternion_ln(target_quat)

        x_att = np.concatenate([cur_theta, cur_ang_vel])
        x_att_des = np.concatenate([target_theta, cur_ang_vel])
        u_att = -self.K_att @ (x_att - x_att_des)
        return u_att

    def _quaternion_product(q1, q2):
        q1 = np.asarray(q1)
        q2 = np.asarray(q2)
        q1 = q1.reshape(4)
        q2 = q2.reshape(4)

        p = q1
        q = q2

        return np.array([p[0]*q[0] - p[1]*q[1]- p[2]*q[2]- p[3]*q[3],
                         p[0]*q[1] + p[1]*q[0]+ p[2]*q[3]- p[3]*q[2],
                         p[0]*q[2] - p[1]*q[3]+ p[2]*q[0]+ p[3]*q[1],
                         p[0]*q[3] + p[1]*q[2]- p[2]*q[1]+ p[3]*q[0]], dtype=np.float64).reshape(4)
    
    def _quaternion_conjugate(self, q):
        q = np.asarray(q)
        q = q.reshape(4)
        return np.array([q[0], -q[1], -q[2], -q[3]]).reshape(4)
    
    def _quaternion_ln(self, q):
        q = np.asarray(q)
        q = q.reshape(4)

        q_norm = np.linalg.norm(q)
        q_complex_norm = np.linalg.norm(q[1:])

        if q_complex_norm < 1e-8:
            return np.log(q_norm)

        theta = np.arccos(np.clip(q[0] / q_norm, -1.0, 1.0))

        ln_q = np.log(q_norm) + (q[1:] / q_complex_norm) * theta

        return ln_q
    
    def quaternion_rotate_vector(self, q, v):
        q = np.asarray(q)
        v = np.asarray(v)
        q = q.reshape(4)
        v = v.reshape(3)
        q = q / np.linalg.norm(q)

        v_q = np.concatenate([[0], v])

        # Compute q* ⊗ v ⊗ q
        q_conj = self._quaternion_conjugate(q)
        temp = self._quaternion_product(q_conj, v_q)
        v_rotated_q = self._quaternion_product(temp, q)

        v_rotated = v_rotated_q[1:]
        return v_rotated
    
    def _get_target_quat(self, desired_direction, target_yaw):
        pass

