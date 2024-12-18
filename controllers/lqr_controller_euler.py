import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import control as ct


class LQRControl(BaseControl):
    """LQR Controller class for Crazyflies."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
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

        if self.DRONE_MODEL == DroneModel.CF2X:
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

        self.K_x, self.K_y, self.K_z, self.K_phi, self.K_theta, self.K_psi = self._compute_lqr_gains()
        self.integral_error_x, self.integral_error_y, self.integral_error_z = 0.0, 0.0, 0.0
        self.integral_error_phi, self.integral_error_theta, self.integral_error_psi = 0.0, 0.0, 0.0

        self.reset()

    def _compute_lqr_gains(self):
        A_x = np.array([[0, 1], [0, 0]])
        B_x = np.array([[0], [self.g]])
        C_x = np.array([1, 0])
        A_x_bar = np.block([[A_x, np.zeros((2,1))],  # Integrative action
                            [-C_x, np.zeros((1,1))]])
        B_x_bar = np.vstack((B_x, np.zeros((1,1))))

        A_y = np.array([[0, 1], [0, 0]])
        B_y = np.array([[0], [-self.g]])
        C_y = np.array([1, 0])
        A_y_bar = np.block([[A_y, np.zeros((2,1))], 
                            [-C_y, np.zeros((1,1))]])
        B_y_bar = np.vstack((B_y, np.zeros((1,1))))
        
        A_z = np.array([[0, 1], [0, 0]])
        B_z = np.array([[0], [1/self.mass]])
        C_z = np.array([1, 0])
        A_z_bar = np.block([[A_z, np.zeros((2,1))], 
                            [-C_z, np.zeros((1,1))]])
        B_z_bar = np.vstack((B_z, np.zeros((1,1))))
        
        A_phi = np.array([[0, 1], [0, 0]]) # Roll
        B_phi = np.array([[0], [1/self.Ixx]])
        C_phi = np.array([1, 0])
        A_phi_bar = np.block([[A_phi, np.zeros((2,1))], 
                            [-C_phi, np.zeros((1,1))]])
        B_phi_bar = np.vstack((B_phi, np.zeros((1,1))))

        A_theta = np.array([[0, 1], [0, 0]]) # Pitch
        B_theta = np.array([[0], [1/self.Iyy]])
        C_theta = np.array([1, 0])
        A_theta_bar = np.block([[A_theta, np.zeros((2,1))], 
                            [-C_theta, np.zeros((1,1))]])
        B_theta_bar = np.vstack((B_theta, np.zeros((1,1))))

        A_psi = np.array([[0, 1], [0, 0]]) # Yaw
        B_psi = np.array([[0], [1/self.Izz]])
        C_psi = np.array([1, 0])
        A_psi_bar = np.block([[A_psi, np.zeros((2,1))], 
                            [-C_psi, np.zeros((1,1))]])
        B_psi_bar = np.vstack((B_psi, np.zeros((1,1))))


        Q_x = np.diag([15, 1, 10])
        R_x = 1000

        Q_y = np.diag([15, 1, 10]) 
        R_y = 1000

        Q_z = np.diag([5, 1, 3])
        R_z = 3

        Q_phi = np.diag([25, 1, 1500]) # Roll
        R_phi = 200

        Q_theta = np.diag([25, 1, 1500]) # Pitch
        R_theta = 200

        Q_psi = np.diag([7, 1, 15]) 
        R_psi = 1

        # Calculate LQR gains
        K_x, _, _ = ct.lqr(A_x_bar, B_x_bar, Q_x, R_x)
        K_y, _, _ = ct.lqr(A_y_bar, B_y_bar, Q_y, R_y)
        K_z, _, _ = ct.lqr(A_z_bar, B_z_bar, Q_z, R_z)
        K_phi, _, _ = ct.lqr(A_phi_bar, B_phi_bar, Q_phi, R_phi)
        K_theta, _, _ = ct.lqr(A_theta_bar, B_theta_bar, Q_theta, R_theta)
        K_psi, _, _ = ct.lqr(A_psi_bar, B_psi_bar, Q_psi, R_psi)

        return K_x, K_y, K_z, K_phi, K_theta, K_psi

    def reset(self):
        super().reset()
        self.integral_error_x, self.integral_error_y, self.integral_error_z = 0.0, 0.0, 0.0
        self.integral_error_phi, self.integral_error_theta, self.integral_error_psi = 0.0, 0.0, 0.0

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
        
        self.control_counter += 1

        thrust, computed_target_rpy, pos_e = self._position_control(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        
        target_torques = self._attitude_control(control_timestep,
                                                cur_quat,
                                                cur_ang_vel,
                                                computed_target_rpy,
                                                target_rpy_rates)
        

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        cur_rpy = p.getEulerFromQuaternion(cur_quat)

        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
        

    def _position_control(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel):
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        self.integral_error_x += pos_e[0] * control_timestep
        self.integral_error_y += pos_e[1] * control_timestep
        self.integral_error_z += pos_e[2] * control_timestep

        x_input = np.hstack((pos_e[0], vel_e[0], self.integral_error_x))
        y_input = np.hstack((pos_e[1], vel_e[1], self.integral_error_y))
        z_input = np.hstack((pos_e[2], vel_e[2], self.integral_error_z))


        target_roll = -self.K_y @ y_input
        target_pitch = self.K_x @ x_input
        thrust = self.K_z @ z_input + self.mass * self.g

        target_euler = np.array([target_roll[0], target_pitch[0], target_rpy[2]])
        
        return thrust, target_euler, pos_e
        


    def _attitude_control(self,
                          control_timestep,
                          cur_quat,
                          cur_ang_vel,
                          target_euler,
                          target_rpy_rate):
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rate - cur_rpy/control_timestep

        self.integral_error_phi += rpy_e[0] * control_timestep
        self.integral_error_theta += rpy_e[1] * control_timestep
        self.integral_error_psi += rpy_e[2] * control_timestep

        phi_input = np.hstack((rpy_e[0], rpy_rate_e[0], self.integral_error_phi))
        theta_input = np.hstack((rpy_e[1], rpy_rate_e[1], self.integral_error_theta))
        psi_input = np.hstack((rpy_e[2], rpy_rate_e[2], self.integral_error_psi))


        roll_torque = self.K_phi @ phi_input
        pitch_torque = self.K_theta @ theta_input
        yaw_torque = self.K_psi @ psi_input

        target_torques = np.array([roll_torque[0], pitch_torque[0], yaw_torque[0]])
        
        return target_torques
