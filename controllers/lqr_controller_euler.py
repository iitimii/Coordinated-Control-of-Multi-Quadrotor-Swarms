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
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] LQRControl requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
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

        self.K_x, self.K_y, self.K_z, self.K_phi, self.K_theta, self.K_psi = self._compute_K()
        self.x_error_integral=self.y_error_integral=self.z_error_integral=0.0
        self.phi_error_integral=self.theta_error_integral=self.psi_error_integral=0.0
        self.reset()

    def _compute_K(self):
        # x dynamics
        A_x = np.array([[0, 1], [0, 0],])
        B_x = np.array([[0], [1/self.g]])
        C_x = np.array([1, 0])
        A_x_aug = np.block([[A_x, np.zeros((2, 1))], [-C_x, np.zeros((1, 1))]])
        B_x_aug = np.vstack([B_x, [[0]]])
        Q_x = np.diag([15, 1, 10])
        R_x = 1000
        K_x, _, _ = ct.lqr(A_x_aug, B_x_aug, Q_x, R_x)

        # y dynamics
        A_y = np.array([[0, 1], [0, 0],])
        B_y = np.array([[0], [1/-self.g]])
        C_y = np.array([1, 0])
        A_y_aug = np.block([[A_y, np.zeros((2, 1))], [-C_y, np.zeros((1, 1))]])
        B_y_aug = np.vstack([B_y, [[0]]])
        Q_y = np.diag([15, 1, 10])
        R_y = 1000
        K_y, _, _ = ct.lqr(A_y_aug, B_y_aug, Q_y, R_y)

        # z dynamics
        A_z = np.array([[0, 1], [0, 0],])
        B_z = np.array([[0], [1/self.mass]])
        C_z = np.array([1, 0])
        A_z_aug = np.block([[A_z, np.zeros((2, 1))], [-C_z, np.zeros((1, 1))]])
        B_z_aug = np.vstack([B_z, [[0]]])
        Q_z = np.diag([50, 10, 1])
        R_z = 100
        K_z, _, _ = ct.lqr(A_z_aug, B_z_aug, Q_z, R_z)

        # phi dynamics
        A_phi = np.array([[0, 1], [0, 0],])
        B_phi = np.array([[0], [1/self.Ixx]])
        C_phi = np.array([1, 0])
        A_phi_aug = np.block([[A_phi, np.zeros((2, 1))], [-C_phi, np.zeros((1, 1))]])
        B_phi_aug = np.vstack([B_phi, [[0]]])
        Q_phi = np.diag([25, 1, 1500])
        R_phi = 200
        K_phi, _, _ = ct.lqr(A_phi_aug, B_phi_aug, Q_phi, R_phi)

        # theta dynamics
        A_theta = np.array([[0, 1], [0, 0],])
        B_theta = np.array([[0], [1/self.Iyy]])
        C_theta = np.array([1, 0])
        A_theta_aug = np.block([[A_theta, np.zeros((2, 1))], [-C_theta, np.zeros((1, 1))]])
        B_theta_aug = np.vstack([B_theta, [[0]]])
        Q_theta = np.diag([25, 1, 1500])
        R_theta = 200
        K_theta, _, _ = ct.lqr(A_theta_aug, B_theta_aug, Q_theta, R_theta)

        # psi dynamics
        A_psi = np.array([[0, 1], [0, 0],])
        B_psi = np.array([[0], [1/self.Izz]])
        C_psi = np.array([1, 0])
        A_psi_aug = np.block([[A_psi, np.zeros((2, 1))], [-C_psi, np.zeros((1, 1))]])
        B_psi_aug = np.vstack([B_psi, [[0]]])
        Q_psi = np.diag([7, 1, 15])
        R_psi = 15
        K_psi, _, _ = ct.lqr(A_psi_aug, B_psi_aug, Q_psi, R_psi)

        return K_x, K_y, K_z, K_phi, K_theta, K_psi

    def reset(self):
        super().reset()
        self.phi_error_integral=self.theta_error_integral=self.psi_error_integral=0.0


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

        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        cur_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_vel) # Convert velocity to body frame
        target_vel = Rotation.from_euler('XYZ', target_rpy).inv().apply(target_vel) # Convert velocity to body frame
        cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame

        x = np.array([cur_pos[0], cur_vel[0], self.x_error_integral])
        r = np.array([target_pos[0], target_vel[0], 0])
        x_error = x - r
        target_acc_x = -np.dot(self.K_x, x_error)

        x = np.array([cur_pos[1], cur_vel[1], self.y_error_integral])
        r = np.array([target_pos[1], target_vel[1], 0])
        y_error = x - r
        target_acc_y = -np.dot(self.K_y, y_error)

        x = np.array([cur_pos[2], cur_vel[2], self.z_error_integral])
        r = np.array([target_pos[2], target_vel[2], 0])
        z_error = x - r
        target_thrust = -np.dot(self.K_z, z_error) + self.g*self.mass

        psi = cur_rpy[2]
        target_phi = target_acc_y*np.cos(psi) - target_acc_x*np.sin(psi)
        target_theta = target_acc_x*np.cos(psi) + target_acc_y*np.sin(psi)
        max_angle = 10
        angle_rad = np.radians(max_angle)
        target_phi = np.clip(target_phi, -angle_rad, angle_rad)
        target_theta = np.clip(target_theta, -angle_rad, angle_rad)

        x = np.array([cur_rpy[0], cur_ang_vel[0], self.phi_error_integral])
        r = np.array([target_phi[0], target_rpy_rates[0], 0])
        phi_error = x - r
        target_Mx = -np.dot(self.K_phi, phi_error)

        x = np.array([cur_rpy[1], cur_ang_vel[1], self.theta_error_integral])
        r = np.array([target_theta[0], target_rpy_rates[1], 0])
        theta_error = x - r
        target_My = -np.dot(self.K_theta, theta_error)

        x = np.array([cur_rpy[2], cur_ang_vel[2], self.psi_error_integral])
        r = np.array([target_rpy[2], target_rpy_rates[2], 0])
        psi_error = x - r
        target_Mz = -np.dot(self.K_psi, psi_error)

        thrust = np.maximum(0, target_thrust)
        target_torques = np.hstack((target_Mx, target_My, target_Mz))
        target_torques = np.clip(target_torques, -3200, 3200)

        thrust = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        pos_e = target_pos - cur_pos
        rpy_e = target_rpy - cur_rpy

        # self.x_error_integral -= x_error[0] * control_timestep
        # self.x_error_integral = np.clip(self.x_error_integral, -100., 100.)
        # self.y_error_integral -= y_error[0] * control_timestep
        # self.y_error_integral = np.clip(self.y_error_integral, -100., 100.)
        # self.z_error_integral -= z_error[0] * control_timestep
        # self.z_error_integral = np.clip(self.z_error_integral, -100., 100.)

        # self.phi_error_integral -= phi_error[0] * control_timestep
        # self.phi_error_integral = np.clip(self.phi_error_integral, -100., 100.)
        # self.theta_error_integral -= theta_error[0] * control_timestep
        # self.theta_error_integral = np.clip(self.theta_error_integral, -100., 100.)
        # self.psi_error_integral -= psi_error[0] * control_timestep
        # self.psi_error_integral = np.clip(self.psi_error_integral, -100., 100.)

        print(f"Target Thrust: {target_thrust}, Target Torques: {target_torques}, PWM: {pwm}")
        return rpm, pos_e, rpy_e