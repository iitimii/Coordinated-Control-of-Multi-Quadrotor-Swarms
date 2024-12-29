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

        self.K_x, self.K_y, self.K_z, self.K_psi = self._compute_K()
        self.x_error_integral=self.y_error_integral=self.z_error_integral=0.0
        self.phi_error_integral=self.theta_error_integral=self.psi_error_integral=0.0
        self.reset()

    def _compute_K(self):
        # x and theta dynamics
        Ax = np.array([[0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, self.g, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0]])
        Bx = np.array(
            [[0.0],
            [0.0],
            [0.0],
            [1 / self.Ixx]])
        # C_x = np.array([1, 0])
        # A_x_aug = np.block([[A_x, np.zeros((2, 1))], [-C_x, np.zeros((1, 1))]])
        # B_x_aug = np.vstack([B_x, [[0]]])
        Q_x = np.diag([10000, 100, 10, 1])
        R_x = 10
        K_x, _, _ = ct.lqr(Ax, Bx, Q_x, R_x)

        # y and phi dynamics
        Ay = np.array(
            [[0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -self.g, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]])
        By = np.array(
            [[0.0],
            [0.0],
            [0.0],
            [1 / self.Iyy]])
        # C_y = np.array([1, 0])
        # A_y_aug = np.block([[A_y, np.zeros((2, 1))], [-C_y, np.zeros((1, 1))]])
        # B_y_aug = np.vstack([B_y, [[0]]])
        Q_y = np.diag([10, 10, 10, 1])
        R_y = 0.11
        K_y, _, _ = ct.lqr(Ay, By, Q_y, R_y)

        # z dynamics
        A_z = np.array([[0, 1], [0, 0],])
        B_z = np.array([[0], [1/self.mass]])
        # C_z = np.array([1, 0])
        # A_z_aug = np.block([[A_z, np.zeros((2, 1))], [-C_z, np.zeros((1, 1))]])
        # B_z_aug = np.vstack([B_z, [[0]]])
        Q_z = np.diag([1000, 100])
        R_z = 0.11
        K_z, _, _ = ct.lqr(A_z, B_z, Q_z, R_z)

        # psi dynamics
        A_psi = np.array([[0, 1], [0, 0],])
        B_psi = np.array([[0], [1/self.Izz]])
        # C_psi = np.array([1, 0])
        # A_psi_aug = np.block([[A_psi, np.zeros((2, 1))], [-C_psi, np.zeros((1, 1))]])
        # B_psi_aug = np.vstack([B_psi, [[0]]])
        Q_psi = np.diag([10, 1])
        R_psi = 0.11
        K_psi, _, _ = ct.lqr(A_psi, B_psi, Q_psi, R_psi)

        return K_x, K_y, K_z, K_psi

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

        x = np.array([cur_pos[0], cur_vel[0], cur_rpy[1], cur_ang_vel[1]])
        r = np.array([target_pos[0], target_vel[0], target_rpy[1], target_rpy_rates[1]])
        x_error = x - r
        target_My = -np.dot(self.K_x, x_error)
        # target_theta = np.clip(target_theta, -0.5, 0.5)

        x = np.array([cur_pos[1], cur_vel[1], cur_rpy[0], cur_ang_vel[0]])
        r = np.array([target_pos[1], target_vel[1], target_rpy[0], target_rpy_rates[0]])
        y_error = x - r
        target_Mx = -np.dot(self.K_y, y_error)

        x = np.array([cur_pos[2], cur_vel[2]])
        r = np.array([target_pos[2], target_vel[2]])
        z_error = x - r
        target_thrust = -np.dot(self.K_z, z_error)


        x = np.array([cur_rpy[2], cur_ang_vel[2]])
        r = np.array([target_rpy[2], target_rpy_rates[2]])
        psi_error = x - r
        target_Mz = -np.dot(self.K_psi, psi_error)


        target_torques = np.hstack((target_Mx, target_My, target_Mz))
        target_thrust = np.maximum(0, target_thrust)

        thrust = (math.sqrt(target_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        pos_e = target_pos - cur_pos
        rpy_e = target_rpy - cur_rpy

        return rpm, pos_e, rpy_e