import math
import numpy as np
import pybullet as pyb
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_lyapunov

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import control as ct


class NonlinearMRAC(BaseControl):
    """Nonlinear Model Reference Adaptive Controller class for Crazyflies."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] NonlinearMRAC requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
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
            
        self.lambda_param = 1

        self.reference_model()

        self.reset()

    def reference_model(self):
        Ka = self.Ka
        Km = self.Km
        g = self.g
        m = self.mass
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        l = self.l
        

        A_ref = np.block([[np.zeros((3,3)) , np.eye(3)], 
                          [np.zeros((3,3)) , np.zeros((3,3))]])
        
        b_sub = np.array([[-self.lambda_param*Ka*l/Iyy, 0, self.lambda_param*Ka*l/Iyy, 0],
                          [0, -self.lambda_param*Ka*l/Ixx, 0, self.lambda_param*Ka*l/Ixx],
                          [-Ka/m , -Ka/m, -Ka/m, -Ka/m]])
        
        B_ref = np.block([[np.zeros((3,4))],
                          [b_sub]])
        
        
        # Q = 30*np.eye(6)
        # R = 1000*np.eye(4)
        # K_ref, self.P, E = ct.lqr(A_ref, B_ref, Q, R)

        desired_poles = [-1, -2, -3, -4, -5, -6]
        K_ref = ct.place(A_ref, B_ref, desired_poles)

        self.Kr_nonlin_ctr = np.linalg.pinv(B_ref) @ (A_ref-B_ref@K_ref)

        self.Am = A_ref - B_ref@K_ref
        self.Bm = B_ref

        Q = 30*np.eye(6)
        self.P = solve_lyapunov(self.Am.T, -Q)

        # Adaptation rates
        self.gamma_x = np.eye(6) * 50
        self.gamma_r = np.eye(4) * 50
        self.gamma_alpha = np.eye(6) * 50
    
        temp_K = -K_ref.T

        temp1 = np.array([[-Ka/m , -Ka/m , -Ka/m, -Ka/m],
                          [0 , -self.lambda_param*Ka*l/Ixx , 0 , self.lambda_param*Ka*l/Ixx],
                          [-self.lambda_param*Ka*l/Iyy , 0 , self.lambda_param*Ka*l/Iyy , 0]])
        
        temp2 = np.array([[self.lambda_param , g , 0 , 0 , 0 , 0],
                [0 , 0 , self.lambda_param*(-Ixx+Iyy-Izz)/Ixx , g , 0 , 0],
                [0 , 0 , 0 , 0 , -self.lambda_param*(-Ixx+Iyy+Izz)/Iyy , -g]])
        
        self.Kalpha = (np.linalg.pinv(temp1) @ temp2).T

        self.Kx = temp_K
        self.Kr = np.eye(4)

    def transform_state(self, X):
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = X
       #1, 2, 3,   4,     5,   6,     7,     8,    9, 10, 11, 12

        x_prime = x + self.lambda_param * (-np.cos(phi) * np.sin(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi))
        y_prime = y + self.lambda_param * (-np.cos(phi) * np.sin(theta) * np.sin(psi) + np.sin(phi) * np.cos(psi))
        z_prime = z - self.lambda_param * np.cos(phi) * np.cos(theta)
        x_dot_prime = x_dot + self.lambda_param * (np.sin(phi) * np.sin(theta) * np.cos(psi) * p - np.cos(phi) * np.sin(psi) * p - np.cos(theta) * np.cos(psi) * q)
        y_dot_prime = y_dot + self.lambda_param * (np.sin(phi) * np.sin(theta) * np.sin(psi) * p + np.cos(phi) * np.cos(psi) * p - np.cos(theta) * np.sin(psi) * q)
        z_dot_prime = z_dot + self.lambda_param * (np.sin(phi) * np.cos(theta) * p + np.sin(theta) * q)

        return np.array([x_prime, y_prime, z_prime, x_dot_prime, y_dot_prime, z_dot_prime])

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
        
        cur_rpy = np.array(pyb.getEulerFromQuaternion(cur_quat))
        cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame

        if self.control_counter == 0:
            Xm = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(12, 1)
            self.Xm = self.transform_state(Xm)

        self.control_counter += 1

        X = np.hstack((cur_pos, cur_rpy, cur_vel, cur_ang_vel)).reshape(-1, 1)
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = X
        X_prime = self.transform_state(X)

        r_pos = np.hstack([target_pos, target_vel]).reshape(-1, 1) # x, y, z, x_dot, y_dot, z_dot
        r1_prime = r_pos[0] + self.lambda_param * (-np.cos(phi) * np.sin(theta) * np.cos(psi) - np.sin(phi) *np.sin(psi))
        r2_prime = r_pos[1] + self.lambda_param * (-np.cos(phi) * np.sin(theta) * np.sin(psi) + np.sin(phi) *np.cos(psi))
        r3_prime = r_pos[2] - self.lambda_param * np.cos(phi) * np.cos(theta)
        r7_prime = r_pos[3] + self.lambda_param * (np.sin(phi) * np.sin(theta) * np.cos(psi) * p - np.cos(phi) * np.sin(psi)*p - np.cos(theta)*np.cos(psi)*q)
        r8_prime = r_pos[4] + self.lambda_param * (np.sin(phi) * np.sin(theta) * np.sin(psi) * p + np.cos(phi) * np.cos(psi)*p - np.cos(theta)*np.sin(psi)*q)
        r9_prime = r_pos[5] + self.lambda_param * (np.sin(phi) * np.cos(theta) * p + np.sin(theta) * q)
        r_prime = np.array([r1_prime, r2_prime, r3_prime, r7_prime, r8_prime, r9_prime]).reshape(-1, 1)
        rt_prime = -self.Kr_nonlin_ctr @ r_prime

        phi_x = np.array([[p*p + q*q],
                        [np.cos(phi) * np.cos(theta)],
                        [q * r],
                        [np.sin(phi) * np.cos(theta)],
                        [p * r],
                        [np.sin(theta)]]).reshape(-1, 1)
        
        # self.Xm = np.copy(r_pos)
        e_prime = X_prime - self.Xm
        
        # The B_prime matrix for the plant in the transformed coords
        R = np.array([[np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi) , np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi) , np.cos(theta)*np.cos(psi)],
            [np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(theta)*np.sin(psi)],
            [np.cos(phi)*np.cos(theta) , np.sin(phi)*np.cos(theta) , -np.sin(theta)]])
        
        Ra = R[0,0] 
        Rb = R[0,1] 
        Rc = R[0,2] 
        Rd = R[1,0] 
        Re = R[1,1] 
        Rf = R[1,2] 
        Rg = R[2,0] 
        Rh = R[2,1] 
        Ri = R[2,2]

        Ka = self.Ka
        Km = self.Km
        m = self.mass
        l = self.l
        Iyy = self.Iyy
        Ixx = self.Ixx
        lambda_param = self.lambda_param

        Rstar = np.array([[-Ka*Ra/m - lambda_param*Ka*l*Rc/Iyy , -Ka*Ra/m - lambda_param*Ka*l*Rb/Ixx , -Ka*Ra/m + lambda_param*Ka*l*Rc/Iyy , -Ka*Ra/m + lambda_param*Ka*l*Rb/Ixx],
         [-Ka*Rd/m - lambda_param*Ka*l*Rf/Iyy , -Ka*Rd/m - lambda_param*Ka*l*Re/Ixx , -Ka*Rd/m + lambda_param*Ka*l*Rf/Iyy , -Ka*Rd/m + lambda_param*Ka*l*Re/Ixx],
         [-Ka*Rg/m - lambda_param*Ka*l*Ri/Iyy , -Ka*Rg/m - lambda_param*Ka*l*Rh/Ixx , -Ka*Rg/m + lambda_param*Ka*l*Ri/Iyy , -Ka*Rg/m + lambda_param*Ka*l*Rh/Ixx]]).reshape(3,4)
        
        B_prime = np.vstack([np.zeros((3,4)), Rstar])

        ut = self.Kx.T @ X_prime + self.Kr.T @ rt_prime - self.Kalpha.T @ phi_x

        Kx_dot = -self.gamma_x @ X_prime @ e_prime.T @ self.P @ B_prime
        Kr_dot = -self.gamma_r @ rt_prime @ e_prime.T @ self.P @ B_prime
        Kalpha_dot = self.gamma_alpha @ phi_x @ e_prime.T @ self.P @ B_prime

        self.Kx += Kx_dot * control_timestep
        self.Kr += Kr_dot * control_timestep
        self.Kalpha += Kalpha_dot * control_timestep

        Xm = np.copy(self.Xm)
        Xm_dot = self.Am @ self.Xm + self.Bm @ rt_prime
        self.Xm += Xm_dot*control_timestep

        T = Ka * ut.sum()
        tau_x = Ka*l * (ut[3] - ut[1])
        tau_y = Ka*l * (ut[0] - ut[2])
        tau_z = Km * (ut[0] - ut[1] + ut[2] - ut[3])



        thrust = np.maximum(0, T)
        target_torques = np.hstack((tau_x, tau_y, tau_z))
        target_torques = np.clip(target_torques, -3200, 3200)

        thrust = (math.sqrt(thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm, X_prime, Xm, e_prime
