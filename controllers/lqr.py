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
            print("[ERROR] LQRControl requires DroneModel.CF2X or DroneModel.CF2P")
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

        self.K = self._compute_K()
        self.reset()

    def _compute_K(self, phi=0,
                        theta=0,
                        psi=0,
                        u=0,
                        v=0,
                        w=0,
                        p=0,
                        q=0,
                        r=0):
        

        A = np.array([[0, 0, 0, v*theta+w*psi, v*phi+w, -v+w*phi, 1, theta*phi-psi, theta+psi*phi, 0, 0, 0], #x
                      [0, 0, 0, v*psi*theta-w, v*psi*phi+w*psi, u+v*theta*phi+w*theta, psi, psi*theta*phi+1, psi*theta-phi, 0, 0, 0], #y
                      [0, 0, 0, v, -u, 0, -theta, phi, 1, 0, 0, 0], #z
                      [0, 0, 0, q*theta, q*phi+r, 0, 0, 0, 0, 1, phi*theta, theta], #phi
                      [0, 0, 0, -r, 0, 0, 0, 0, 0, 0, 1, -phi], #theta
                      [0, 0, 0, q, 0, 0, 0, 0, 0, 0, phi, 1], #psi
                      [0, 0, 0, 0, self.g, 0, 0, r, -q, 0, -w, v], #u
                      [0, 0, 0, -self.g, 0, 0, -r, 0, p, w, 0, -u], #v
                      [0, 0, 0, 0, 0, 0, q, -p, 0, -v, u, 0], #w
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r*(self.Iyy-self.Izz)/self.Ixx, q*(self.Iyy-self.Izz)/self.Ixx], #p
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, r*(self.Izz-self.Ixx)/self.Iyy, 0, p*(self.Izz-self.Ixx)/self.Iyy], #q
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, q*(self.Ixx-self.Iyy)/self.Izz, p*(self.Ixx-self.Iyy)/self.Izz, 0]]) #r
        
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1/self.mass, 0, 0, 0],
                      [0, 1/self.Ixx, 0, 0],
                      [0, 0, 1/self.Iyy, 0],
                      [0, 0, 0, 1/self.Izz]])


        Q = np.diag([1e1, 1e1, 1e2, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        #x, y, z, phi, theta, psi, u, v, w, p, q, r
        R = np.diag([1e-1, 1e3, 1e3, 1e3])
        #Thrust, Mx, My, Mz

        K, _, _ = ct.lqr(A, B, Q, R)
        # K[K < 1e-6] = 0
        # K = [[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], #Mx roll
        #     [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], #My pitch
        #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]] 

        return K

    def reset(self):
        super().reset()

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

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        cur_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_vel) # Convert velocity to body frame
        target_vel = Rotation.from_euler('XYZ', target_rpy).inv().apply(target_vel) # Convert velocity to body frame
        cur_ang_vel = Rotation.from_euler('XYZ', cur_rpy).inv().apply(cur_ang_vel) # Convert angular velocity to body frame

        x = np.array([cur_pos[0], cur_pos[1], cur_pos[2], cur_rpy[0], cur_rpy[1], cur_rpy[2], cur_vel[0], cur_vel[1], cur_vel[2], cur_ang_vel[0], cur_ang_vel[1], cur_ang_vel[2]])
        r = np.array([target_pos[0], target_pos[1], target_pos[2], target_rpy[0], target_rpy[1], target_rpy[2], target_vel[0], target_vel[1], target_vel[2], target_rpy_rates[0], target_rpy_rates[1], target_rpy_rates[2]])

        self.K = self._compute_K(phi=cur_rpy[0], theta=cur_rpy[1], psi=cur_rpy[2],
                                 u=cur_vel[0], v=cur_vel[1], w=cur_vel[2],
                                 p=cur_ang_vel[0], q=cur_ang_vel[1], r=cur_ang_vel[2])

        u = -np.dot(self.K, x - r)
        
        target_thrust = u[0]
        target_torques = np.hstack((u[1], u[2], u[3]))
        target_thrust = np.maximum(0, target_thrust)

        thrust = (math.sqrt(target_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        pos_e = target_pos - cur_pos
        rpy_e = target_rpy - cur_rpy

        return rpm, pos_e, rpy_e