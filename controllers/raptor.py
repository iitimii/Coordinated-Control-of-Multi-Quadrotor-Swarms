import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_lyapunov

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

import control as ct

from foundation_policy import Raptor


class RAPTORCtrl(BaseControl):
    """Raptor Foundation Model Controller class for Crazyflies.
        Original Implementation at: https://github.com/rl-tools/raptor"""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] RAPTOR requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
            exit()

        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))

        self.prev_action = np.zeros((4))
        self.policy = Raptor()
        self.policy.reset()
        self.reset()

    
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
        
        position = cur_pos - target_pos
        cur_orientation = np.array(p.getMatrixFromQuaternion(cur_quat)).flatten()
        cur_ang_vel = Rotation.from_quat(cur_quat).inv().apply(cur_ang_vel) # Convert angular velocity to body frame
        action = self.prev_action

        observation = np.hstack((position, cur_orientation, cur_vel, cur_ang_vel, action)).reshape(1, -1)
        action = self.policy.evaluate_step(observation)[0]

        self.prev_action = action #[front-right, back-right, back-left, front-left]
        
        rpm = np.array([action[0], action[1], action[2], action[3]])
        rpm = np.array(self.HOVER_RPM * (1+0.05*rpm))

        print(f"RPM: {rpm}")

        return rpm
