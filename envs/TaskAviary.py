import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TaskAviary(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
        self.EPISODE_LEN_SEC = 20
        self.MAX_RANGE = 3.0
        
    def _addObstacles(self):
        p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        
        
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return 0
    
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return False
        
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)

        if (abs(state[0]) > self.MAX_RANGE or abs(state[1]) > self.MAX_RANGE or state[2] > self.MAX_RANGE
             or abs(state[7]) > .4 or abs(state[8]) > .4
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        return {"answer": 42} # Calculated by the Deep Thought supercomputer in 7.5M years, very funny