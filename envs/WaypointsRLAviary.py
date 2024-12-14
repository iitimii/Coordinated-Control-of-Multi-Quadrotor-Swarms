import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class WaypointsAviary(BaseRLAviary):
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
        

        self.TARGET_WAYPOINTS = np.array([[1.8, 1.2, 0.9], [1.5, -1.7, 0.5], [0.3, 1.9, 1.2], [-2, 0.8, 1.7], [1.8, 1.2, 0.9]])
        self.current_waypoint_idx = 0 
        self.EPISODE_LEN_SEC = 20
        self.MAX_RANGE = 3.0

        
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        current_target = self.TARGET_WAYPOINTS[self.current_waypoint_idx]
        ret = max(0, 2 - np.linalg.norm(current_target - state[0:3])**4) # 2 - x^4
        return ret
    
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        current_target = self.TARGET_WAYPOINTS[self.current_waypoint_idx]

        if np.linalg.norm(current_target - state[0:3]) < .2:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.TARGET_WAYPOINTS):
                return True
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