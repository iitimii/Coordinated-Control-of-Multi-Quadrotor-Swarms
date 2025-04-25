# import numpy as np
# from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
# from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
# from gymnasium import spaces

# class MultiCoverageAviary(BaseRLAviary):
#     """
#     Multi-agent RL environment for quick area coverage.
    
#     Drones are controlled by a velocity-controlled PID controller.
#     They are rewarded for covering new grid cells in a defined 2D area, and
#     penalized for collisions. The controller is forced to maintain z = 1.
    
#     Additionally, the observation returned to each agent includes the flattened
#     coverage grid, so the agent knows which parts of the area have already been visited.
#     """

#     def __init__(self,
#                  drone_model: DroneModel = DroneModel.CF2X,
#                  num_drones: int = 2,
#                  neighbourhood_radius: float = np.inf,
#                  initial_xyzs=None,
#                  initial_rpys=None,
#                  physics: Physics = Physics.PYB,
#                  pyb_freq: int = 240,
#                  ctrl_freq: int = 30,
#                  gui: bool = False,
#                  record: bool = False,
#                  obs: ObservationType = ObservationType.KIN):
#         """
#         Initializes the multi-agent area coverage environment.
#         Enforces velocity control (ActionType.VEL) so that we can zero out the z-velocity.
#         """
#         # Force velocity control mode
#         act = ActionType.VEL

#         # Define episode length in seconds
#         self.EPISODE_LEN_SEC = 30

#         # Set the area bounds for coverage (e.g., square from -1 to 1 in X and Y)
#         self.area_bounds = np.array([[-1, 1], [-1, 1]])
#         # Grid resolution (in meters)
#         self.grid_resolution = 0.2
#         # Compute grid dimensions for the area
#         self.grid_shape = (
#             int((self.area_bounds[0, 1] - self.area_bounds[0, 0]) / self.grid_resolution),
#             int((self.area_bounds[1, 1] - self.area_bounds[1, 0]) / self.grid_resolution)
#         )
#         # Coverage grid; cell is True if visited, else False
#         self.coverage_grid = np.zeros(self.grid_shape, dtype=bool)

#         # Collision parameters: if drones come closer than threshold, apply penalty.
#         self.collision_dist_threshold = 0.2
#         self.collision_penalty = 5.0

#         # Call the BaseRLAviary initializer.
#         super().__init__(drone_model=drone_model,
#                          num_drones=num_drones,
#                          neighbourhood_radius=neighbourhood_radius,
#                          initial_xyzs=initial_xyzs,
#                          initial_rpys=initial_rpys,
#                          physics=physics,
#                          pyb_freq=pyb_freq,
#                          ctrl_freq=ctrl_freq,
#                          gui=gui,
#                          record=record,
#                          obs=obs,
#                          act=act)

#         # Force initial altitude to be 1.
#         self._setInitialAltitude()

#     def _setInitialAltitude(self):
#         """
#         Ensures all drones start at z = 1.
#         """
#         self.INIT_XYZS[:, 2] = 1.0

#     def _updateCoverage(self):
#         """
#         Updates the coverage grid based on current drone positions.
#         Returns:
#             int: Number of newly covered grid cells this step.
#         """
#         new_cells = 0
#         for i in range(self.NUM_DRONES):
#             state = self._getDroneStateVector(i)
#             x, y = state[0], state[1]
#             # Check if within the defined bounds
#             if (self.area_bounds[0, 0] <= x <= self.area_bounds[0, 1] and
#                 self.area_bounds[1, 0] <= y <= self.area_bounds[1, 1]):
#                 # Compute grid indices (ensuring indices remain valid)
#                 ix = int((x - self.area_bounds[0, 0]) / self.grid_resolution)
#                 iy = int((y - self.area_bounds[1, 0]) / self.grid_resolution)
#                 ix = min(ix, self.grid_shape[0] - 1)
#                 iy = min(iy, self.grid_shape[1] - 1)
#                 # Mark grid cell if not already visited
#                 if not self.coverage_grid[ix, iy]:
#                     self.coverage_grid[ix, iy] = True
#                     new_cells += 1
#         return new_cells

#     def _computeReward(self):
#         """
#         Computes the reward at each step:
#           - Rewards new grid cells covered.
#           - Applies a penalty if any drones come too close.
#         Returns:
#             float: Total reward for the step.
#         """
#         reward = 0.0
#         # Reward new grid coverage
#         new_cells = self._updateCoverage()
#         reward += new_cells

#         # Collision avoidance: penalize if drones are too close.
#         for i in range(self.NUM_DRONES):
#             state_i = self._getDroneStateVector(i)
#             pos_i = state_i[0:3]
#             for j in range(i + 1, self.NUM_DRONES):
#                 state_j = self._getDroneStateVector(j)
#                 pos_j = state_j[0:3]
#                 if np.linalg.norm(pos_i - pos_j) < self.collision_dist_threshold:
#                     reward -= self.collision_penalty
#         return reward

#     def _computeTerminated(self):
#         """
#         Terminates the episode if the entire area is covered.
#         Returns:
#             bool: True if episode is done.
#         """
#         if np.all(self.coverage_grid):
#             return True
#         return False

#     def _computeTruncated(self):
#         """
#         Checks truncation conditions (e.g., out-of-bounds, excessive tilt, or timeout).
#         Returns:
#             bool: True if the episode should be truncated.
#         """
#         for i in range(self.NUM_DRONES):
#             state = self._getDroneStateVector(i)
#             x, y, z = state[0], state[1], state[2]
#             if (abs(x) > 2.0 or abs(y) > 2.0 or z > 2.0):
#                 return True
#             if (abs(state[7]) > 0.4 or abs(state[8]) > 0.4):
#                 return True
#         if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
#             return True
#         return False

#     def _computeInfo(self):
#         """
#         Returns additional info, e.g., area coverage statistics.
#         Returns:
#             dict: Includes area coverage percentage and number of visited cells.
#         """
#         total_cells = self.grid_shape[0] * self.grid_shape[1]
#         area_covered = np.sum(self.coverage_grid) / total_cells
#         return {"area_coverage": area_covered,
#                 "visited_cells": int(np.sum(self.coverage_grid))}

#     def _preprocessAction(self, action):
#         """
#         Forces the z velocity component to zero to maintain altitude = 1.
        
#         Parameters:
#             action (np.ndarray): The action for each drone.
#         Returns:
#             np.ndarray: The RPM commands computed by the parent's controller.
#         """
#         mod_action = np.array(action, copy=True)
#         if mod_action.ndim == 2 and mod_action.shape[1] >= 4:
#             mod_action[:, 2] = 0.0
#         return super()._preprocessAction(mod_action)

#     def step(self, actions):
#         """
#         Executes one simulation step.
#         Preprocesses the velocity commands, steps the simulation, and computes rewards.
        
#         Parameters:
#             actions (np.ndarray): Action commands for all drones.
#         Returns:
#             obs, reward, terminated, truncated, info: Typical Gym step returns.
#         """
#         rpm_commands = self._preprocessAction(actions)
#         obs, reward_parent, terminated, truncated, info = super().step(rpm_commands)
#         custom_reward = self._computeReward()
#         reward = reward_parent + custom_reward
#         return obs, reward, terminated, truncated, info

#     def reset(self):
#         """
#         Resets the environment and clears the coverage grid.
#         Returns:
#             np.ndarray: The initial observation.
#         """
#         self.coverage_grid = np.zeros(self.grid_shape, dtype=bool)
#         return super().reset()

#     def _observationSpace(self):
#         """
#         Extends the base kinematic observation space by concatenating the flattened
#         coverage grid.
        
#         Returns:
#             gymnasium.spaces.Box: The observation space.
#         """
#         base_space = super()._observationSpace()
#         if self.OBS_TYPE == ObservationType.KIN:
#             # Compute the size of the flattened grid.
#             grid_flat_size = self.grid_shape[0] * self.grid_shape[1]
#             # Create lower and upper bounds for grid observations [0,1]
#             grid_low = np.zeros((self.NUM_DRONES, grid_flat_size))
#             grid_high = np.ones((self.NUM_DRONES, grid_flat_size))
#             # Concatenate along the last axis.
#             new_low = np.hstack([base_space.low, grid_low])
#             new_high = np.hstack([base_space.high, grid_high])
#             return spaces.Box(low=new_low, high=new_high, dtype=np.float32)
#         else:
#             return base_space

#     def _computeObs(self):
#         """
#         Computes the current observation by concatenating the base kinematic state 
#         (and action buffer) with the flattened coverage grid.
        
#         Returns:
#             np.ndarray: Observation of shape (NUM_DRONES, base_dim + grid_flat_size).
#         """
#         # Get the base observation (e.g., 12-dim state + action buffer)
#         base_obs = super()._computeObs()
#         if self.OBS_TYPE == ObservationType.KIN:
#             grid_flat = self.coverage_grid.flatten()  # shape: (grid_flat_size,)
#             # Replicate the grid for each drone
#             grid_obs = np.tile(grid_flat, (self.NUM_DRONES, 1))
#             # Concatenate along the feature axis
#             return np.hstack([base_obs, grid_obs]).astype(np.float32)
#         else:
#             return base_obs


# # Example usage:
# if __name__ == "__main__":
#     env = MultiCoverageAviary(num_drones=2, gui=True)
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         # Generate random XY velocity commands with a positive scaling factor in col 3.
#         actions = np.random.uniform(low=-0.5, high=0.5, size=(env.NUM_DRONES, 4))
#         actions[:, 3] = np.abs(actions[:, 3])
#         obs, reward, terminated, truncated, info = env.step(actions)
#         total_reward += reward
#         done = terminated or truncated
#         print(f"Step Reward: {reward:.2f}, Area Covered: {info['area_coverage']*100:.1f}%")
#     print("Episode finished. Total reward:", total_reward)


import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from controllers.pid_controller import DSLPIDControl
from gymnasium import spaces

class MultiCoverageAviary(BaseRLAviary):
    """
    Multi-agent RL environment for area coverage.
    
    Drones are controlled by a velocity-controlled PID controller.
    They are rewarded for covering new grid cells in a defined 2D area,
    and penalized for collisions. The drone's altitude is forced to z = 1.
    
    The observations returned are a dictionary with:
      - "state": contains base kinematic information (and optionally the action buffer)
      - "grid": a raw coverage grid showing which cells have been visited.
    
    The agent is then responsible for processing the grid via a CNN in its network.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 area=100,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN):
        """
        Initializes the multi-agent area coverage environment.
        Enforces velocity control (ActionType.VEL) to zero out the z-velocity.
        """

        act = ActionType.PID

        # Define episode length (seconds)
        self.EPISODE_LEN_SEC = 30

        # Set the area bounds (e.g., square from -10 to 10 in X and Y)
        self.area_length = np.sqrt(area)
        length = self.area_length // 2
        self.area_bounds = np.array([[-length, length], [-length, length]])
        # Grid resolution (in meters)
        self.grid_resolution = 0.2
        # Compute grid dimensions for the area: (rows, cols)
        self.grid_shape = (
            int((self.area_bounds[0, 1] - self.area_bounds[0, 0]) / self.grid_resolution),
            int((self.area_bounds[1, 1] - self.area_bounds[1, 0]) / self.grid_resolution)
        )
        # Coverage grid (boolean): True means visited cell.
        self.coverage_grid = np.zeros(self.grid_shape, dtype=bool)

        # Collision parameters: if drones come closer than threshold, apply penalty.
        self.collision_dist_threshold = 0.2
        self.collision_penalty = 5.0

        # Initialize the base class.
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]

        # Force all drones to start at z = 1.
        self._setInitialAltitude()

    def _setInitialAltitude(self):
        """
        Ensures all drones start at z = 1.
        """
        self.INIT_XYZS[:, 2] = 1.0

    def _updateCoverage(self):
        """
        Updates the coverage grid using current drone positions.
        Returns:
            int: Number of newly covered grid cells this step.
        """
        new_cells = 0
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y = state[0], state[1]
            # Only if within the defined bounds.
            if (self.area_bounds[0, 0] <= x <= self.area_bounds[0, 1] and
                self.area_bounds[1, 0] <= y <= self.area_bounds[1, 1]):
                # Compute grid indices (limit indices to the grid shape)
                ix = int((x - self.area_bounds[0, 0]) / self.grid_resolution)
                iy = int((y - self.area_bounds[1, 0]) / self.grid_resolution)
                ix = min(ix, self.grid_shape[0] - 1)
                iy = min(iy, self.grid_shape[1] - 1)
                if not self.coverage_grid[ix, iy]:
                    self.coverage_grid[ix, iy] = True
                    new_cells += 1
        return new_cells

    def _computeReward(self):
        """
        Computes the reward for the current step.
         - Rewards new grid cells covered.
         - Penalizes if drones are too close (collision risk).
        Returns:
            float: Total reward for the step.
        """
        reward = 0.0
        new_cells = self._updateCoverage()
        reward += new_cells

        # Collision penalty.
        for i in range(self.NUM_DRONES):
            state_i = self._getDroneStateVector(i)
            pos_i = state_i[0:3]
            for j in range(i + 1, self.NUM_DRONES):
                state_j = self._getDroneStateVector(j)
                pos_j = state_j[0:3]
                if np.linalg.norm(pos_i - pos_j) < self.collision_dist_threshold:
                    reward -= self.collision_penalty
        return reward

    def _computeTerminated(self):
        """
        Terminates the episode if the entire area is covered.
        Returns:
            bool: True if episode is done.
        """
        return np.all(self.coverage_grid)

    def _computeTruncated(self):
        """
        Checks whether the episode should be truncated (e.g., drones go out-of-bounds,
        excessive tilt, or timeout).
        Returns:
            bool: True if episode is truncated.
        """
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y, z = state[0], state[1], state[2]
            if (abs(x) > 30 or abs(y) > 30 or z > 30):
                return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        """
        Returns additional information (area coverage stats).
        Returns:
            dict: Contains area coverage percentage and number of visited cells.
        """
        total_cells = self.grid_shape[0] * self.grid_shape[1]
        area_covered = np.sum(self.coverage_grid) / total_cells
        return {"area_coverage": area_covered,
                "visited_cells": int(np.sum(self.coverage_grid))}

    def _preprocessAction(self, action):
        """
        Forces the z velocity component to zero so that altitude remains at 1.
        Parameters:
            action (np.ndarray): Input for each drone.
        Returns:
            np.ndarray: The RPM commands computed by the parent's controller.
        """

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=np.hstack([state[0:2], 1]), # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=np.hstack([(self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector)[0:2], 0]) # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=np.hstack([next_pos[0:2], 1])
                                                        )
                rpm[k,:] = rpm_k
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    def step(self, actions):
        """
        Executes one step of simulation.
        Preprocesses actions, steps the simulation, and computes reward.
        Parameters:
            actions (np.ndarray): Action commands for all drones.
        Returns:
            obs, reward, terminated, truncated, info: Standard Gym step outputs.
        """
        rpm_commands = self._preprocessAction(actions)
        obs, reward_parent, terminated, truncated, info = super().step(rpm_commands)
        custom_reward = self._computeReward()
        reward = reward_parent + custom_reward
        return obs, reward, terminated, truncated, info

    def reset(self):
        """
        Resets the environment and clears the coverage grid.
        Returns:
            dict: The initial observation.
        """
        self.coverage_grid = np.zeros(self.grid_shape, dtype=bool)
        return super().reset()

    def _observationSpace(self):
        """
        Returns a dictionary observation space with two keys:
            - "state": based on BaseRLAviary's kinematics (and action buffer).
            - "grid": the raw coverage grid.
        """
        base_space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            grid_space = spaces.Box(low=0, high=1,
                                    shape=self.coverage_grid.shape,
                                    dtype=np.float32)
            return spaces.Dict({
                "state": base_space,
                "grid": grid_space
            })
        else:
            return super()._observationSpace()

    def _computeObs(self):
        """
        Computes the observation as a dictionary with:
            - "state": the base kinematic observation.
            - "grid": the raw coverage grid.
        Both are provided per drone.
        """
        base_obs = super()._computeObs()  # shape: (NUM_DRONES, state_dim)
        if self.OBS_TYPE == ObservationType.KIN:
            # The grid is the same for all drones. We return it with its natural 2D shape.
            grid_obs = self.coverage_grid.astype(np.float32)
            # Create a dictionary: "state" per drone and "grid" as the global grid.
            # Option 1: Return replicated grid for each drone:
            # grid_obs = np.tile(grid_obs[None, ...], (self.NUM_DRONES, 1, 1))
            # Option 2: Return the grid once (depending on the expected input for your agent)
            return {"state": base_obs, "grid": grid_obs}
        else:
            return super()._computeObs()


# Example usage:
if __name__ == "__main__":
    env = MultiCoverageAviary(num_drones=2, gui=True)
    obs = env.reset()  # obs is a dict: {"state": ..., "grid": ...}
    done = False
    total_reward = 0
    while not done:
        # Generate random XY velocity commands for demonstration (shape: (NUM_DRONES, 4)).
        actions = np.random.uniform(low=-0.5, high=0.5, size=(env.NUM_DRONES, 4))
        actions[:, 3] = np.abs(actions[:, 3])
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        done = terminated or truncated
        print(f"Step Reward: {reward:.2f}, Area Covered: {info['area_coverage']*100:.1f}%")
    print("Episode finished. Total reward:", total_reward)
