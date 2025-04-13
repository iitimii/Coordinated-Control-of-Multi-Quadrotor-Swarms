import os
import time
import math
import numpy as np
import pybullet as p
from scipy.optimize import linear_sum_assignment

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from controllers.pid_controller import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

# Constants
DEFAULT_DRONES = DroneModel.CF2X
DEFAULT_NUM_DRONES = 7
DEFAULT_PHYSICS = Physics.PYB
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = 'results'

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER
        ):

    # Initialize drones in random positions (x, y between -3 and 3, z=1)
    INIT_XYZS = np.random.uniform(-3, 3, (num_drones, 3))
    INIT_XYZS[:, 2] = 1  # Set fixed altitude
    INIT_RPYS = np.zeros((num_drones, 3))

    # Generate target square formation
    target_xyzs = square_grid_formation(grid_size=1.5, center_point=(0, 0, 1), num_drones=num_drones)
    target_rpys = np.zeros((num_drones, 3))

    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video
                     )

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder
                    )

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    action = np.zeros((num_drones, 4))
    START = time.time()

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)

        # Get current positions and update assignments
        current_positions = np.array([ob[0:3] for ob in obs])
        
        # Optimal assignment using Hungarian algorithm
        cost_matrix = np.linalg.norm(current_positions[:, None] - target_xyzs, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_targets = target_xyzs[col_ind]

        # Update drone controls
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=assigned_targets[j],
                target_rpy=target_rpys[j]
            )

            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack((assigned_targets[j], target_rpys[j], np.zeros(6)))
                       )

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("swarm_formation")
    if plot:
        logger.plot()

def square_grid_formation(grid_size=1.0, center_point=(0, 0, 1), num_drones=7):
    """Generate square formation positions with perfect centering"""
    positions = []
    grid_side = math.ceil(math.sqrt(num_drones))
    half_extent = grid_size * (grid_side - 1) / 2
    
    count = 0
    for i in range(grid_side):
        for j in range(grid_side):
            if count >= num_drones:
                break
            x = center_point[0] + (i - (grid_side-1)/2) * grid_size
            y = center_point[1] + (j - (grid_side-1)/2) * grid_size
            positions.append([x, y, center_point[2]])
            count += 1
    return np.array(positions)

if __name__ == "__main__":
    run()