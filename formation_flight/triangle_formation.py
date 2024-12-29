import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from controllers.pid_controller import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 6
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):

    R = 1.3
    d = 0.5
    INIT_XYZS = np.array([[i, 0, 0] for i in range(num_drones)])  # shape(num_drones, 3)
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    # Initially set target position in a triangle formation
    target_xyzs = triangle_formation(side_length=1, center_point=(0, 0, 1), num_drones=num_drones)
    target_rpys = np.array([[0, 0, 0] for i in range(num_drones)])

    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui
                     )

    PYB_CLIENT = env.getPyBulletClient()
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        # Update target positions for triangle formation
        target_xyzs = triangle_formation(side_length=1, center_point=(0, 0, 1), num_drones=num_drones)

        # Assign drones to positions based on shortest distance
        assignments = assign_positions_to_drones(INIT_XYZS, target_xyzs)

        for j in range(num_drones):
            assigned_position = target_xyzs[assignments[j]]
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                  state=obs[j],
                                                                  target_pos=assigned_position,
                                                                  target_rpy=target_rpys[j, :]
                                                                  )

        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i / env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack((target_xyzs[j, :], target_rpys[j, :], np.zeros(6)))
                       )

        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("triangle_formation")
    if plot:
        logger.plot()

def triangle_formation(side_length=1.0, center_point=np.array([0, 0, 1]), num_drones=5):
    """
    Generate positions for drones in a triangular formation.
    The formation is created row by row, with each row containing one more drone than the previous one.
    """
    target_xyzs = []
    row = 0
    count = 0

    while count < num_drones:
        # Number of drones in the current row
        drones_in_row = row + 1

        for i in range(drones_in_row):
            if count >= num_drones:
                break

            x_offset = (i - row / 2) * side_length  # Center the row
            y_offset = -row * (math.sqrt(3) / 2) * side_length  # Stagger rows downward

            target_xyzs.append([center_point[0] + x_offset, center_point[1] + y_offset, center_point[2]])
            count += 1

        row += 1

    return np.array(target_xyzs)

def assign_positions_to_drones(initial_positions, target_positions):
    """
    Assign drones to target positions based on the shortest distance.
    """
    num_drones = len(initial_positions)
    assignments = [-1] * num_drones
    assigned_targets = [False] * num_drones

    for drone_index in range(num_drones):
        min_distance = float('inf')
        best_target_index = -1
        for target_index in range(num_drones):
            if not assigned_targets[target_index]:
                distance = np.linalg.norm(initial_positions[drone_index] - target_positions[target_index])
                if distance < min_distance:
                    min_distance = distance
                    best_target_index = target_index

        assignments[drone_index] = best_target_index
        assigned_targets[best_target_index] = True

    return assignments

if __name__ == "__main__":
    run()
