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
from trajectory_tracking.trajGen3D import get_MST_coefficients, generate_trajectory

from formation_flight.collision_avoidance import collision_avoidance
from utils.utils import initialize_drones

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 10
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 30
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
    INIT_XYZS = INIT_XYZS = initialize_drones(num_drones, 0.5, [-3,3])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])

    target_xyzs = np.array([[0, i*d, 1] for i in range(num_drones)])
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

    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        target_xyzs = consensus(obs) 

        for j in range(num_drones):
            # traj = np.vstack((obs[j, :3], target_xyzs[j]))
            # coeff_x, coeff_y, coeff_z = get_MST_coefficients(traj)
            # state = generate_trajectory(1, 1, traj, coeff_x, coeff_y, coeff_z)
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=target_xyzs[j],
                                                                    # target_vel=state.vel,
                                                                    target_rpy=target_rpys[j, :]
                                                                    )

        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack((target_xyzs[j, :], target_rpys[j, :], np.zeros(6)))
                       )

        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("consensus")
    if plot:
        logger.plot()

def consensus(current_states, radius=2, height=1):
    current_positions = np.array([state[:2] for state in current_states])
    centroid = np.mean(current_positions, axis=0)
    num_drones = current_positions.shape[0]

    angles = np.linspace(0, 2 * np.pi, num_drones, endpoint=False)

    target_xyzs = np.array([
        [
            centroid[0] + radius * np.cos(angle),  # x-coordinate
            centroid[1] + radius * np.sin(angle),  # y-coordinate
            height
        ]
        for angle in angles
    ])
    target_xyzs = collision_avoidance(target_xyzs, current_states, safe_distance=0.5)
    return target_xyzs

if __name__ == "__main__":
    run()
