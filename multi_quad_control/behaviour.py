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

from utils.utils import initialize_drones

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 20
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

def behaviour(obs, num_drones, separation_weight=3.5, alignment_weight=1.0, cohesion_weight=5.0):
    positions = np.array([o[0:3] for o in obs])
    velocities = np.array([o[10:13] for o in obs])  # Assuming vel is at index 10-13
    target_positions = np.zeros_like(positions)

    for i in range(num_drones):
        pos_i = positions[i]
        vel_i = velocities[i]

        separation = np.zeros(3)
        alignment = np.zeros(3)
        cohesion = np.zeros(3)
        neighbor_count = 0

        for j in range(num_drones):
            if i != j:
                pos_j = positions[j]
                vel_j = velocities[j]
                dist = np.linalg.norm(pos_j - pos_i)
                if dist < 2.0 and dist > 1e-2:
                    separation += (pos_i - pos_j) / (dist**2)
                    alignment += vel_j
                    cohesion += pos_j
                    neighbor_count += 1

        if neighbor_count > 0:
            separation /= neighbor_count
            alignment /= neighbor_count
            cohesion = (cohesion / neighbor_count) - pos_i

        behaviour_vector = (
            separation_weight * separation +
            alignment_weight * alignment +
            cohesion_weight * cohesion
        )

        next_pos = pos_i + behaviour_vector * 0.05  # timestep gain
        next_pos[2] = 1.0  # Fixed altitude
        target_positions[i] = next_pos

    return target_positions

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
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
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
        target_xyzs = behaviour(obs, num_drones) 

        for j in range(num_drones):
            action[j,:], _, _  = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP, # action[j, :], _, _
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
    logger.save_as_csv("behaviour")
    if plot:
        logger.plot()


    

if __name__ == "__main__":
    run()