import os
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from controllers.pid_controller import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from utils.utils import initialize_drones

# === CONFIG ===
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

CONSENSUS_GAIN = 0.1
INNER_RADIUS = 1.5           # Radius of the rotating formation
INNER_ROT_SPEED = 0.3        # Speed of rotation around formation center

OUTER_RADIUS = 10.5           # Radius of the global trajectory (formation center)
OUTER_ROT_SPEED = 0.15       # Speed of the circle center around origin


def consensus(obs, d_offsets, num_drones, sim_time, fixed_altitude=1.0):
    current_positions = np.zeros((num_drones, 2))
    target_positions = np.zeros((num_drones, 3))
    
    # Move the formation center in a circular trajectory
    center_x = OUTER_RADIUS * np.cos(OUTER_ROT_SPEED * sim_time)
    center_y = OUTER_RADIUS * np.sin(OUTER_ROT_SPEED * sim_time)
    center = np.array([center_x, center_y])

    # Rotate the local offsets
    inner_angle = INNER_ROT_SPEED * sim_time
    rotation_matrix = np.array([[np.cos(inner_angle), -np.sin(inner_angle)],
                                [np.sin(inner_angle),  np.cos(inner_angle)]])
    rotated_offsets = np.dot(d_offsets[:, :2], rotation_matrix.T)

    # Get full desired position for each drone
    desired_positions = center + rotated_offsets

    for i in range(num_drones):
        current_positions[i] = np.array(obs[i][:2])

    for i in range(num_drones):
        consensus_sum = np.zeros(2)
        for j in range(num_drones):
            if i == j:
                continue
            diff_current = current_positions[j] - current_positions[i]
            diff_desired = desired_positions[j] - desired_positions[i]
            consensus_sum += (diff_current - diff_desired)

        xy_target = current_positions[i] + CONSENSUS_GAIN * consensus_sum
        target_positions[i] = np.array([xy_target[0], xy_target[1], fixed_altitude])
    
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

    # Offsets for a static circular formation (body frame)
    d_offsets = np.array([
        [INNER_RADIUS * np.cos(2 * np.pi * i / num_drones),
         INNER_RADIUS * np.sin(2 * np.pi * i / num_drones),
         0]
        for i in range(num_drones)
    ])

    # Initial drone placement
    INIT_XYZS = initialize_drones(num_drones, 0.5, [-3, 3])
    INIT_RPYS = np.array([[0, 0, 0] for _ in range(num_drones)])
    target_rpys = np.array([[0, 0, 0] for _ in range(num_drones)])

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
                     user_debug_gui=user_debug_gui)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
    action = np.zeros((num_drones, 4))
    START = time.time()

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)
        sim_time = i / env.CTRL_FREQ

        target_xyzs = consensus(obs, d_offsets, num_drones, sim_time)

        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_xyzs[j],
                target_rpy=target_rpys[j]
            )

        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=sim_time,
                       state=obs[j],
                       control=np.hstack((target_xyzs[j], target_rpys[j], np.zeros(6))))

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("double_circle_consensus")
    if plot:
        logger.plot()


if __name__ == "__main__":
    run()
