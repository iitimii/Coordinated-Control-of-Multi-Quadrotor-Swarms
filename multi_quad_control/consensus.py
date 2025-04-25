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
from controllers.mrac import MRAC
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from utils.utils import initialize_drones

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 5
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
        ctrl = [MRAC(drone_model=drone) for i in range(num_drones)]

    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        target_xyzs = consensus(env, obs, 10, time.time()-START) 

        for j in range(num_drones):
            action[j,:], X_actual, Xm, error  = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP, # action[j, :], _, _
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

# def consensus(env, current_states, alpha=1.0, gamma=1.0, k=1.0, height=1):
#     """
#     Consensus-based formation control algorithm.
    
#     Parameters:
#     - env: The simulation environment (to access adjacency matrix).
#     - current_states: Current state of all drones (position and velocity).
#     - alpha: Damping constant.
#     - gamma: Coupling constant.
#     - k: Consensus gain.
#     - height: Desired height for all drones.
    
#     Returns:
#     - target_positions: Updated positions for the drones based on consensus.
#     """
#     # Get the adjacency matrix
#     adj_matrix = env._getAdjacencyMatrix()
#     num_drones = len(current_states)

#     # Extract positions and velocities from current_states
#     positions = np.array([state[0:3] for state in current_states])  # Assuming state[0:3] is position
#     velocities = np.array([state[3:6] for state in current_states])  # Assuming state[3:6] is velocity

#     # Initialize target positions and velocities
#     target_positions = np.copy(positions)
#     target_velocities = np.copy(velocities)

#     # Update positions and velocities using consensus algorithm
#     for i in range(num_drones):
#         neighbors = np.where(adj_matrix[i] > 0)[0]  # Find neighbors of drone i
#         if len(neighbors) > 0:
#             # Compute consensus terms
#             position_diff = np.sum([positions[i] - positions[j] for j in neighbors], axis=0)
#             velocity_diff = np.sum([velocities[i] - velocities[j] for j in neighbors], axis=0)

#             # Update velocity (second-order dynamics)
#             target_velocities[i] = velocities[i] - alpha * velocities[i] \
#                                    - k * (position_diff + gamma * velocity_diff)

#             # Update position
#             target_positions[i] = positions[i] + target_velocities[i] * env.CTRL_TIMESTEP

#     # Adjust height for formation
#     target_positions[:, 2] = height

#     return target_positions

# def consensus(env, current_states, switch_time, sim_time, alpha=1.0, gamma=1.0, k=1.0, 
#               line_spacing=0.5, triangle_spacing=1.0, height=1):
#     """
#     Consensus-based dynamic formation control (line to triangle).
    
#     Parameters:
#     - env: Simulation environment (to access adjacency matrix).
#     - current_states: Current state of all drones (position and velocity).
#     - switch_time: Time (in seconds) to switch from line to triangle formation.
#     - sim_time: Current simulation time.
#     - alpha: Damping constant.
#     - gamma: Coupling constant.
#     - k: Consensus gain.
#     - line_spacing: Spacing between drones in the line formation.
#     - triangle_spacing: Spacing between drones in the triangle formation.
#     - height: Desired height for all drones.
    
#     Returns:
#     - target_positions: Updated positions for the drones based on the active formation.
#     """
#     # Get the adjacency matrix
#     adj_matrix = env._getAdjacencyMatrix()
#     num_drones = len(current_states)

#     # Extract positions and velocities
#     positions = np.array([state[0:3] for state in current_states])
#     velocities = np.array([state[3:6] for state in current_states])

#     # Initialize target positions and velocities
#     target_positions = np.copy(positions)
#     target_velocities = np.copy(velocities)

#     if sim_time < switch_time:
#         # Line formation
#         x_c, z_c = 0, height  # Line is fixed at x_c and z_c
#         for i in range(num_drones):
#             target_positions[i] = [x_c, i * line_spacing, z_c]
#     else:
#         # Triangle formation
#         x_c, z_c = 0, height  # Triangle base center
#         for i in range(num_drones):
#             # Compute triangle row and column (row-major layout)
#             row = int(np.floor(np.sqrt(2 * i)))  # Row index
#             col = i - (row * (row + 1)) // 2     # Column index within the row
#             target_positions[i] = [x_c + col * triangle_spacing, 
#                                    -row * triangle_spacing, 
#                                    z_c]

#     # Adjust positions using consensus algorithm (optional smoothing)
#     for i in range(num_drones):
#         neighbors = np.where(adj_matrix[i] > 0)[0]  # Find neighbors of drone i
#         if len(neighbors) > 0:
#             # Consensus terms
#             position_diff = np.sum([positions[i] - positions[j] for j in neighbors], axis=0)
#             velocity_diff = np.sum([velocities[i] - velocities[j] for j in neighbors], axis=0)

#             # Update velocity (second-order dynamics)
#             target_velocities[i] = velocities[i] - alpha * velocities[i] \
#                                    - k * (position_diff + gamma * velocity_diff)

#             # Update position based on velocity
#             target_positions[i] += target_velocities[i] * env.CTRL_TIMESTEP

#     return target_positions

def consensus(
    env, current_states, switch_time, sim_time, alpha=1.0, gamma=1.0, k=1.0,
    line_spacing=0.5, circle_radius=2.0, height=1, d_safe=0.5, avoidance_gain=2.0
):
    """
    Consensus-based formation control with collision avoidance and formation switching.
    
    Parameters:
    - env: Simulation environment (to access adjacency matrix).
    - current_states: Current state of all drones (position and velocity).
    - switch_time: Time to switch from line to circle formation.
    - sim_time: Current simulation time.
    - alpha: Damping constant.
    - gamma: Coupling constant.
    - k: Consensus gain.
    - line_spacing: Desired spacing for line formation.
    - circle_radius: Radius for circle formation.
    - height: Desired height for all drones.
    - d_safe: Minimum allowable distance between drones.
    - avoidance_gain: Strength of collision avoidance repulsion.
    
    Returns:
    - target_positions: Updated target positions for the drones.
    """
    # Get adjacency matrix
    adj_matrix = env._getAdjacencyMatrix()
    num_drones = len(current_states)

    # Extract positions and velocities
    positions = np.array([state[0:3] for state in current_states])
    velocities = np.array([state[3:6] for state in current_states])

    # Initialize target positions and velocities
    target_positions = np.copy(positions)
    target_velocities = np.copy(velocities)

    # Formation switching logic
    if sim_time < switch_time:
        # Line formation
        for i in range(num_drones):
            target_positions[i] = [0, i * line_spacing, height]
    else:
        # Circle formation
        center = np.array([0, 0, height])
        angle_increment = 2 * np.pi / num_drones
        for i in range(num_drones):
            angle = i * angle_increment
            target_positions[i] = [
                center[0] + circle_radius * np.cos(angle),
                center[1] + circle_radius * np.sin(angle),
                center[2],
            ]

    # Apply collision avoidance
    for i in range(num_drones):
        neighbors = np.where(adj_matrix[i] > 0)[0]  # Find neighbors
        repulsive_force = np.zeros(3)  # Initialize repulsion

        # Compute repulsive forces from neighbors
        for j in range(num_drones):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < d_safe:
                    direction = (positions[i] - positions[j]) / distance
                    repulsive_force += avoidance_gain * (1 / distance - 1 / d_safe) * (1 / distance**2) * direction

        # Consensus adjustment for velocity
        if len(neighbors) > 0:
            position_diff = np.sum([positions[i] - positions[j] for j in neighbors], axis=0)
            velocity_diff = np.sum([velocities[i] - velocities[j] for j in neighbors], axis=0)

            # Update velocity with consensus dynamics
            target_velocities[i] = velocities[i] - alpha * velocities[i] \
                                   - k * (position_diff + gamma * velocity_diff)

        # Update position with consensus and repulsion
        target_positions[i] += target_velocities[i] * env.CTRL_TIMESTEP + repulsive_force * env.CTRL_TIMESTEP

    return target_positions

if __name__ == "__main__":
    run()
