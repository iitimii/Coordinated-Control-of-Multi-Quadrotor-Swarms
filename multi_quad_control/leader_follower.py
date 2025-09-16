import os
import time
import argparse
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from controllers.pid_controller import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from utils.utils import initialize_drones

# Default parameters
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

# Global parameters for formation control
R = 1.3              # Leader's circular trajectory radius (for default, V, line, triangle modes)
d = 0.5              # Relative offset distance used in various formations
circle_radius = 2.0  # Radius used for circle formation

# Parameters for collision avoidance
SAFE_DISTANCE = 0.5      # Minimum allowed distance between drones
AVOIDANCE_GAIN = 0.2     # Gain multiplier for repulsion effect

# Global variables for formation mode and simulation time reference
formation_mode = "default"  # default: leader-follower
simulation_start = None

def get_formation_targets(num_drones):
    """
    Computes target positions for the drones based on the global formation_mode.
    
    Modes:
      - "default": Leader-follower (with simple alternating offsets)
      - "v": V formation with leader at the apex.
      - "circle": Drones evenly on a circle centered at the leader.
      - "line": Drones lined up along the negative X-axis of the leader.
      - "square": Drones evenly along a square's perimeter centered on the leader.
      - "grid": Drones arranged in a grid centered on the leader.
      - "triangle": Drones distributed along the edges of an equilateral triangle.
    """
    global formation_mode
    t = time.time() - simulation_start
    # Leader position along circular trajectory (for default, v, line, triangle)
    leader_x = R * math.cos(0.2 * t)
    leader_y = R * math.sin(0.2 * t)
    leader_z = 1.0  # Constant altitude
    leader_pos = np.array([leader_x, leader_y, leader_z])
    
    target_positions = np.zeros((num_drones, 3))
    
    if formation_mode in ["default", "v", "line", "triangle"]:
        # Leader is drone 0
        target_positions[0] = leader_pos
        
        if formation_mode == "default":
            # Default: Follower drones maintain alternating fixed offsets relative to leader.
            for i in range(1, num_drones):
                if i % 2 == 0:
                    offset = np.array([d, 0.0, 0.0])
                else:
                    offset = np.array([-d, 0.0, 0.0])
                target_positions[i] = leader_pos + offset

        elif formation_mode == "v":
            # V formation: Leader at apex. Alternate drones placed along the arms of a V.
            for i in range(1, num_drones):
                idx = (i + 1) // 2  # determines the distance from the leader
                if i % 2 == 1:
                    # Right wing: offset to the right and backwards
                    offset = np.array([-idx * d, idx * d, 0])
                else:
                    # Left wing: offset to the left and backwards
                    offset = np.array([-idx * d, -idx * d, 0])
                target_positions[i] = leader_pos + offset

        elif formation_mode == "line":
            # Line formation: Drones line up along the negative X-axis from leader.
            for i in range(1, num_drones):
                offset = np.array([-i * d, 0, 0])
                target_positions[i] = leader_pos + offset

        elif formation_mode == "triangle":
            # Triangle formation: Place drones along the perimeter of an equilateral triangle.
            side = 2 * d * (num_drones / 4)  # side length scaled with number of drones
            v1 = leader_pos  # Leader vertex
            # Compute approximate vertices for an equilateral triangle
            v2 = leader_pos + np.array([-side, side * math.tan(math.radians(30)), 0])
            v3 = leader_pos + np.array([side, side * math.tan(math.radians(30)), 0])
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            followers = num_drones - 1
            per_edge = max(1, followers // 3)
            idx = 1
            for (start, end) in edges:
                for j in range(1, per_edge + 1):
                    if idx >= num_drones:
                        break
                    ratio = j / (per_edge + 1)
                    point = start * (1 - ratio) + end * ratio
                    target_positions[idx] = point
                    idx += 1
            while idx < num_drones:
                ratio = (idx) / (followers + 1)
                point = v1 * (1 - ratio) + v2 * ratio
                target_positions[idx] = point
                idx += 1

    elif formation_mode == "circle":
        # Circle formation: Leader's position is the center of the circle.
        center = leader_pos
        for i in range(num_drones):
            angle = (2 * math.pi / num_drones) * i
            offset = np.array([circle_radius * math.cos(angle),
                               circle_radius * math.sin(angle),
                               0])
            target_positions[i] = center + offset

    elif formation_mode == "square":
        # Square formation: Arrange drones along the perimeter of a square centered on leader.
        center = leader_pos
        side_len = 2 * d * math.sqrt(num_drones)  # side length scaled with number of drones
        perimeter = 4 * side_len
        remaining = num_drones - 1
        for i in range(1, num_drones):
            fraction = ((i - 1) / remaining) * perimeter
            if fraction < side_len:
                pos = np.array([-side_len/2 + fraction, side_len/2, 0])
            elif fraction < 2 * side_len:
                pos = np.array([side_len/2, side_len/2 - (fraction - side_len), 0])
            elif fraction < 3 * side_len:
                pos = np.array([side_len/2 - (fraction - 2*side_len), -side_len/2, 0])
            else:
                pos = np.array([-side_len/2, -side_len/2 + (fraction - 3*side_len), 0])
            target_positions[i] = center + pos

    elif formation_mode == "grid":
        # Grid formation: Arrange drones in rows and columns centered around leader.
        center = leader_pos
        cols = math.ceil(math.sqrt(num_drones))
        rows = math.ceil(num_drones / cols)
        spacing = d * 2
        x_offsets = (np.arange(cols) - (cols - 1) / 2.0) * spacing
        y_offsets = (np.arange(rows) - (rows - 1) / 2.0) * spacing
        
        grid_points = []
        for r_idx in range(rows):
            for c_idx in range(cols):
                grid_points.append(np.array([x_offsets[c_idx], y_offsets[r_idx], 0]))
        for i in range(num_drones):
            target_positions[i] = center + grid_points[i]
    
    return target_positions

def apply_collision_avoidance(target_positions, obs, safe_distance=SAFE_DISTANCE, avoidance_gain=AVOIDANCE_GAIN):
    """
    Adjusts the formation target positions using a basic repulsive potential field.
    For each drone, if any other drone is closer than safe_distance (based on current positions),
    a repulsion term is added to the target position.
    
    Parameters:
      - target_positions: np.array shape (num_drones, 3)
      - obs: list/array of states; assume first 3 entries are [x, y, z]
      - safe_distance: minimum allowed separation
      - avoidance_gain: scaling factor for repulsion force
    """
    num_drones = target_positions.shape[0]
    adjusted_targets = np.copy(target_positions)
    
    # Loop through each drone
    for i in range(num_drones):
        pos_i = np.array(obs[i][:3])
        repulsion = np.zeros(3)
        
        # Compare with every other drone
        for j in range(num_drones):
            if i == j:
                continue
            pos_j = np.array(obs[j][:3])
            diff = pos_i - pos_j
            distance = np.linalg.norm(diff)
            # If drones are too close, add a repulsion force
            if distance < safe_distance and distance > 1e-6:
                repulsion += avoidance_gain * (diff / distance) * (safe_distance - distance)
        adjusted_targets[i] += repulsion
    return adjusted_targets

def check_keyboard_events():
    """
    Checks for keyboard events and updates the global formation_mode.
    
    Keys:
      - "v": V formation
      - "c": Circle formation
      - "l": Line formation
      - "s": Square formation
      - "g": Grid formation
      - "t": Triangle formation
    """
    global formation_mode
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if (k == ord('v')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "v"
            print("Switched to V formation")
        elif (k == ord('c')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "circle"
            print("Switched to circle formation")
        elif (k == ord('l')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "line"
            print("Switched to line formation")
        elif (k == ord('s')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "square"
            print("Switched to square formation")
        elif (k == ord('g')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "grid"
            print("Switched to grid formation")
        elif (k == ord('t')) and (v & p.KEY_WAS_TRIGGERED):
            formation_mode = "triangle"
            print("Switched to triangle formation")

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
    global simulation_start, formation_mode
    simulation_start = time.time()
    formation_mode = "default"

    INIT_XYZS = initialize_drones(num_drones, 0.5, [-3, 3])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])
    target_xyzs = np.array([[0, i * d, 1] for i in range(num_drones)])
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
                    colab=colab)

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        check_keyboard_events()
        target_xyzs = get_formation_targets(num_drones)
        # Apply collision avoidance adjustment to target positions
        target_xyzs = apply_collision_avoidance(target_xyzs, obs, safe_distance=SAFE_DISTANCE, avoidance_gain=AVOIDANCE_GAIN)

        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                                        control_timestep=env.CTRL_TIMESTEP,
                                        state=obs[j],
                                        target_pos=target_xyzs[j],
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
    # logger.save()
    # logger.save_as_csv("formation_control")
    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    run()
