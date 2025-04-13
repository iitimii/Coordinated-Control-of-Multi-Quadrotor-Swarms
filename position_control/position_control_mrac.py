import time

import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pybullet_data

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from controllers.mrac import MRAC
from controllers.nonlinear_mrac import NonlinearMRAC
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_DURATION_SEC = 10
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

    INIT_XYZS = np.array([[0, 0, 0]])
    INIT_RPYS = np.array([[0, 0, 0]])

    TARGET_POS = np.array([[0, 0, 1]])
    TARGET_RPY = np.array([[0, 0, 0]])

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
    
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger ################################# TODO Make Logger update in realtime
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
        ctrl = [MRAC(drone_model=drone) for i in range(num_drones)]

    
    #### Run the simulation ####################################
    X_actual_list, Xm_list, error_list, target_list = [], [], [], []
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        # TARGET_POS = TARGET_POS + 0.001
        # TARGET_POS[0, 2] = 1

        for j in range(num_drones):
            action[j,:], X_actual, Xm, error = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[j],
                                                                target_pos=TARGET_POS[j, :],
                                                                target_rpy=TARGET_RPY[j, :]
                                                                )
        X_actual_list.append(X_actual)
        Xm_list.append(Xm)
        error_list.append(error)
        target_list.append(np.hstack((TARGET_POS[0, :], TARGET_RPY[0, :], np.zeros(6))))

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                    timestamp=i/env.CTRL_FREQ,
                    state=obs[j],
                    control=np.hstack([TARGET_POS[j, :], TARGET_RPY[j, :], np.zeros(6)])
                    )
            
        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()

    logger.save()
    logger.save_as_csv("position_control_mrac")

    if plot:
        plt.figure(figsize=(15, 9))
        X_actual_list = np.array(X_actual_list)
        Xm_list = np.array(Xm_list)
        error_list = np.array(error_list)
        target_list = np.array(target_list)
        states = ["x", "y", "z", "phi", "theta", "psi", "x_dot", "y_dot", "z_dot", "p", "q", "r"] if X_actual_list.shape[1] == 12 else ["x", "y", "z", "x_dot", "y_dot", "z_dot"]

        for i, state in enumerate(states):
            if len(states) == 6:
                plt.subplot(2, 3, i+1)
            else:
                plt.subplot(3, 4, i+1)
            plt.tight_layout()
            plt.plot(X_actual_list[:, i], label='actual')
            plt.plot(Xm_list[:, i], label='model')
            plt.plot(target_list[:, i], label='target', linestyle='--', alpha=0.5, color='black', linewidth=1)
            plt.xlabel('time')
            plt.ylabel(state)
        
        plt.figlegend(['actual plant', 'reference model', 'target'], loc='upper right', ncol=3, labelspacing=0.)

        plt.show()

        # logger.plot()


if __name__ == "__main__":
    run()