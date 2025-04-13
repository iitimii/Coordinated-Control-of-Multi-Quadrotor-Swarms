import os
from envs.TaskAviary import TaskAviary
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from controllers.pid_controller import DSLPIDControl
from PIL import Image
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy


DRONE_MODEL = DroneModel.CF2X
NUM_DRONES = 1
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 120
DEFAULT_OBS = ObservationType("rgb")
DEFAULT_ACT = ActionType("one_d_pid")  # Not used
DEFAULT_RECORD = False
DEFAULT_IMG_RES = np.array([640, 480])
DEFAULT_OUTPUT_FOLDER = "results"
d = 1


def run(output_folder=DEFAULT_OUTPUT_FOLDER):
    filename = os.path.join(
        output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    )
    if not os.path.exists(filename):
        os.makedirs(filename + "/")

    INIT_XYZS = np.array([[0, 0, 1] for i in range(NUM_DRONES)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(NUM_DRONES)])

    train_env = make_vec_env(
        TaskAviary,
        env_kwargs=dict(
            drone_model=DRONE_MODEL,
            controller=DSLPIDControl,
            num_drones=NUM_DRONES,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
            pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=DEFAULT_RECORD,
            image_res=DEFAULT_IMG_RES,
            gui=False,
            debug_image=False,
        ),
        n_envs=1,
        seed=0,
    )

    eval_env = TaskAviary(
        drone_model=DRONE_MODEL,
        controller=DSLPIDControl,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=DEFAULT_RECORD,
        image_res=DEFAULT_IMG_RES,
        gui=False,
        debug_image=False,
    )

    model = PPO('MultiInputPolicy', train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)
    
    model.learn(10)

    test_env = TaskAviary(
        drone_model=DRONE_MODEL,
        controller=DSLPIDControl,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        record=DEFAULT_RECORD,
        image_res=DEFAULT_IMG_RES,
        gui=True,
        debug_image=False,
    )


    test_env.reset()
    action = np.array([[0, 0, 0.1, 0]])  # shape (1,4) x,y,z,yaw
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs_rgb, obs_kin = (
            obs["rgba"],
            obs["kinematics"],
        )  # returns image for only one drone for now, error in copying to slice array

        test_env.render()


if __name__ == "__main__":
    run()
