import os
from datetime import datetime
from copy import deepcopy
import time
import numpy as np
import pybullet as p
from gymnasium import spaces
from PIL import Image
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from controllers.pid_controller import DSLPIDControl

class TaskAviary(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.RACE,
                 controller=DSLPIDControl,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 image_res=np.array([64, 48]),
                 debug_image=False,
                 obs: ObservationType=ObservationType.RGB,
                 act: ActionType=ActionType.RPM):
        
        self.MAX_RANGE = 3.0
        self.DEBUG_IMAGE = debug_image
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
                         
        
        self.ctrl = [controller(drone_model=drone_model) for i in range(num_drones)]
        
        self.EPISODE_LEN_SEC = 20
        self.IMG_RES = image_res
        self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
        self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
        self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))

        self.obs_kin = np.zeros((self.NUM_DRONES,12))

        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self.CLIENT)
        
    def _addObstacles(self):
        self.cube_id = p.loadURDF("cube_small.urdf",
                       [0, 0, 0],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        
    def _actionSpace(self):
        size = 4 # x,y,z,yaw
        act_lower_bound = np.array([-self.MAX_RANGE*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+self.MAX_RANGE*np.ones(size) for i in range(self.NUM_DRONES)])

        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
    
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    

    
    def _preprocessAction(self, action):
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            # TODO Augment Target with Min Snap Traj
            state = self._getDroneStateVector(k)
            res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=target[0:3],
                                                    target_rpy=np.array([0,0,target[3]]),
                                                    # target_vel=target[6:9],
                                                    # target_rpy_rates=target[9:12]
                                                    )
            rpm[k,:] = res
        return rpm
    

    def _getDroneImages(self, nth_drone, segmentation: bool=False):

        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([10, 0, -1000])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0.5*self.L, 0, -0.5*self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[1, 0, 0],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        
        rgb = np.reshape(rgb, (h, w, 4))        
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg
    
    
    def _observationSpace(self):
        image_space = spaces.Box(low=0, high=255,
                            shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
        #### Add action buffer to observation space ################
        act_lo = -self.MAX_RANGE
        act_hi = +self.MAX_RANGE
        for i in range(self.ACTION_BUFFER_SIZE):
            obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
            obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
        
        kinematics_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        return spaces.Dict({
            'rgba': image_space,
            'kinematics': kinematics_space
        })

    def _computeObs(self):
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            self.obs_kin[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)

            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                rgb, dep, seg = self._getDroneImages(i, segmentation=False)
                # self.rgb[i,:,:,:] = rgb # TODO the issue is copying this array, fix later when using multi drones
                self.rgb = rgb

                if self.DEBUG_IMAGE:
                    os.makedirs("results", exist_ok=True)
                    print(self.rgb.shape)
                    Image.fromarray(self.rgb, 'RGBA').save(f'results/image_{i}.png')

        # ret_rgb = np.array(self.rgb).reshape(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4).astype('float32')
        ret_rgb = self.rgb
        ret_kin = np.array([self.obs_kin[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

        for i in range(self.ACTION_BUFFER_SIZE):
            ret_kin = np.hstack([ret_kin, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

        return {"rgba": ret_rgb,
                "kinematics": ret_kin}
        
        
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        cube_info = p.getBodyInfo(self.cube_id)
        print(cube_info)
        return 0
    
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return False
        
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)

        if (abs(state[0]) > self.MAX_RANGE or abs(state[1]) > self.MAX_RANGE or state[2] > self.MAX_RANGE
             or abs(state[7]) > .4 or abs(state[8]) > .4
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        return {"answer": 42} # Calculated by the Deep Thought supercomputer in 7.5M years, very funny