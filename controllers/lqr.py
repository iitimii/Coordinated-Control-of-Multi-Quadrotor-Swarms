import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
import control as ct

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class LQRControl(BaseControl):
    """LQR Controller class for Crazyflies using quaternion error for attitude control."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] LQRControl requires DroneModel.CF2X, CF2P, or RACE")
            exit()

        # Get UAV properties from URDF
        self.Ixx = self._getURDFParameter("ixx")
        self.Iyy = self._getURDFParameter("iyy")
        self.Izz = self._getURDFParameter("izz")
        self.mass = self._getURDFParameter("m")
        self.l = self._getURDFParameter("arm")
        self.g = g

        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.RACE]:
            self.MIXER_MATRIX = np.array([ 
                [-0.5, -0.5, -1],
                [-0.5,  0.5,  1],
                [ 0.5,  0.5, -1],
                [ 0.5, -0.5,  1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1,  -1],
                [+1, 0,   1],
                [0,  1,  -1],
                [-1, 0,   1]
            ])

        # For the new state definition (pos, vel, att, ang_vel):
        #  state indices: 0-2: pos, 3-5: vel, 6-8: attitude error, 9-11: angular velocity error
        self.K = self._compute_K()  # LQR gain (computed once for hover)
        self.reset()

    def _compute_K(self):
        # Linear model about hover with 12 states and 4 inputs.
        # State vector:
        #   x = [ pos (3), vel (3), att_error (3), ang_vel (3) ]
        # Dynamics:
        #   pos_dot = vel
        #   vel_dot = [ g * (θ_error), -g * (φ_error), 1/mass * delta_thrust ]
        #   att_dot = angular velocity (small-angle approximation)
        #   ang_vel_dot = I^-1 * delta_torques
        # The A and B matrices are built accordingly.
        A = np.zeros((12, 12))
        # Position derivatives
        A[0:3, 3:6] = np.eye(3)
        # Lateral acceleration due to attitude error:
        # x-axis acceleration: ax = +g * (θ_error)  →  A[3, 7] = g
        # y-axis acceleration: ay = -g * (φ_error)  →  A[4, 6] = -g
        A[3, 7] = self.g
        A[4, 6] = -self.g
        # Attitude error dynamics: derivative of attitude error = angular velocity error
        A[6:9, 9:12] = np.eye(3)
        # B matrix: control inputs: [delta_thrust, Mx, My, Mz]
        B = np.zeros((12, 4))
        # Thrust affects only the vertical acceleration (assumed z direction index 5)
        B[5, 0] = 1 / self.mass
        # Moments affect angular acceleration
        B[9, 1] = 1 / self.Ixx
        B[10, 2] = 1 / self.Iyy
        B[11, 3] = 1 / self.Izz

        # Define cost matrices (tune these weights as required)
        Q = np.diag([1e1, 1e1, 1e1,    # pos error weights
                     1e1, 1e1, 1e1,    # velocity error weights
                     1e2, 1e2, 1e2,    # attitude error weights
                     1e-3, 1e-3, 1e-3])  # angular velocity error weights

        R = np.diag([1e-1, 1e3, 1e3, 1e3])  # cost for [delta_thrust, Mx, My, Mz]

        # Compute LQR gain matrix
        K, _, _ = ct.lqr(A, B, Q, R)
        return K

    def reset(self):
        super().reset()

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy,  # target attitude as quaternion [x, y, z, w]
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """
        Inputs are in the inertial frame.
        cur_quat and target_quat are provided as [x, y, z, w].
        """
        self.control_counter += 1

        # Compute position and velocity errors in the inertial frame.
        pos_error = target_pos - cur_pos
        vel_error = target_vel - cur_vel

        # Compute attitude error: q_error = q_target * inv(q_current)
        q_curr = Rotation.from_quat(cur_quat)
        target_quat = Rotation.from_euler('xyz', target_rpy, degrees=True).as_quat()
        q_target = Rotation.from_quat(target_quat)
        # Compute the error quaternion. (For small errors, the corresponding rotation vector
        # is an adequate representation.)
        q_error = q_target * q_curr.inv()
        # Ensure shortest path (if scalar part negative, invert)
        q_error_arr = q_error.as_quat()  # [x, y, z, w]
        if q_error_arr[3] < 0:
            q_error_arr = -q_error_arr
        # Convert error quaternion to rotation vector (attitude error)
        att_error = Rotation.from_quat(q_error_arr).as_rotvec()

        # Angular velocity error (in inertial frame, or if you prefer body frame, adjust accordingly)
        ang_vel_error = target_rpy_rates - cur_ang_vel

        # Assemble the error state vector.
        # Ordering: [pos_error (3), vel_error (3), att_error (3), ang_vel_error (3)]
        x_error = np.hstack((pos_error, vel_error, att_error, ang_vel_error))

        # Compute control input: u = -K x_error.
        # Here, u[0] is the additional thrust required (delta_thrust) and u[1:4] are the moments.
        u = -np.dot(self.K, x_error)

        # The nominal thrust to hover is mass * g, so add it:
        target_thrust = u[0] + self.mass * self.g
        # The thrust must be nonnegative.
        target_thrust = max(target_thrust, 0)
        target_torques = u[1:4]

        # Convert thrust and torques into PWM commands.
        # The conversion here assumes a mapping defined by:
        # pwm = thrust_component + MIXER_MATRIX * torques, with appropriate scaling.
        thrust = (math.sqrt(target_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm, pos_error, att_error
