import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
import control as ct

class LQRControl(BaseControl):
    """LQR Controller with quaternion dynamics and gain scheduling."""
    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]:
            print("[ERROR] LQRControl requires DroneModel.CF2X or DroneModel.CF2P or DroneModel.RACE")
            exit()
        self.Ixx = self._getURDFParameter("ixx")
        self.Iyy = self._getURDFParameter("iyy")
        self.Izz = self._getURDFParameter("izz")
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.mass = self._getURDFParameter("m")
        self.l = self._getURDFParameter("arm")
        self.g = g
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
                [0, -1, -1],
                [+1,  0,  1],
                [0,  1, -1],
                [-1,  0,  1]
            ])

        self.K = self._compute_K()  # Compute the LQR gain matrix K.
        self.reset()

    def _compute_K(self):
        # Define the linearized state-space model.
        # State: x = [x, y, z, xdot, ydot, zdot, φ, θ, ψ, p, q, r]ᵀ.
        # Under hover conditions the linearized equations are:
        #   ṗ = v       (position dynamics)
        #   v̇ = [g·θ; -g·φ; δU/m]   (coupling between small roll/pitch and horizontal acceleration;
        #                              δU is the thrust deviation from mg)
        #   Attitude: quaternion vector part error is approximated by Euler angles,
        #         and rate = 0.5*angular velocity.
        #
        # A is then a 12x12 matrix:
        A = np.zeros((12, 12))
        # Position derivative
        A[0, 3] = 1
        A[1, 4] = 1
        A[2, 5] = 1
        # Linearized translational dynamics (small-angle approximations):
        A[3, 7] = self.g   # x acceleration from pitch error (θ)
        A[4, 6] = -self.g  # y acceleration from roll error (φ)
        # Attitude kinematics: for small errors, quaternion (or Euler angle) rate ~0.5 angular rate.
        A[6:9, 9:12] = 0.5 * np.eye(3)
        # B is the input matrix (12x4) where the inputs are: 
        #   u = [δU, τₓ, τ_y, τ_z]ᵀ.
        B = np.zeros((12, 4))
        # Thrust deviation affects z-acceleration.
        B[5, 0] = 1 / self.mass
        # Torques affect angular acceleration. Compute inverse inertia:
        J_inv = np.linalg.inv(self.J)
        # Map the torque inputs to angular acceleration in the last three states.
        B[9:12, 1:4] = J_inv

        # Q and R matrices.
        # (Tuning parameters: you can adjust these for your specific application.)
        Q = np.diag([100, 100, 100,      # position error weight
                    10,  10,  10,        # velocity error weight
                    500, 500, 500,       # attitude error weight (using Euler equivalent for small errors)
                    50,  50,  50])       # angular rate error weight
        R = np.diag([1, 1, 1, 1])          # control input effort

        # Compute the LQR gain matrix K
        K, _, _ = ct.lqr(A, B, Q, R)
        return K

    def reset(self):
        super().reset()

    def computeControl(self,
                    control_timestep,
                    cur_pos,
                    cur_quat,       # current attitude as quaternion [x, y, z, w]
                    cur_vel,
                    cur_ang_vel,    # expressed in inertial frame; convert to body frame
                    target_pos,
                    target_rpy=np.zeros(3),
                    target_vel=np.zeros(3),
                    target_rpy_rates=np.zeros(3)):

        self.control_counter += 1

        # Convert current angular velocity to the body frame.
        # (Assume cur_quat is in [x, y, z, w] format.)
        r_current = Rotation.from_quat(cur_quat)
        cur_ang_vel_body = r_current.inv().apply(cur_ang_vel)

        # Build the state error vector.
        # 1. Position error.
        pos_error = target_pos - cur_pos
        # 2. Velocity error.
        vel_error = target_vel - cur_vel

        # 3. Attitude error using quaternions.
        # First compute desired quaternion from target Euler angles.
        r_des = Rotation.from_euler('xyz', target_rpy)
        q_des = r_des.as_quat()  # [x, y, z, w]
        # Compute the quaternion error as q_err = q_des ⊗ (q_current)^-1.
        q_err = r_des * r_current.inv()
        q_err_quat = q_err.as_quat()  # in [x, y, z, w]
        # Use only the vector part for feedback.
        att_error = q_err_quat[:3]

        # 4. Angular velocity error.
        ang_vel_error = target_rpy_rates - cur_ang_vel_body

        # Form full 12-dimensional state error:
        # x_error = [pos_error; vel_error; att_error; ang_vel_error]
        X_error = np.hstack((pos_error, vel_error, att_error, ang_vel_error))

        # Compute control input correction using LQR law.
        # u = [δU, τₓ, τ_y, τ_z]
        u = -np.dot(self.K, X_error)

        # The net thrust must overcome gravity so add mg.
        # u[0] is the thrust deviation from the hover thrust.
        thrust = self.mass * self.g + u[0]
        # Limit thrust to be nonnegative.
        thrust = np.maximum(0, thrust)

        # Extract desired torques from the control law.
        target_torques = u[1:4]
        # Optionally, clip torques to actuator limits.
        target_torques = np.clip(target_torques, -3200, 3200)

        # For debugging.
        # print(f"Pos error: {pos_error}, Attitude error: {att_error}")
        # print(f"Control thrust (N): {thrust}, Control torques (N·m): {target_torques}")

        # Convert thrust and torques into motor commands.
        # Here the mixer matrix distributes the required thrust and torques into individual propeller commands.
        # First, compute a quantity that relates thrust to PWM scaling.
        # (Typically, these conversions require solving the mixer equation and applying inverse transforms.)
        # For simplicity, we assume:
        thrust_pwm = (math.sqrt(thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        pwm = thrust_pwm + np.dot(self.MIXER_MATRIX, target_torques)
        # Enforce PWM limits.
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        # Convert back to RPM.
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        return rpm, pos_error, att_error
