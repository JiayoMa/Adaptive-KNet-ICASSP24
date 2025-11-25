"""
MSCKF - Multi-State Constraint Kalman Filter

Python implementation based on msckf_mono.
This module implements the core MSCKF algorithm for visual-inertial odometry.

Reference:
- "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
  by Mourikis and Roumeliotis (ICRA 2007)
"""

import numpy as np
from scipy.linalg import expm, qr
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Optional
from copy import deepcopy

from .types import (
    CameraParams,
    CameraState,
    IMUState,
    IMUReading,
    NoiseParams,
    MSCKFParams,
    FeatureTrack,
    quaternion_multiply,
    quaternion_normalize,
    skew_symmetric,
    omega_matrix,
    build_update_quaternion
)


class MSCKF:
    """
    Multi-State Constraint Kalman Filter for Visual-Inertial Odometry.
    
    This class implements the standard MSCKF algorithm using an Extended Kalman Filter
    for state estimation.
    """
    
    def __init__(self):
        # Parameters
        self.camera: Optional[CameraParams] = None
        self.noise_params: Optional[NoiseParams] = None
        self.msckf_params: Optional[MSCKFParams] = None
        
        # States
        self.imu_state: Optional[IMUState] = None
        self.cam_states: List[CameraState] = []
        self.pruned_states: List[CameraState] = []
        
        # Feature tracking
        self.feature_tracks: List[FeatureTrack] = []
        self.tracked_feature_ids: List[int] = []
        self.last_feature_id: int = 0
        self.num_feature_tracks_residualized: int = 0
        
        # Covariance matrices
        self.imu_covar: np.ndarray = np.zeros((15, 15))
        self.cam_covar: np.ndarray = np.zeros((0, 0))
        self.imu_cam_covar: np.ndarray = np.zeros((15, 0))
        
        # Jacobians
        self.F: np.ndarray = np.zeros((15, 15))
        self.Phi: np.ndarray = np.zeros((15, 15))
        self.G: np.ndarray = np.zeros((15, 12))
        
        # Full state covariance
        self.P: np.ndarray = np.zeros((15, 15))
        
        # Map
        self.map_points: List[np.ndarray] = []
        
        # Chi-squared test table (for gating)
        self.chi_squared_test_table: List[float] = []
        
        # Initial state
        self.pos_init: np.ndarray = np.zeros(3)
        
    def initialize(self, camera: CameraParams, noise_params: NoiseParams, 
                   msckf_params: MSCKFParams, imu_state: IMUState):
        """
        Initialize the MSCKF filter.
        
        Args:
            camera: Camera parameters
            noise_params: Noise parameters
            msckf_params: MSCKF algorithm parameters
            imu_state: Initial IMU state
        """
        self.camera = camera
        self.noise_params = noise_params
        self.msckf_params = msckf_params
        
        # Initialize IMU state
        self.imu_state = imu_state.copy()
        self.pos_init = imu_state.p_I_G.copy()
        self.imu_state.p_I_G_null = imu_state.p_I_G.copy()
        self.imu_state.v_I_G_null = imu_state.v_I_G.copy()
        self.imu_state.q_IG_null = imu_state.q_IG.copy()
        
        # Initialize covariance
        self.imu_covar = noise_params.initial_imu_covar.copy()
        
        # Initialize chi-squared test table (confidence level 0.95)
        from scipy.stats import chi2
        self.chi_squared_test_table = [chi2.ppf(0.05, i) for i in range(1, 100)]
        
        # Reset tracking
        self.last_feature_id = 0
        self.feature_tracks = []
        self.tracked_feature_ids = []
        self.cam_states = []
        self.num_feature_tracks_residualized = 0
        
    def _calc_F(self, imu_state: IMUState, measurement: IMUReading):
        """Calculate the continuous-time state transition matrix F."""
        self.F = np.zeros((15, 15))
        
        omega_hat = measurement.omega - imu_state.b_g
        a_hat = measurement.a - imu_state.b_a
        C_IG = Rotation.from_quat([imu_state.q_IG[1], imu_state.q_IG[2], 
                                   imu_state.q_IG[3], imu_state.q_IG[0]]).as_matrix()
        
        # d(theta)/d(theta)
        self.F[0:3, 0:3] = -skew_symmetric(omega_hat)
        # d(theta)/d(b_g)
        self.F[0:3, 3:6] = -np.eye(3)
        # d(v)/d(theta)
        self.F[6:9, 0:3] = -C_IG.T @ skew_symmetric(a_hat)
        # d(v)/d(b_a)
        self.F[6:9, 9:12] = -C_IG.T
        # d(p)/d(v)
        self.F[12:15, 6:9] = np.eye(3)
        
    def _calc_G(self, imu_state: IMUState):
        """Calculate the noise Jacobian matrix G."""
        self.G = np.zeros((15, 12))
        
        C_IG = Rotation.from_quat([imu_state.q_IG[1], imu_state.q_IG[2],
                                   imu_state.q_IG[3], imu_state.q_IG[0]]).as_matrix()
        
        # Gyroscope noise
        self.G[0:3, 0:3] = -np.eye(3)
        # Gyroscope bias noise
        self.G[3:6, 3:6] = np.eye(3)
        # Accelerometer noise
        self.G[6:9, 6:9] = -C_IG.T
        # Accelerometer bias noise
        self.G[9:12, 9:12] = np.eye(3)
        
    def _propagate_imu_state_rk4(self, imu_state: IMUState, measurement: IMUReading) -> IMUState:
        """
        Propagate IMU state using RK4 integration.
        
        Args:
            imu_state: Current IMU state
            measurement: IMU measurement
            
        Returns:
            Propagated IMU state
        """
        imu_state_prop = imu_state.copy()
        dT = measurement.dT
        
        omega_vec = measurement.omega - imu_state.b_g
        omega_psi = 0.5 * omega_matrix(omega_vec)
        
        # Convert quaternion for RK4 (MSCKF uses [-x, -y, -z, w] format internally)
        y0 = np.array([-imu_state.q_IG[1], -imu_state.q_IG[2], 
                       -imu_state.q_IG[3], imu_state.q_IG[0]])
        
        # RK4 integration for quaternion
        k0 = omega_psi @ y0
        k1 = omega_psi @ (y0 + k0 * dT / 4)
        k2 = omega_psi @ (y0 + (k0 / 8 + k1 / 8) * dT)
        k3 = omega_psi @ (y0 + (-k1 / 2 + k2) * dT)
        k4 = omega_psi @ (y0 + (3 * k0 / 16 + 9 * k3 / 16) * dT)
        k5 = omega_psi @ (y0 + (-3 * k0 / 7 + 2 * k1 / 7 + 12 * k2 / 7 
                                - 12 * k3 / 7 + 8 * k4 / 7) * dT)
        
        y_t = y0 + (7 * k0 + 32 * k2 + 12 * k3 + 32 * k4 + 7 * k5) * dT / 90
        
        # Convert back to [w, x, y, z] format
        q_prop = np.array([y_t[3], -y_t[0], -y_t[1], -y_t[2]])
        q_prop = quaternion_normalize(q_prop)
        
        imu_state_prop.q_IG = q_prop
        
        # Velocity integration
        C_IG = Rotation.from_quat([imu_state.q_IG[1], imu_state.q_IG[2],
                                   imu_state.q_IG[3], imu_state.q_IG[0]]).as_matrix()
        delta_v = (C_IG.T @ (measurement.a - imu_state.b_a) + imu_state.g) * dT
        imu_state_prop.v_I_G = imu_state.v_I_G + delta_v
        
        # Position integration
        imu_state_prop.p_I_G = imu_state.p_I_G + imu_state.v_I_G * dT
        
        return imu_state_prop
        
    def propagate(self, measurement: IMUReading):
        """
        Propagate the IMU state using IMU measurements.
        
        Args:
            measurement: IMU reading with angular velocity and linear acceleration
        """
        # Calculate Jacobians
        self._calc_F(self.imu_state, measurement)
        self._calc_G(self.imu_state)
        
        # Propagate IMU state
        imu_state_prop = self._propagate_imu_state_rk4(self.imu_state, measurement)
        
        # Discrete-time state transition matrix
        self.F *= measurement.dT
        self.Phi = expm(self.F)
        
        # Observability constraints
        R_kk_1 = Rotation.from_quat([self.imu_state.q_IG_null[1], self.imu_state.q_IG_null[2],
                                     self.imu_state.q_IG_null[3], self.imu_state.q_IG_null[0]]).as_matrix()
        
        R_new = Rotation.from_quat([imu_state_prop.q_IG[1], imu_state_prop.q_IG[2],
                                    imu_state_prop.q_IG[3], imu_state_prop.q_IG[0]]).as_matrix()
        
        self.Phi[0:3, 0:3] = R_new @ R_kk_1.T
        
        u = R_kk_1 @ self.imu_state.g
        s = (u.T @ u) ** (-1) * u.T
        
        A1 = self.Phi[6:9, 0:3]
        tmp = self.imu_state.v_I_G_null - imu_state_prop.v_I_G
        w1 = skew_symmetric(tmp) @ self.imu_state.g
        self.Phi[6:9, 0:3] = A1 - np.outer(A1 @ u - w1, s)
        
        A2 = self.Phi[12:15, 0:3]
        tmp = measurement.dT * self.imu_state.v_I_G_null + self.imu_state.p_I_G_null - imu_state_prop.p_I_G
        w2 = skew_symmetric(tmp) @ self.imu_state.g
        self.Phi[12:15, 0:3] = A2 - np.outer(A2 @ u - w2, s)
        
        # Covariance propagation
        Q_d = self.G @ self.noise_params.Q_imu @ self.G.T * measurement.dT
        imu_covar_prop = self.Phi @ (self.imu_covar + Q_d) @ self.Phi.T
        
        # Update state and covariance
        self.imu_state = imu_state_prop
        self.imu_state.q_IG_null = self.imu_state.q_IG.copy()
        self.imu_state.v_I_G_null = self.imu_state.v_I_G.copy()
        self.imu_state.p_I_G_null = self.imu_state.p_I_G.copy()
        
        self.imu_covar = (imu_covar_prop + imu_covar_prop.T) / 2.0
        
        if self.imu_cam_covar.shape[1] > 0:
            self.imu_cam_covar = self.Phi @ self.imu_cam_covar
            
    def augment_state(self, state_id: int, time: float):
        """
        Augment the state with a new camera state.
        
        Args:
            state_id: Unique identifier for the camera state
            time: Timestamp
        """
        self.map_points.clear()
        
        # Compute camera pose from IMU pose
        q_CI = self.camera.q_CI
        R_CI = Rotation.from_quat([q_CI[1], q_CI[2], q_CI[3], q_CI[0]]).as_matrix()
        R_IG = Rotation.from_quat([self.imu_state.q_IG[1], self.imu_state.q_IG[2],
                                   self.imu_state.q_IG[3], self.imu_state.q_IG[0]]).as_matrix()
        
        q_CG = Rotation.from_matrix(R_CI @ R_IG).as_quat()
        q_CG = np.array([q_CG[3], q_CG[0], q_CG[1], q_CG[2]])  # Convert to [w, x, y, z]
        q_CG = quaternion_normalize(q_CG)
        
        cam_state = CameraState(
            q_CG=q_CG,
            p_C_G=self.imu_state.p_I_G + R_IG.T @ self.camera.p_C_I,
            time=time,
            state_id=state_id,
            last_correlated_id=-1
        )
        
        # Build full covariance matrix
        n_cam = len(self.cam_states)
        if n_cam > 0:
            P_size = 15 + 6 * n_cam
            self.P = np.zeros((P_size, P_size))
            self.P[0:15, 0:15] = self.imu_covar
            self.P[0:15, 15:] = self.imu_cam_covar
            self.P[15:, 0:15] = self.imu_cam_covar.T
            self.P[15:, 15:] = self.cam_covar
        else:
            self.P = self.imu_covar.copy()
            
        # Augmentation Jacobian
        J = np.zeros((6, 15 + 6 * n_cam))
        J[0:3, 0:3] = R_CI
        J[3:6, 0:3] = skew_symmetric(R_IG.T @ self.camera.p_C_I)
        J[3:6, 12:15] = np.eye(3)
        
        # Augment covariance
        temp_mat = np.eye(15 + 6 * n_cam + 6, 15 + 6 * n_cam)
        temp_mat[15 + 6 * n_cam:, :] = J
        
        P_aug = temp_mat @ self.P @ temp_mat.T
        P_aug = (P_aug + P_aug.T) / 2.0
        
        # Update states and covariance
        self.cam_states.append(cam_state)
        self.imu_covar = P_aug[0:15, 0:15]
        
        new_cam_size = 6 * len(self.cam_states)
        self.cam_covar = P_aug[15:15+new_cam_size, 15:15+new_cam_size]
        self.imu_cam_covar = P_aug[0:15, 15:15+new_cam_size]
        
    def add_features(self, features: List[np.ndarray], feature_ids: List[int]):
        """
        Add newly detected features to tracking.
        
        Args:
            features: List of 2D feature observations
            feature_ids: Corresponding feature IDs
        """
        for i, feature_id in enumerate(feature_ids):
            if feature_id not in self.tracked_feature_ids:
                track = FeatureTrack(
                    feature_id=feature_id,
                    observations=[features[i].copy()]
                )
                
                cam_state = self.cam_states[-1]
                cam_state.tracked_feature_ids.append(feature_id)
                track.cam_state_indices.append(cam_state.state_id)
                
                self.feature_tracks.append(track)
                self.tracked_feature_ids.append(feature_id)
                
    def measurement_update(self, H_o: np.ndarray, r_o: np.ndarray, R_o: np.ndarray):
        """
        Perform EKF measurement update.
        
        This is the core EKF update step that can be replaced with KalmanNet.
        
        Args:
            H_o: Measurement Jacobian
            r_o: Measurement residual
            R_o: Measurement noise covariance
        """
        if r_o.size == 0:
            return
            
        # Build full covariance matrix
        n_cam = len(self.cam_states)
        P_size = 15 + 6 * n_cam
        P = np.zeros((P_size, P_size))
        P[0:15, 0:15] = self.imu_covar
        if self.cam_covar.size > 0:
            P[0:15, 15:] = self.imu_cam_covar
            P[15:, 0:15] = self.imu_cam_covar.T
            P[15:, 15:] = self.cam_covar
            
        # QR decomposition for numerical stability
        Q_mat, R_mat = qr(H_o, mode='economic')
        
        # Find non-zero rows
        non_zero_rows = np.any(np.abs(R_mat) > 1e-12, axis=1)
        num_non_zero = np.sum(non_zero_rows)
        
        if num_non_zero == 0:
            return
            
        T_H = R_mat[non_zero_rows, :]
        Q_1 = Q_mat[:, non_zero_rows]
        
        r_n = Q_1.T @ r_o
        R_n = Q_1.T @ R_o @ Q_1
        
        # Kalman gain computation (standard EKF)
        S = T_H @ P @ T_H.T + R_n
        K = P @ T_H.T @ np.linalg.inv(S)
        
        # State correction
        delta_x = K @ r_n
        
        # Update IMU state
        self.imu_state.q_IG = quaternion_multiply(
            build_update_quaternion(delta_x[0:3]), 
            self.imu_state.q_IG
        )
        self.imu_state.b_g += delta_x[3:6]
        self.imu_state.v_I_G += delta_x[6:9]
        self.imu_state.b_a += delta_x[9:12]
        self.imu_state.p_I_G += delta_x[12:15]
        
        # Update camera states
        for i, cam_state in enumerate(self.cam_states):
            cam_state.q_CG = quaternion_multiply(
                build_update_quaternion(delta_x[15 + 6*i:18 + 6*i]),
                cam_state.q_CG
            )
            cam_state.q_CG = quaternion_normalize(cam_state.q_CG)
            cam_state.p_C_G += delta_x[18 + 6*i:21 + 6*i]
            
        # Covariance correction
        I_KH = np.eye(P_size) - K @ T_H
        P_corrected = I_KH @ P @ I_KH.T + K @ R_n @ K.T
        P_corrected = (P_corrected + P_corrected.T) / 2.0
        
        self.imu_covar = P_corrected[0:15, 0:15]
        if n_cam > 0:
            self.cam_covar = P_corrected[15:, 15:]
            self.imu_cam_covar = P_corrected[0:15, 15:]
            
    def get_imu_state(self) -> IMUState:
        """Get the current IMU state."""
        return self.imu_state
        
    def get_cam_states(self) -> List[CameraState]:
        """Get all camera states."""
        return self.cam_states
        
    def get_position(self) -> np.ndarray:
        """Get the current position estimate."""
        return self.imu_state.p_I_G.copy()
        
    def get_orientation(self) -> np.ndarray:
        """Get the current orientation as quaternion [w, x, y, z]."""
        return self.imu_state.q_IG.copy()
        
    def get_covariance(self) -> np.ndarray:
        """Get the full state covariance matrix."""
        n_cam = len(self.cam_states)
        P_size = 15 + 6 * n_cam
        P = np.zeros((P_size, P_size))
        P[0:15, 0:15] = self.imu_covar
        if self.cam_covar.size > 0:
            P[0:15, 15:] = self.imu_cam_covar
            P[15:, 0:15] = self.imu_cam_covar.T
            P[15:, 15:] = self.cam_covar
        return P
