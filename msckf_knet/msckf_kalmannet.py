"""
MSCKF-KalmanNet Fusion Module

This module implements the fusion of Multi-State Constraint Kalman Filter (MSCKF)
with KalmanNet, replacing the traditional EKF update with a learned Kalman gain.

The key idea is to use KalmanNet's neural network to estimate the Kalman gain
instead of computing it analytically from the linearized system model.
This can provide better performance when:
1. The system model is uncertain or partially known
2. The noise statistics are non-Gaussian or time-varying
3. The linearization errors are significant
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple
from copy import deepcopy

from .msckf import MSCKF
from .vio_system_model import VIOSystemModel, MSCKFKalmanNetSystemModel
from .types import (
    CameraParams,
    CameraState,
    IMUState,
    IMUReading,
    NoiseParams,
    MSCKFParams,
    build_update_quaternion,
    quaternion_multiply,
    quaternion_normalize
)


class MSCKFKalmanNet(MSCKF):
    """
    MSCKF with KalmanNet for Learned Kalman Gain Estimation.
    
    This class extends the standard MSCKF by replacing the EKF measurement
    update with KalmanNet's learned Kalman gain estimation.
    
    The architecture maintains the MSCKF structure for:
    - IMU state propagation
    - Camera state augmentation
    - Feature tracking
    - Multi-state constraints
    
    But uses KalmanNet for:
    - Kalman gain computation
    - State update based on visual observations
    """
    
    # Covariance update learning rate for KalmanNet
    COVAR_UPDATE_ALPHA = 0.1
    # Covariance reduction factor when update fails
    COVAR_REDUCTION_FACTOR = 0.99
    
    def __init__(self, kalmannet_model: Optional[nn.Module] = None,
                 hypernetwork: Optional[nn.Module] = None,
                 use_adaptive: bool = True):
        """
        Initialize MSCKF-KalmanNet.
        
        Args:
            kalmannet_model: Pre-trained or trainable KalmanNet model
            hypernetwork: Optional hypernetwork for adaptive KalmanNet
            use_adaptive: Whether to use adaptive (hypernetwork) mode
        """
        super().__init__()
        
        # KalmanNet components
        self.kalmannet = kalmannet_model
        self.hypernetwork = hypernetwork
        self.use_adaptive = use_adaptive
        
        # System model for KalmanNet
        self.sys_model: Optional[MSCKFKalmanNetSystemModel] = None
        
        # Device
        self.device = torch.device('cpu')
        
        # State tracking for KalmanNet
        self.knet_initialized = False
        self.m1x_posterior: Optional[torch.Tensor] = None
        self.m1x_posterior_previous: Optional[torch.Tensor] = None
        self.m1x_prior_previous: Optional[torch.Tensor] = None
        self.y_previous: Optional[torch.Tensor] = None
        
        # Noise statistics for hypernetwork
        self.current_sow: Optional[torch.Tensor] = None  # State of World
        
    def set_device(self, device: torch.device):
        """Set the device for computation."""
        self.device = device
        if self.kalmannet is not None:
            self.kalmannet = self.kalmannet.to(device)
        if self.hypernetwork is not None:
            self.hypernetwork = self.hypernetwork.to(device)
            
    def initialize(self, camera: CameraParams, noise_params: NoiseParams,
                   msckf_params: MSCKFParams, imu_state: IMUState):
        """
        Initialize the MSCKF-KalmanNet filter.
        
        Args:
            camera: Camera parameters
            noise_params: Noise parameters
            msckf_params: MSCKF algorithm parameters
            imu_state: Initial IMU state
        """
        # Initialize base MSCKF
        super().initialize(camera, noise_params, msckf_params, imu_state)
        
        # Initialize system model for KalmanNet
        self.sys_model = MSCKFKalmanNetSystemModel(
            max_cam_states=msckf_params.max_cam_states,
            T=100, T_test=100,
            q2=1.0, r2=1.0
        )
        
        # Initialize KalmanNet state
        self._init_kalmannet_state()
        
    def _init_kalmannet_state(self):
        """Initialize KalmanNet internal state from MSCKF IMU state."""
        if self.imu_state is None:
            return
            
        # Convert IMU state to tensor
        # State order: [theta_error(3), b_g(3), v(3), b_a(3), p(3)]
        state = np.concatenate([
            np.zeros(3),  # Orientation error (initially zero)
            self.imu_state.b_g,
            self.imu_state.v_I_G,
            self.imu_state.b_a,
            self.imu_state.p_I_G
        ])
        
        self.m1x_posterior = torch.tensor(state, dtype=torch.float32).view(-1, 1)
        self.m1x_posterior = self.m1x_posterior.to(self.device)
        
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        self.m1x_prior_previous = self.m1x_posterior.clone()
        
        # Initialize observation
        if self.sys_model is not None:
            self.y_previous = self.sys_model.h(self.m1x_posterior.unsqueeze(0)).squeeze(0)
        else:
            self.y_previous = torch.zeros(6, 1, device=self.device)
            
        self.knet_initialized = True
        
    def set_kalmannet_model(self, kalmannet: nn.Module):
        """Set the KalmanNet model."""
        self.kalmannet = kalmannet.to(self.device)
        
    def set_hypernetwork(self, hypernetwork: nn.Module):
        """Set the hypernetwork for adaptive KalmanNet."""
        self.hypernetwork = hypernetwork.to(self.device)
        
    def set_state_of_world(self, sow: torch.Tensor):
        """
        Set the current state of world for adaptive estimation.
        
        Args:
            sow: State of world tensor containing noise statistics
        """
        self.current_sow = sow.to(self.device)
        
    def _update_sys_model_dynamics(self):
        """Update system model dynamics from current MSCKF state."""
        if self.sys_model is None:
            return
            
        # Convert MSCKF Phi to torch tensor
        F_torch = torch.tensor(self.Phi[0:15, 0:15], dtype=torch.float32)
        self.sys_model.update_dynamics(F_torch)
        
        # Update noise covariance from MSCKF
        Q_torch = torch.tensor(
            self.G @ self.noise_params.Q_imu @ self.G.T,
            dtype=torch.float32
        )
        # Truncate to 15x15 for IMU state
        Q_torch = Q_torch[0:15, 0:15]
        self.sys_model.update_noise(Q_torch)
        
    def _kalmannet_estimate_gain(self, innovation: torch.Tensor,
                                  H: torch.Tensor) -> torch.Tensor:
        """
        Use KalmanNet to estimate Kalman gain.
        
        Args:
            innovation: Measurement innovation
            H: Measurement Jacobian
            
        Returns:
            Estimated Kalman gain
        """
        if self.kalmannet is None:
            raise ValueError("KalmanNet model not set")
            
        # Update system model with current dynamics
        self._update_sys_model_dynamics()
        
        # Prepare observation for KalmanNet
        y = self.y_previous + innovation
        
        # Adapt observation dimension if needed
        if y.shape[0] != self.sys_model.n:
            # Truncate or pad to match expected dimension
            target_n = self.sys_model.n
            if y.shape[0] > target_n:
                y = y[:target_n]
            else:
                y = torch.nn.functional.pad(y, (0, 0, 0, target_n - y.shape[0]))
                
        # Batch dimension for KalmanNet
        y_batch = y.unsqueeze(0)  # [1, n, 1]
        
        # Use KalmanNet to estimate state
        if self.use_adaptive and self.hypernetwork is not None and self.current_sow is not None:
            # Adaptive mode: use hypernetwork to generate weights
            weights_knet, weights_cm_gain, weights_cm_shift = None, None, None
            
            # Check if using context modulation
            if hasattr(self.kalmannet, 'use_context_mod') and self.kalmannet.use_context_mod:
                # Generate context modulation weights
                cm_weights = self.hypernetwork(self.current_sow.unsqueeze(0))
                weights_cm_gain, weights_cm_shift = cm_weights
                m1x_est = self.kalmannet(y_batch, 
                                          weights_cm_gain=weights_cm_gain.squeeze(0),
                                          weights_cm_shift=weights_cm_shift.squeeze(0))
            else:
                # Generate full KalmanNet weights
                weights_knet = self.hypernetwork(self.current_sow.unsqueeze(0))
                m1x_est = self.kalmannet(y_batch, weights_knet=weights_knet.squeeze(0))
        else:
            # Standard mode: use pre-trained KalmanNet
            m1x_est = self.kalmannet(y_batch)
            
        # Extract Kalman gain from KalmanNet's internal state
        if hasattr(self.kalmannet, 'KGain'):
            kalman_gain = self.kalmannet.KGain.squeeze(0)  # [m, n]
        else:
            # If KalmanNet doesn't expose gain, compute from state update
            # K = (x_post - x_prior) / innovation
            m1x_prior = self.sys_model.f(self.m1x_posterior.unsqueeze(0)).squeeze(0)
            delta_x = m1x_est.squeeze(0) - m1x_prior
            # Pseudo-inverse to get approximate gain
            kalman_gain = delta_x @ torch.pinverse(innovation.view(1, -1))
            
        # Update KalmanNet state tracking
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        self.m1x_prior_previous = self.sys_model.f(self.m1x_posterior.unsqueeze(0)).squeeze(0)
        self.m1x_posterior = m1x_est.squeeze(0)
        self.y_previous = y
        
        return kalman_gain
        
    def measurement_update_kalmannet(self, H_o: np.ndarray, r_o: np.ndarray, 
                                      R_o: np.ndarray):
        """
        Perform measurement update using KalmanNet.
        
        This replaces the standard EKF measurement update with KalmanNet's
        learned Kalman gain estimation.
        
        Args:
            H_o: Measurement Jacobian (from MSCKF)
            r_o: Measurement residual
            R_o: Measurement noise covariance
        """
        if r_o.size == 0:
            return
            
        if not self.knet_initialized:
            self._init_kalmannet_state()
            
        # Convert numpy to torch
        H_torch = torch.tensor(H_o, dtype=torch.float32, device=self.device)
        r_torch = torch.tensor(r_o, dtype=torch.float32, device=self.device).view(-1, 1)
        R_torch = torch.tensor(R_o, dtype=torch.float32, device=self.device)
        
        # Reduce to IMU state dimension for KalmanNet
        H_imu = H_torch[:, 0:15]  # Only IMU state part
        
        # Get innovation (measurement residual)
        innovation = r_torch
        
        try:
            # Use KalmanNet to estimate Kalman gain
            K_knet = self._kalmannet_estimate_gain(innovation, H_imu)
            
            # Ensure gain dimensions match
            if K_knet.shape[1] < r_torch.shape[0]:
                r_torch = r_torch[:K_knet.shape[1]]
            elif K_knet.shape[1] > r_torch.shape[0]:
                K_knet = K_knet[:, :r_torch.shape[0]]
                
            # Compute state update
            delta_x_imu = K_knet @ r_torch
            delta_x_imu = delta_x_imu.cpu().numpy().flatten()
            
        except (RuntimeError, ValueError, torch.LinAlgError) as e:
            # Fallback to standard EKF if KalmanNet fails
            print(f"KalmanNet failed, falling back to EKF: {e}")
            super().measurement_update(H_o, r_o, R_o)
            return
        except AttributeError as e:
            # Handle case where KalmanNet model structure is unexpected
            print(f"KalmanNet model error, falling back to EKF: {e}")
            super().measurement_update(H_o, r_o, R_o)
            return
            
        # Apply update to IMU state
        # State order: [theta_error(3), b_g(3), v(3), b_a(3), p(3)]
        if len(delta_x_imu) >= 15:
            self.imu_state.q_IG = quaternion_multiply(
                build_update_quaternion(delta_x_imu[0:3]),
                self.imu_state.q_IG
            )
            self.imu_state.b_g += delta_x_imu[3:6]
            self.imu_state.v_I_G += delta_x_imu[6:9]
            self.imu_state.b_a += delta_x_imu[9:12]
            self.imu_state.p_I_G += delta_x_imu[12:15]
            
        # Update camera states using standard EKF (or extended KalmanNet)
        # For now, we only use KalmanNet for IMU state
        n_cam = len(self.cam_states)
        if n_cam > 0 and H_o.shape[1] > 15:
            # Camera state update still uses analytical gain
            self._update_camera_states_ekf(H_o, r_o, R_o)
            
        # Update covariance (simplified - could be improved)
        self._update_covariance_kalmannet(H_o, R_o)
        
    def _update_camera_states_ekf(self, H_o: np.ndarray, r_o: np.ndarray, 
                                   R_o: np.ndarray):
        """
        Update camera states using standard EKF.
        
        This is used when KalmanNet only handles IMU state.
        """
        # Build full covariance matrix
        n_cam = len(self.cam_states)
        P_size = 15 + 6 * n_cam
        P = np.zeros((P_size, P_size))
        P[0:15, 0:15] = self.imu_covar
        if self.cam_covar.size > 0:
            P[0:15, 15:] = self.imu_cam_covar
            P[15:, 0:15] = self.imu_cam_covar.T
            P[15:, 15:] = self.cam_covar
            
        # Camera state part of Jacobian
        H_cam = H_o[:, 15:]
        
        if H_cam.size == 0:
            return
            
        # Simplified camera update using standard EKF formula
        S_cam = H_cam @ P[15:, 15:] @ H_cam.T + R_o
        try:
            K_cam = P[15:, 15:] @ H_cam.T @ np.linalg.inv(S_cam)
        except np.linalg.LinAlgError:
            return
            
        delta_cam = K_cam @ r_o
        
        # Update camera states
        for i, cam_state in enumerate(self.cam_states):
            if 6*i + 6 <= len(delta_cam):
                cam_state.q_CG = quaternion_multiply(
                    build_update_quaternion(delta_cam[6*i:6*i+3]),
                    cam_state.q_CG
                )
                cam_state.q_CG = quaternion_normalize(cam_state.q_CG)
                cam_state.p_C_G += delta_cam[6*i+3:6*i+6]
                
    def _update_covariance_kalmannet(self, H_o: np.ndarray, R_o: np.ndarray):
        """
        Update covariance after KalmanNet update.
        
        This uses a simplified covariance update that accounts for the
        learned Kalman gain.
        """
        # For now, use a conservative covariance reduction
        # A more sophisticated approach would propagate uncertainty through KalmanNet
        
        # Reduce IMU covariance based on observation information
        obs_dim = min(H_o.shape[0], 15)
        H_imu = H_o[:obs_dim, :15]
        
        # Information gain from observation
        try:
            R_inv = np.linalg.inv(R_o[:obs_dim, :obs_dim])
            info_gain = H_imu.T @ R_inv @ H_imu
            
            # Joseph form update (conservative)
            self.imu_covar = (1 - self.COVAR_UPDATE_ALPHA) * self.imu_covar + \
                             self.COVAR_UPDATE_ALPHA * np.linalg.inv(
                np.linalg.inv(self.imu_covar) + info_gain
            )
        except np.linalg.LinAlgError:
            # If inversion fails, reduce covariance by fixed factor
            self.imu_covar *= self.COVAR_REDUCTION_FACTOR
            
        # Ensure symmetry
        self.imu_covar = (self.imu_covar + self.imu_covar.T) / 2.0
        
    def measurement_update(self, H_o: np.ndarray, r_o: np.ndarray, R_o: np.ndarray):
        """
        Perform measurement update.
        
        If KalmanNet is available, uses the learned Kalman gain.
        Otherwise, falls back to standard EKF update.
        
        Args:
            H_o: Measurement Jacobian
            r_o: Measurement residual
            R_o: Measurement noise covariance
        """
        if self.kalmannet is not None:
            self.measurement_update_kalmannet(H_o, r_o, R_o)
        else:
            super().measurement_update(H_o, r_o, R_o)
            
    def init_kalmannet_hidden(self):
        """Initialize KalmanNet hidden states for a new sequence."""
        if self.kalmannet is not None and hasattr(self.kalmannet, 'init_hidden'):
            self.kalmannet.init_hidden()
            
    def get_kalmannet_state(self) -> Optional[torch.Tensor]:
        """Get the current KalmanNet state estimate."""
        return self.m1x_posterior


def create_msckf_kalmannet(kalmannet_path: Optional[str] = None,
                           hypernetwork_path: Optional[str] = None,
                           use_adaptive: bool = True,
                           device: str = 'cpu') -> MSCKFKalmanNet:
    """
    Factory function to create MSCKF-KalmanNet.
    
    Args:
        kalmannet_path: Path to pre-trained KalmanNet weights (not yet implemented)
        hypernetwork_path: Path to pre-trained hypernetwork weights (not yet implemented)
        use_adaptive: Whether to use adaptive mode
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Configured MSCKFKalmanNet instance
        
    Note:
        Loading pre-trained weights is not yet implemented. To use pre-trained models,
        manually load the weights and call set_kalmannet_model() / set_hypernetwork().
    """
    msckf_knet = MSCKFKalmanNet(use_adaptive=use_adaptive)
    msckf_knet.set_device(torch.device(device))
    
    # Note: Weight loading not implemented - users should load weights manually
    # Example:
    #   kalmannet = KalmanNetNN()
    #   kalmannet.load_state_dict(torch.load(kalmannet_path))
    #   msckf_knet.set_kalmannet_model(kalmannet)
        
    return msckf_knet
