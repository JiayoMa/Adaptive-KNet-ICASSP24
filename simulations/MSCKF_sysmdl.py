"""# **Class: System Model for MSCKF (Multi-State Constraint Kalman Filter)**

This module provides the system model for Visual-Inertial Odometry using MSCKF.

MSCKF State Vector:
    - IMU state (16 dimensions): 
        * quaternion (4): orientation
        * position (3): camera position in world frame
        * velocity (3): camera velocity in world frame  
        * gyroscope bias (3): IMU gyro bias
        * accelerometer bias (3): IMU accel bias
    - Camera states (6N dimensions for N poses):
        * quaternion (4) + position (3) for each camera pose

The state evolves according to IMU measurements (gyroscope and accelerometer),
and observations come from feature tracks across multiple camera frames.

This implementation is compatible with the AdaptiveKNet framework.
"""

import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class MSCKFSystemModel:
    """
    MSCKF System Model for Visual-Inertial Odometry
    
    Args:
        n_poses_max: Maximum number of camera poses in sliding window
        T: Sequence length for training
        T_test: Sequence length for testing
        dt: Time step between measurements (seconds)
        device: torch device (cpu or cuda)
    """
    
    def __init__(self, n_poses_max=10, T=100, T_test=100, dt=0.01, device='cpu'):
        
        self.device = torch.device(device)
        self.dt = dt
        self.n_poses_max = n_poses_max
        
        # IMU state dimension (fixed)
        self.m_imu = 16  # quaternion(4) + position(3) + velocity(3) + gyro_bias(3) + accel_bias(3)
        
        # Camera pose dimension per pose
        self.m_pose = 7  # quaternion(4) + position(3)
        
        # Current number of camera poses (dynamic, starts with 0)
        self.n_poses = 0
        
        # Total state dimension (dynamic)
        self.m = self.m_imu  # Will increase as poses are added
        
        # Observation dimension (variable depending on number of features)
        # Each feature provides 2D measurement (u, v) in image plane
        self.n_features = 20  # Number of features tracked
        self.n = 2 * self.n_features  # Total observation dimension
        
        # Sequence length
        self.T = T
        self.T_test = T_test
        
        # Process noise covariance (IMU noise)
        # Gyroscope noise
        self.gyro_noise_std = 0.01  # rad/s
        # Accelerometer noise  
        self.accel_noise_std = 0.1  # m/s^2
        # Gyroscope bias random walk
        self.gyro_bias_noise_std = 0.0001  # rad/s
        # Accelerometer bias random walk
        self.accel_bias_noise_std = 0.001  # m/s^2
        
        # Measurement noise covariance (image feature noise)
        self.feature_noise_std = 1.0  # pixels
        
        # Build covariance matrices
        self._build_covariance_matrices()
        
        # Initialize state
        self.m1x_0 = None
        self.m2x_0 = None
        
        # Priors for KalmanNet (used for initialization)
        self.prior_Q = torch.eye(self.m).to(self.device)
        self.prior_Sigma = torch.zeros(self.m, self.m).to(self.device)
        self.prior_S = torch.eye(self.n).to(self.device)
        
    def _build_covariance_matrices(self):
        """Build process and measurement noise covariance matrices"""
        
        # Process noise affects: gyro(3), accel(3), gyro_bias(3), accel_bias(3)
        Q_imu = torch.zeros(self.m_imu, self.m_imu)
        
        # Gyroscope noise (affects orientation through integration)
        Q_imu[0:3, 0:3] = (self.gyro_noise_std ** 2) * torch.eye(3)
        
        # Accelerometer noise (affects velocity through integration)
        Q_imu[7:10, 7:10] = (self.accel_noise_std ** 2) * torch.eye(3)
        
        # Gyroscope bias random walk
        Q_imu[10:13, 10:13] = (self.gyro_bias_noise_std ** 2) * torch.eye(3)
        
        # Accelerometer bias random walk
        Q_imu[13:16, 13:16] = (self.accel_bias_noise_std ** 2) * torch.eye(3)
        
        self.Q = Q_imu.to(self.device)
        self.q2 = torch.tensor(self.gyro_noise_std ** 2).to(self.device)
        
        # Measurement noise (pixel coordinates)
        R_features = (self.feature_noise_std ** 2) * torch.eye(self.n)
        self.R = R_features.to(self.device)
        self.r2 = torch.tensor(self.feature_noise_std ** 2).to(self.device)
    
    def f(self, x, u=None):
        """
        State transition function (IMU propagation)
        
        Args:
            x: State tensor [batch_size, m, 1]
            u: IMU measurements [batch_size, 6, 1] (gyro + accel)
            
        Returns:
            x_next: Predicted state [batch_size, m, 1]
        """
        batch_size = x.shape[0]
        
        # If no IMU input provided, use zero (for testing)
        if u is None:
            u = torch.zeros(batch_size, 6, 1).to(x.device)
        
        x_next = x.clone()
        
        # Extract IMU state components
        # q = x[:, 0:4, :]  # quaternion
        p = x[:, 4:7, :]    # position
        v = x[:, 7:10, :]   # velocity
        bg = x[:, 10:13, :] # gyro bias
        ba = x[:, 13:16, :] # accel bias
        
        # Extract IMU measurements
        gyro = u[:, 0:3, :] - bg
        accel = u[:, 3:6, :] - ba
        
        # Simplified IMU integration (first-order Euler)
        # In practice, would use proper quaternion integration
        
        # Update velocity (with gravity compensation)
        g = torch.tensor([0, 0, -9.81]).view(1, 3, 1).to(x.device)
        v_next = v + (accel + g) * self.dt
        
        # Update position
        p_next = p + v * self.dt + 0.5 * (accel + g) * (self.dt ** 2)
        
        # Update IMU state
        x_next[:, 4:7, :] = p_next
        x_next[:, 7:10, :] = v_next
        
        # Biases evolve with random walk (handled by process noise)
        # Camera poses remain fixed once added
        
        return x_next
    
    def h(self, x):
        """
        Observation function (projects 3D features to 2D image)
        
        Args:
            x: State tensor [batch_size, m, 1]
            
        Returns:
            y: Observation tensor [batch_size, n, 1]
        """
        batch_size = x.shape[0]
        
        # Simplified projection model
        # In practice, would project 3D map points using camera poses
        
        # Extract camera position from current IMU state
        p_cam = x[:, 4:7, :]  # position
        
        # Simulate feature observations with simple linear projection
        # This is simplified - real MSCKF would use proper camera projection
        H_simplified = torch.randn(self.n, self.m).to(x.device) * 0.1
        H_simplified = H_simplified.unsqueeze(0).expand(batch_size, -1, -1)
        
        y = torch.bmm(H_simplified, x)
        
        return y
    
    def InitSequence(self, m1x_0, m2x_0):
        """Initialize sequence with initial state"""
        self.m1x_0 = m1x_0.to(self.device)
        self.m2x_0 = m2x_0.to(self.device)
        self.x_prev = self.m1x_0
        
    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):
        """Initialize batched sequence"""
        self.m1x_0_batch = m1x_0_batch.to(self.device)
        self.m2x_0_batch = m2x_0_batch.to(self.device)
        self.x_prev = self.m1x_0_batch
    
    def UpdateCovariance_Matrix(self, Q, R):
        """Update covariance matrices"""
        self.Q = Q.to(self.device)
        self.R = R.to(self.device)
    
    def add_camera_pose(self):
        """
        Add a new camera pose to the state vector
        This is called when a new keyframe is added to the MSCKF sliding window
        """
        if self.n_poses < self.n_poses_max:
            self.n_poses += 1
            self.m = self.m_imu + self.m_pose * self.n_poses
            
            # Update covariance dimensions
            Q_new = torch.zeros(self.m, self.m).to(self.device)
            Q_new[:self.Q.shape[0], :self.Q.shape[1]] = self.Q
            self.Q = Q_new
            
            # Update priors
            self.prior_Q = torch.eye(self.m).to(self.device)
            self.prior_Sigma = torch.zeros(self.m, self.m).to(self.device)
    
    def remove_camera_pose(self, pose_idx):
        """
        Remove a camera pose from the state vector
        This is called when marginalizing out old poses
        """
        if self.n_poses > 0 and pose_idx < self.n_poses:
            self.n_poses -= 1
            self.m = self.m_imu + self.m_pose * self.n_poses
            
            # Update covariance dimensions
            Q_new = torch.zeros(self.m, self.m).to(self.device)
            # Copy relevant blocks (excluding removed pose)
            start_idx = self.m_imu + pose_idx * self.m_pose
            end_idx = start_idx + self.m_pose
            
            if pose_idx == 0:
                Q_new[:self.m, :self.m] = self.Q[end_idx:, end_idx:]
            else:
                Q_new[:start_idx, :start_idx] = self.Q[:start_idx, :start_idx]
                if end_idx < self.Q.shape[0]:
                    remaining = self.Q.shape[0] - end_idx
                    Q_new[start_idx:start_idx+remaining, start_idx:start_idx+remaining] = \
                        self.Q[end_idx:, end_idx:]
            
            self.Q = Q_new
            
            # Update priors
            self.prior_Q = torch.eye(self.m).to(self.device)
            self.prior_Sigma = torch.zeros(self.m, self.m).to(self.device)
    
    def GenerateBatch(self, args, size, T, randomInit=False):
        """
        Generate batch of MSCKF sequences for training
        
        Args:
            args: Configuration arguments
            size: Batch size
            T: Sequence length
            randomInit: Whether to use random initial conditions
        """
        
        # Initialize conditions
        if randomInit:
            # Random initialization around hover condition
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            for i in range(size):
                # Quaternion (identity + noise)
                q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
                # Position (small random)
                p0 = torch.randn(3) * 0.1
                # Velocity (small random)
                v0 = torch.randn(3) * 0.1
                # Biases (small random)
                bg0 = torch.randn(3) * 0.001
                ba0 = torch.randn(3) * 0.01
                
                init_state = torch.cat([q0, p0, v0, bg0, ba0])
                self.m1x_0_rand[i, :self.m_imu, 0] = init_state
            
            self.Init_batched_sequence(self.m1x_0_rand, 
                                      torch.eye(self.m).unsqueeze(0).expand(size, -1, -1))
        else:
            # Fixed initialization (hover condition)
            q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
            p0 = torch.zeros(3)
            v0 = torch.zeros(3)
            bg0 = torch.zeros(3)
            ba0 = torch.zeros(3)
            
            init_state = torch.cat([q0, p0, v0, bg0, ba0])
            m1x_0 = torch.zeros(self.m, 1)
            m1x_0[:self.m_imu, 0] = init_state
            
            initConditions = m1x_0.view(1, self.m, 1).expand(size, -1, -1)
            self.Init_batched_sequence(initConditions, 
                                      torch.eye(self.m).unsqueeze(0).expand(size, -1, -1))
        
        # Allocate arrays
        self.Input = torch.zeros(size, self.n, T).to(self.device)
        self.Target = torch.zeros(size, self.m, T).to(self.device)
        
        # Generate IMU measurements (simplified)
        u_seq = torch.zeros(size, 6, T).to(self.device)
        # Add some motion pattern (circular motion for testing)
        t = torch.arange(T).float().to(self.device) * self.dt
        u_seq[:, 0, :] = 0.1 * torch.sin(2 * np.pi * 0.5 * t)  # gyro_x
        u_seq[:, 3, :] = 0.5 * torch.cos(2 * np.pi * 0.5 * t)  # accel_x
        u_seq[:, 4, :] = 0.5 * torch.sin(2 * np.pi * 0.5 * t)  # accel_y
        
        # Set initial state
        self.x_prev = self.m1x_0_batch
        xt = self.x_prev
        
        # Generate sequences
        for t_idx in range(T):
            # State propagation with IMU
            ut = u_seq[:, :, t_idx:t_idx+1]
            xt_no_noise = self.f(self.x_prev, ut)
            
            # Add process noise
            if not torch.equal(self.Q, torch.zeros_like(self.Q)):
                mean = torch.zeros(size, self.m)
                # Use only IMU part of Q for noise generation
                Q_imu_part = self.Q[:self.m_imu, :self.m_imu]
                eq = torch.zeros(size, self.m, 1).to(self.device)
                eq[:, :self.m_imu, :] = torch.randn(size, self.m_imu, 1).to(self.device) * \
                                        torch.sqrt(torch.diag(Q_imu_part)).view(1, -1, 1)
                xt = xt_no_noise + eq
            else:
                xt = xt_no_noise
            
            # Generate observations
            yt_no_noise = self.h(xt)
            
            # Add measurement noise
            if not torch.equal(self.R, torch.zeros_like(self.R)):
                er = torch.randn(size, self.n, 1).to(self.device) * self.feature_noise_std
                yt = yt_no_noise + er
            else:
                yt = yt_no_noise
            
            # Save to arrays
            self.Target[:, :, t_idx] = torch.squeeze(xt, 2)
            self.Input[:, :, t_idx] = torch.squeeze(yt, 2)
            
            # Update previous state
            self.x_prev = xt
