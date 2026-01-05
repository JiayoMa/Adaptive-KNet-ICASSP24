"""# **MSCKF-AdaptiveKNet Dimension Adapter**

This module handles dimension mapping between MSCKF state space and AdaptiveKNet.

Key Challenges:
1. MSCKF has variable state dimension (as camera poses are added/removed)
2. AdaptiveKNet expects fixed dimensions during training
3. Need to map high-dimensional MSCKF state to KNet-compatible dimensions

Solution Strategy:
- Use a fixed "base" dimension for KalmanNet training
- Map variable MSCKF state to fixed dimension via:
  * Padding: Pad smaller states to maximum dimension
  * Projection: Project high-dim state to lower-dim representation
  * Splitting: Separate IMU state from camera poses, process independently

The adapter provides:
- Dimension compatibility layer
- State space transformations
- Kalman gain dimension handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCKFDimensionAdapter(nn.Module):
    """
    Adapter to handle dimension mismatch between MSCKF and AdaptiveKNet
    
    Args:
        msckf_m_max: Maximum MSCKF state dimension
        msckf_n: MSCKF observation dimension  
        knet_m: Fixed state dimension for KalmanNet
        knet_n: Fixed observation dimension for KalmanNet
        adaptation_method: Method to adapt dimensions ('pad', 'project', or 'split')
    """
    
    def __init__(self, msckf_m_max, msckf_n, knet_m, knet_n, 
                 adaptation_method='project', device='cpu'):
        super(MSCKFDimensionAdapter, self).__init__()
        
        self.msckf_m_max = msckf_m_max
        self.msckf_n = msckf_n
        self.knet_m = knet_m
        self.knet_n = knet_n
        self.adaptation_method = adaptation_method
        self.device = torch.device(device)
        
        # Build adaptation layers based on method
        if adaptation_method == 'project':
            # Learnable projection matrices
            self.state_encoder = nn.Linear(msckf_m_max, knet_m, bias=False)
            self.state_decoder = nn.Linear(knet_m, msckf_m_max, bias=False)
            self.obs_encoder = nn.Linear(msckf_n, knet_n, bias=False)
            self.obs_decoder = nn.Linear(knet_n, msckf_n, bias=False)
            
            # Initialize with truncated identity
            with torch.no_grad():
                # State encoder initialization
                min_state_dim = min(knet_m, msckf_m_max)
                if knet_m <= msckf_m_max:
                    self.state_encoder.weight.data[:, :knet_m] = torch.eye(knet_m)
                else:
                    self.state_encoder.weight.data[:min_state_dim, :min_state_dim] = torch.eye(min_state_dim)
                
                # State decoder initialization    
                if msckf_m_max <= knet_m:
                    self.state_decoder.weight.data[:msckf_m_max, :msckf_m_max] = torch.eye(msckf_m_max)
                else:
                    self.state_decoder.weight.data[:min_state_dim, :min_state_dim] = torch.eye(min_state_dim)
                
                # Observation encoder initialization
                min_obs_dim = min(knet_n, msckf_n)
                if knet_n <= msckf_n:
                    self.obs_encoder.weight.data[:, :knet_n] = torch.eye(knet_n)
                else:
                    self.obs_encoder.weight.data[:min_obs_dim, :min_obs_dim] = torch.eye(min_obs_dim)
                
                # Observation decoder initialization    
                if msckf_n <= knet_n:
                    self.obs_decoder.weight.data[:msckf_n, :msckf_n] = torch.eye(msckf_n)
                else:
                    self.obs_decoder.weight.data[:min_obs_dim, :min_obs_dim] = torch.eye(min_obs_dim)
        
        elif adaptation_method == 'split':
            # Split IMU state (16-dim) from camera poses
            self.m_imu = 16
            # Process only IMU state with KalmanNet
            assert knet_m >= self.m_imu, "KNet state dim must accommodate IMU state"
            
    def adapt_state_to_knet(self, x_msckf, msckf_m_current):
        """
        Adapt MSCKF state to KalmanNet-compatible dimension
        
        Args:
            x_msckf: MSCKF state [batch_size, msckf_m_current, 1]
            msckf_m_current: Current MSCKF state dimension
            
        Returns:
            x_knet: KalmanNet state [batch_size, knet_m, 1]
            adaptation_info: Dictionary with adaptation metadata
        """
        batch_size = x_msckf.shape[0]
        
        if self.adaptation_method == 'pad':
            # Zero-padding method
            x_knet = torch.zeros(batch_size, self.knet_m, 1).to(self.device)
            x_knet[:, :msckf_m_current, :] = x_msckf
            adaptation_info = {'method': 'pad', 'original_dim': msckf_m_current}
            
        elif self.adaptation_method == 'project':
            # Projection method using learned linear transformation
            x_flat = x_msckf.squeeze(2)  # [batch_size, msckf_m_current]
            
            # Pad to max dimension if needed
            if msckf_m_current < self.msckf_m_max:
                x_padded = torch.zeros(batch_size, self.msckf_m_max).to(self.device)
                x_padded[:, :msckf_m_current] = x_flat
                x_flat = x_padded
            
            # Project to KNet dimension
            x_proj = self.state_encoder(x_flat)  # [batch_size, knet_m]
            x_knet = x_proj.unsqueeze(2)  # [batch_size, knet_m, 1]
            
            adaptation_info = {
                'method': 'project',
                'original_dim': msckf_m_current,
                'encoder': self.state_encoder
            }
            
        elif self.adaptation_method == 'split':
            # Split method: use only IMU state
            x_imu = x_msckf[:, :self.m_imu, :]  # Extract IMU state
            x_knet = torch.zeros(batch_size, self.knet_m, 1).to(self.device)
            x_knet[:, :self.m_imu, :] = x_imu
            
            adaptation_info = {
                'method': 'split',
                'original_dim': msckf_m_current,
                'camera_poses': x_msckf[:, self.m_imu:, :]  # Store camera poses separately
            }
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
        
        return x_knet, adaptation_info
    
    def adapt_state_from_knet(self, x_knet, adaptation_info):
        """
        Adapt KalmanNet state back to MSCKF dimension
        
        Args:
            x_knet: KalmanNet state [batch_size, knet_m, 1]
            adaptation_info: Metadata from adapt_state_to_knet
            
        Returns:
            x_msckf: MSCKF state [batch_size, msckf_m_current, 1]
        """
        batch_size = x_knet.shape[0]
        msckf_m_current = adaptation_info['original_dim']
        
        if adaptation_info['method'] == 'pad':
            # Extract original dimension
            x_msckf = x_knet[:, :msckf_m_current, :]
            
        elif adaptation_info['method'] == 'project':
            # Inverse projection
            x_flat = x_knet.squeeze(2)  # [batch_size, knet_m]
            x_reconstructed = self.state_decoder(x_flat)  # [batch_size, msckf_m_max]
            
            # Extract to current dimension
            x_msckf = x_reconstructed[:, :msckf_m_current].unsqueeze(2)
            
        elif adaptation_info['method'] == 'split':
            # Reconstruct full state: IMU + camera poses
            x_imu_updated = x_knet[:, :self.m_imu, :]
            camera_poses = adaptation_info['camera_poses']
            
            x_msckf = torch.cat([x_imu_updated, camera_poses], dim=1)
        
        return x_msckf
    
    def adapt_observation_to_knet(self, y_msckf):
        """
        Adapt MSCKF observation to KalmanNet-compatible dimension
        
        Args:
            y_msckf: MSCKF observation [batch_size, msckf_n, 1]
            
        Returns:
            y_knet: KalmanNet observation [batch_size, knet_n, 1]
        """
        batch_size = y_msckf.shape[0]
        
        if self.msckf_n == self.knet_n:
            return y_msckf
        
        if self.adaptation_method in ['pad', 'split']:
            # Simple padding/truncation
            if self.knet_n > self.msckf_n:
                y_knet = torch.zeros(batch_size, self.knet_n, 1).to(self.device)
                y_knet[:, :self.msckf_n, :] = y_msckf
            else:
                y_knet = y_msckf[:, :self.knet_n, :]
                
        elif self.adaptation_method == 'project':
            # Learned projection
            y_flat = y_msckf.squeeze(2)
            y_proj = self.obs_encoder(y_flat)
            y_knet = y_proj.unsqueeze(2)
        
        return y_knet
    
    def adapt_kalman_gain(self, KG_knet, adaptation_info):
        """
        Adapt Kalman gain from KalmanNet dimension to MSCKF dimension
        
        Args:
            KG_knet: Kalman gain from KalmanNet [batch_size, knet_m, knet_n]
            adaptation_info: Metadata from adapt_state_to_knet
            
        Returns:
            KG_msckf: Kalman gain for MSCKF [batch_size, msckf_m_current, msckf_n]
        """
        batch_size = KG_knet.shape[0]
        msckf_m_current = adaptation_info['original_dim']
        
        if adaptation_info['method'] == 'pad':
            # Extract relevant portion
            KG_msckf = KG_knet[:, :msckf_m_current, :self.msckf_n]
            
        elif adaptation_info['method'] == 'project':
            # Transform Kalman gain through projection matrices
            # K_msckf = D_state @ K_knet @ E_obs^T
            # where D_state is state decoder, E_obs is obs encoder
            
            # First adapt observation dimension: K @ E_obs^T
            obs_adapt = self.obs_encoder.weight.t()  # [msckf_n, knet_n]
            obs_adapt_batch = obs_adapt.unsqueeze(0).expand(batch_size, -1, -1)
            KG_obs_adapted = torch.bmm(KG_knet, obs_adapt_batch.transpose(1, 2))
            # [batch_size, knet_m, msckf_n]
            
            # Then adapt state dimension: D_state @ K
            state_adapt = self.state_decoder.weight  # [msckf_m_max, knet_m]
            state_adapt_batch = state_adapt.unsqueeze(0).expand(batch_size, -1, -1)
            KG_full = torch.bmm(state_adapt_batch, KG_obs_adapted)
            # [batch_size, msckf_m_max, msckf_n]
            
            # Extract current dimension
            KG_msckf = KG_full[:, :msckf_m_current, :]
            
        elif adaptation_info['method'] == 'split':
            # Reconstruct gain: IMU gain from KNet, zero for camera poses
            KG_msckf = torch.zeros(batch_size, msckf_m_current, self.msckf_n).to(self.device)
            # Use gain for IMU state
            KG_msckf[:, :self.m_imu, :] = KG_knet[:, :self.m_imu, :self.msckf_n]
            # Camera pose gains remain zero (no direct update from observations)
        
        return KG_msckf
    
    def get_knet_system_functions(self, msckf_model):
        """
        Create KalmanNet-compatible system functions (f, h) from MSCKF model
        
        Args:
            msckf_model: MSCKF system model
            
        Returns:
            f_knet: State transition function for KalmanNet
            h_knet: Observation function for KalmanNet
        """
        
        def f_knet(x_knet, adaptation_info=None):
            """KalmanNet state transition"""
            # Convert to MSCKF dimension
            if adaptation_info is None:
                # Use default (assume max dimension)
                adaptation_info = {
                    'method': self.adaptation_method,
                    'original_dim': self.msckf_m_max
                }
            
            x_msckf = self.adapt_state_from_knet(x_knet, adaptation_info)
            
            # Apply MSCKF dynamics
            x_msckf_next = msckf_model.f(x_msckf)
            
            # Convert back to KalmanNet dimension
            x_knet_next, _ = self.adapt_state_to_knet(x_msckf_next, adaptation_info['original_dim'])
            
            return x_knet_next
        
        def h_knet(x_knet, adaptation_info=None):
            """KalmanNet observation function"""
            # Convert to MSCKF dimension
            if adaptation_info is None:
                adaptation_info = {
                    'method': self.adaptation_method,
                    'original_dim': self.msckf_m_max
                }
            
            x_msckf = self.adapt_state_from_knet(x_knet, adaptation_info)
            
            # Apply MSCKF observation
            y_msckf = msckf_model.h(x_msckf)
            
            # Convert to KalmanNet dimension
            y_knet = self.adapt_observation_to_knet(y_msckf)
            
            return y_knet
        
        return f_knet, h_knet


class AdaptiveKNetMSCKF(nn.Module):
    """
    Wrapper combining AdaptiveKNet with MSCKF through dimension adaptation
    
    This class integrates:
    1. MSCKF system model (variable dimension)
    2. Dimension adapter (handles dimension mismatch)
    3. AdaptiveKNet (fixed dimension, learns Kalman gain)
    """
    
    def __init__(self, msckf_model, knet_model, adapter, device='cpu'):
        super(AdaptiveKNetMSCKF, self).__init__()
        
        self.msckf_model = msckf_model
        self.knet_model = knet_model
        self.adapter = adapter
        self.device = torch.device(device)
        
        # Store adaptation info for sequence
        self.adaptation_info_history = []
        
    def InitSequence(self, M1_0, T):
        """
        Initialize filtering sequence
        
        Args:
            M1_0: Initial state in MSCKF dimension [batch_size, msckf_m, 1]
            T: Sequence length
        """
        self.T = T
        batch_size = M1_0.shape[0]
        msckf_m_current = M1_0.shape[1]
        
        # Adapt initial state to KalmanNet dimension
        M1_0_knet, adaptation_info = self.adapter.adapt_state_to_knet(M1_0, msckf_m_current)
        
        # Initialize KalmanNet
        self.knet_model.InitSequence(M1_0_knet, T)
        
        # Store adaptation info
        self.current_adaptation_info = adaptation_info
        self.adaptation_info_history = []
        
        # Store MSCKF initial state
        self.m1x_posterior_msckf = M1_0.to(self.device)
    
    def forward(self, y_msckf):
        """
        One step of MSCKF filtering with AdaptiveKNet
        
        Args:
            y_msckf: Observation in MSCKF dimension [batch_size, msckf_n, 1]
            
        Returns:
            x_posterior_msckf: Posterior state in MSCKF dimension [batch_size, msckf_m, 1]
        """
        
        # Adapt observation to KalmanNet dimension
        y_knet = self.adapter.adapt_observation_to_knet(y_msckf)
        
        # Run KalmanNet step
        x_posterior_knet = self.knet_model.forward(y_knet)
        
        # Adapt state back to MSCKF dimension
        x_posterior_msckf = self.adapter.adapt_state_from_knet(
            x_posterior_knet, self.current_adaptation_info
        )
        
        # Update stored state
        self.m1x_posterior_msckf = x_posterior_msckf
        
        # Store adaptation info for this step
        self.adaptation_info_history.append(self.current_adaptation_info.copy())
        
        return x_posterior_msckf
    
    def get_kalman_gain_msckf(self):
        """
        Get Kalman gain in MSCKF dimension
        
        Returns:
            KG_msckf: Kalman gain [batch_size, msckf_m, msckf_n]
        """
        # Get KalmanNet's Kalman gain
        KG_knet = self.knet_model.KGain  # [batch_size, knet_m, knet_n]
        
        # Adapt to MSCKF dimension
        KG_msckf = self.adapter.adapt_kalman_gain(KG_knet, self.current_adaptation_info)
        
        return KG_msckf
