"""# **Quick Demo: MSCKF-AdaptiveKNet Integration**

This is a minimal example demonstrating the MSCKF-AdaptiveKNet integration.
Run this to quickly verify the installation and see the system in action.

Usage:
    python demo_msckf_simple.py
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulations.MSCKF_sysmdl import MSCKFSystemModel
from msckf_integration.dimension_adapter import MSCKFDimensionAdapter
from mnets.KNet_mnet import KalmanNetNN
import simulations.config as config


def demo_msckf_adaptiveknet():
    """
    Simple demonstration of MSCKF-AdaptiveKNet integration
    """
    
    print("="*80)
    print("MSCKF-AdaptiveKNet Quick Demo")
    print("="*80)
    
    device = torch.device('cpu')
    
    # ========== Step 1: Create MSCKF System Model ==========
    print("\n[1/5] Creating MSCKF System Model...")
    
    msckf_model = MSCKFSystemModel(
        n_poses_max=5,      # Maximum 5 camera poses
        T=50,               # Short sequences for demo
        T_test=50,
        dt=0.01,            # 100 Hz
        device=device
    )
    
    print(f"  ✓ MSCKF state dimension: {msckf_model.m}")
    print(f"  ✓ MSCKF observation dimension: {msckf_model.n}")
    
    # ========== Step 2: Generate Sample Data ==========
    print("\n[2/5] Generating sample MSCKF trajectories...")
    
    args = config.general_settings()
    args.use_cuda = False
    args.proc_noise_distri = 'normal'
    args.meas_noise_distri = 'normal'
    
    # Generate 10 training sequences
    msckf_model.GenerateBatch(args, size=10, T=50, randomInit=False)
    train_input = msckf_model.Input
    train_target = msckf_model.Target
    
    print(f"  ✓ Generated {train_input.shape[0]} sequences")
    print(f"  ✓ Input shape: {train_input.shape}")
    print(f"  ✓ Target shape: {train_target.shape}")
    
    # ========== Step 3: Create Dimension Adapter ==========
    print("\n[3/5] Setting up Dimension Adapter...")
    
    # KalmanNet will work with IMU state only (16 dimensions)
    knet_m = 16
    knet_n = 40  # Same as observation dimension
    
    adapter = MSCKFDimensionAdapter(
        msckf_m_max=msckf_model.m,
        msckf_n=msckf_model.n,
        knet_m=knet_m,
        knet_n=knet_n,
        adaptation_method='split',  # Use splitting method for simplicity
        device=device
    )
    
    print(f"  ✓ Adaptation method: split")
    print(f"  ✓ MSCKF -> KNet: {msckf_model.m} -> {knet_m}")
    
    # ========== Step 4: Initialize KalmanNet ==========
    print("\n[4/5] Initializing KalmanNet...")
    
    class SimpleSystemModel:
        """Simple wrapper for KalmanNet initialization"""
        def __init__(self, m, n, device):
            self.m = m
            self.n = n
            self.device = device
            self.prior_Q = torch.eye(m)
            self.prior_Sigma = torch.zeros(m, m)
            self.prior_S = torch.eye(n)
            
        def f(self, x):
            """Identity state transition"""
            return x
        
        def h(self, x):
            """Observation function: project to observation space"""
            batch_size = x.shape[0]
            # Create a simple linear observation matrix
            H = torch.randn(self.n, self.m).to(self.device) * 0.1
            H = H.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bmm(H, x)
    
    knet_sys_model = SimpleSystemModel(knet_m, knet_n, device)
    
    knet = KalmanNetNN()
    knet_args = config.general_settings()
    knet_args.use_cuda = False
    knet_args.n_batch = 5
    knet_args.knet_trainable = True
    knet_args.use_context_mod = False
    knet_args.in_mult_KNet = 5
    knet_args.out_mult_KNet = 40
    
    n_params = knet.NNBuild(knet_sys_model, knet_args)
    
    print(f"  ✓ KalmanNet initialized")
    print(f"  ✓ Parameters: {n_params:,}")
    
    # ========== Step 5: Test Forward Pass ==========
    print("\n[5/5] Testing forward pass...")
    
    # Initialize KalmanNet
    batch_size = 2
    T = 10
    
    # Create initial state in MSCKF dimension
    m1_0_msckf = torch.zeros(batch_size, msckf_model.m, 1)
    m1_0_msckf[:, 0, 0] = 1.0  # Set quaternion w=1
    
    # Adapt to KNet dimension
    m1_0_knet, adaptation_info = adapter.adapt_state_to_knet(m1_0_msckf, msckf_model.m)
    
    # Initialize sequence
    knet.batch_size = batch_size
    knet.InitSequence(m1_0_knet, T)
    knet.init_hidden()
    
    # Forward pass through sequence
    x_posterior_list = []
    
    for t in range(T):
        # Get observation (first 2 sequences, time t)
        y_msckf = train_input[:batch_size, :, t:t+1]
        
        # Adapt observation
        y_knet = adapter.adapt_observation_to_knet(y_msckf)
        
        # KalmanNet forward
        x_knet = knet.forward(y_knet)
        
        # Adapt state back to MSCKF
        x_msckf = adapter.adapt_state_from_knet(x_knet, adaptation_info)
        
        x_posterior_list.append(x_msckf)
    
    print(f"  ✓ Successfully processed {T} time steps")
    print(f"  ✓ Output state shape: {x_msckf.shape}")
    
    # Get Kalman gain
    KG_knet = knet.KGain
    KG_msckf = adapter.adapt_kalman_gain(KG_knet, adaptation_info)
    
    print(f"  ✓ Kalman gain shape (KNet): {KG_knet.shape}")
    print(f"  ✓ Kalman gain shape (MSCKF): {KG_msckf.shape}")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)
    print("\nKey achievements:")
    print("  ✓ MSCKF system model created and data generated")
    print("  ✓ Dimension adapter handles variable state dimensions")
    print("  ✓ KalmanNet processes adapted inputs")
    print("  ✓ Kalman gain computed and mapped back to MSCKF dimension")
    print("\nNext steps:")
    print("  • Run full training: python main_msckf_adaptiveknet.py --mode train")
    print("  • See documentation: msckf_integration/README_MSCKF.md")
    print("  • Customize parameters in main_msckf_adaptiveknet.py")
    print("="*80)


if __name__ == '__main__':
    demo_msckf_adaptiveknet()
