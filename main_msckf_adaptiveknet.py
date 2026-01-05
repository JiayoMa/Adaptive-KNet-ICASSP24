"""# **Main Training Script for MSCKF-AdaptiveKNet**

This script demonstrates how to train AdaptiveKNet for MSCKF (Visual-Inertial Odometry).

Usage:
    python main_msckf_adaptiveknet.py --mode train --use_adapter --adaptation_method project

Key Features:
1. Configurable dimension adaptation strategies
2. Support for both standard KalmanNet and Hyper-KalmanNet
3. Synthetic MSCKF trajectory generation for training
4. Model checkpointing and evaluation

Training Process:
1. Create MSCKF system model with specified parameters
2. Generate synthetic training/validation/test datasets
3. Initialize AdaptiveKNet with dimension adapter
4. Train model to predict optimal Kalman gains
5. Evaluate on test set and save results
"""

import torch
import torch.nn as nn
from datetime import datetime
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.MSCKF_sysmdl import MSCKFSystemModel
from msckf_integration.dimension_adapter import MSCKFDimensionAdapter
from msckf_integration.pipeline_msckf import Pipeline_MSCKF_AdaptiveKNet
from mnets.KNet_mnet import KalmanNetNN
import simulations.config as config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MSCKF-AdaptiveKNet')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Training or testing mode')
    
    # Model configuration
    parser.add_argument('--use_adapter', action='store_true', default=True,
                       help='Use dimension adapter')
    parser.add_argument('--adaptation_method', type=str, default='project',
                       choices=['pad', 'project', 'split'],
                       help='Dimension adaptation method')
    parser.add_argument('--knet_m', type=int, default=16,
                       help='KalmanNet state dimension (default: 16 for IMU state)')
    parser.add_argument('--knet_n', type=int, default=40,
                       help='KalmanNet observation dimension')
    
    # MSCKF parameters
    parser.add_argument('--n_poses_max', type=int, default=10,
                       help='Maximum number of camera poses in MSCKF')
    parser.add_argument('--n_features', type=int, default=20,
                       help='Number of tracked features')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (seconds)')
    
    # Dataset parameters
    parser.add_argument('--N_E', type=int, default=500,
                       help='Number of training sequences')
    parser.add_argument('--N_CV', type=int, default=50,
                       help='Number of validation sequences')
    parser.add_argument('--N_T', type=int, default=100,
                       help='Number of test sequences')
    parser.add_argument('--T', type=int, default=100,
                       help='Sequence length for training')
    parser.add_argument('--T_test', type=int, default=100,
                       help='Sequence length for testing')
    
    # Training parameters
    parser.add_argument('--n_steps', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_batch', type=int, default=50,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--in_mult_KNet', type=int, default=5,
                       help='Input multiplier for KalmanNet')
    parser.add_argument('--out_mult_KNet', type=int, default=40,
                       help='Output multiplier for KalmanNet')
    
    # System
    parser.add_argument('--use_cuda', action='store_true', default=False,
                       help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Paths
    parser.add_argument('--results_dir', type=str, default='./results_msckf/',
                       help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main training/testing function"""
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Get timestamp
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print(f"MSCKF-AdaptiveKNet Pipeline - {strTime}")
    print("=" * 80)
    
    # Setup device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    ###############################
    ### Create MSCKF System Model ###
    ###############################
    
    print("\n" + "=" * 80)
    print("1. Creating MSCKF System Model")
    print("=" * 80)
    
    msckf_model = MSCKFSystemModel(
        n_poses_max=args.n_poses_max,
        T=args.T,
        T_test=args.T_test,
        dt=args.dt,
        device=device
    )
    
    msckf_model.n_features = args.n_features
    msckf_model.n = 2 * args.n_features  # Update observation dimension
    
    # Rebuild covariance matrices with new observation dimension
    msckf_model._build_covariance_matrices()
    
    print(f"MSCKF State dimension: {msckf_model.m} (IMU: {msckf_model.m_imu}, "
          f"Poses: {msckf_model.n_poses} x {msckf_model.m_pose})")
    print(f"MSCKF Observation dimension: {msckf_model.n} ({msckf_model.n_features} features x 2)")
    print(f"Time step: {msckf_model.dt}s")
    
    # Initialize model with zero camera poses (start simple)
    m1_0 = torch.zeros(msckf_model.m, 1)
    # Set initial quaternion to identity (w, x, y, z) = (1, 0, 0, 0)
    # Note: Using w-first quaternion convention (Hamilton)
    m1_0[0, 0] = 1.0  # w component of quaternion
    m2_0 = torch.eye(msckf_model.m)
    
    msckf_model.InitSequence(m1_0, m2_0)
    
    #########################
    ### Generate Datasets ###
    #########################
    
    print("\n" + "=" * 80)
    print("2. Generating MSCKF Training Datasets")
    print("=" * 80)
    
    # Training set
    print(f"Generating {args.N_E} training sequences...")
    config_args = config.general_settings()
    config_args.proc_noise_distri = 'normal'
    config_args.meas_noise_distri = 'normal'
    config_args.use_cuda = args.use_cuda
    
    msckf_model.GenerateBatch(config_args, args.N_E, args.T, randomInit=False)
    train_input = msckf_model.Input
    train_target = msckf_model.Target
    train_init = msckf_model.m1x_0_batch
    
    print(f"Training input shape: {train_input.shape}")
    print(f"Training target shape: {train_target.shape}")
    
    # Validation set
    print(f"Generating {args.N_CV} validation sequences...")
    msckf_model.GenerateBatch(config_args, args.N_CV, args.T, randomInit=False)
    cv_input = msckf_model.Input
    cv_target = msckf_model.Target
    cv_init = msckf_model.m1x_0_batch
    
    print(f"Validation input shape: {cv_input.shape}")
    
    # Test set
    print(f"Generating {args.N_T} test sequences...")
    msckf_model.GenerateBatch(config_args, args.N_T, args.T_test, randomInit=False)
    test_input = msckf_model.Input
    test_target = msckf_model.Target
    test_init = msckf_model.m1x_0_batch
    
    print(f"Test input shape: {test_input.shape}")
    
    ##############################
    ### Setup Dimension Adapter ###
    ##############################
    
    if args.use_adapter:
        print("\n" + "=" * 80)
        print("3. Setting up Dimension Adapter")
        print("=" * 80)
        
        # Maximum MSCKF dimension
        msckf_m_max = msckf_model.m_imu + msckf_model.m_pose * args.n_poses_max
        
        adapter = MSCKFDimensionAdapter(
            msckf_m_max=msckf_m_max,
            msckf_n=msckf_model.n,
            knet_m=args.knet_m,
            knet_n=args.knet_n,
            adaptation_method=args.adaptation_method,
            device=device
        )
        
        print(f"Adaptation method: {args.adaptation_method}")
        print(f"MSCKF state dimension (max): {msckf_m_max}")
        print(f"KalmanNet state dimension: {args.knet_m}")
        print(f"MSCKF observation dimension: {msckf_model.n}")
        print(f"KalmanNet observation dimension: {args.knet_n}")
        
        # Update dimensions for KalmanNet
        knet_m = args.knet_m
        knet_n = args.knet_n
    else:
        print("\n" + "=" * 80)
        print("3. No Dimension Adapter (Direct MSCKF)")
        print("=" * 80)
        
        adapter = None
        knet_m = msckf_model.m
        knet_n = msckf_model.n
    
    #########################
    ### Setup KalmanNet ###
    #########################
    
    print("\n" + "=" * 80)
    print("4. Initializing KalmanNet")
    print("=" * 80)
    
    # Create a SystemModel-like object for KalmanNet initialization
    class KNetSystemModel:
        def __init__(self, msckf_model, adapter, knet_m, knet_n):
            self.m = knet_m
            self.n = knet_n
            
            # Use adapted functions if adapter exists
            if adapter is not None:
                self.f, self.h = adapter.get_knet_system_functions(msckf_model)
            else:
                self.f = msckf_model.f
                self.h = msckf_model.h
            
            # Priors
            self.prior_Q = torch.eye(knet_m)
            self.prior_Sigma = torch.zeros(knet_m, knet_m)
            self.prior_S = torch.eye(knet_n)
    
    knet_sys_model = KNetSystemModel(msckf_model, adapter, knet_m, knet_n)
    
    # Initialize KalmanNet
    knet = KalmanNetNN()
    
    # Configuration for KalmanNet
    knet_args = config.general_settings()
    knet_args.use_cuda = args.use_cuda
    knet_args.n_batch = args.n_batch
    knet_args.knet_trainable = True  # Train KalmanNet directly
    knet_args.use_context_mod = False  # No context modulation for first training
    knet_args.in_mult_KNet = args.in_mult_KNet
    knet_args.out_mult_KNet = args.out_mult_KNet
    
    n_params = knet.NNBuild(knet_sys_model, knet_args)
    knet = knet.to(device)
    
    print(f"KalmanNet state dimension: {knet.m}")
    print(f"KalmanNet observation dimension: {knet.n}")
    print(f"KalmanNet parameters: {n_params:,}")
    
    ############################
    ### Setup Training Pipeline ###
    ############################
    
    print("\n" + "=" * 80)
    print("5. Setting up Training Pipeline")
    print("=" * 80)
    
    pipeline = Pipeline_MSCKF_AdaptiveKNet(
        Time=strTime,
        folderName=args.results_dir,
        modelName=f"msckf_adaptiveknet_{args.adaptation_method}"
    )
    
    pipeline.setModel(knet, adapter=adapter)
    
    # Set training parameters
    train_args = argparse.Namespace(
        use_cuda=args.use_cuda,
        n_steps=args.n_steps,
        n_batch=args.n_batch,
        lr=args.lr,
        wd=args.wd
    )
    
    pipeline.setTrainingParams(train_args)
    
    print(f"Training epochs: {args.n_steps}")
    print(f"Batch size: {args.n_batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.wd}")
    
    ###############
    ### Training ###
    ###############
    
    if args.mode == 'train':
        print("\n" + "=" * 80)
        print("6. Training MSCKF-AdaptiveKNet")
        print("=" * 80)
        
        losses = pipeline.NNTrain(
            msckf_model=msckf_model,
            train_input=train_input,
            train_target=train_target,
            cv_input=cv_input,
            cv_target=cv_target,
            path_results=args.results_dir,
            train_init=train_init,
            cv_init=cv_init,
            randomLength=False
        )
        
        print("\nTraining completed!")
        print(f"Results saved to: {args.results_dir}")
    
    ##############
    ### Testing ###
    ##############
    
    print("\n" + "=" * 80)
    print("7. Testing MSCKF-AdaptiveKNet")
    print("=" * 80)
    
    test_results = pipeline.NNTest(
        msckf_model=msckf_model,
        test_input=test_input,
        test_target=test_target,
        path_results=args.results_dir,
        test_init=test_init,
        randomLength=False
    )
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    
    # Save test results
    torch.save({
        'MSE_test_linear_arr': test_results[0],
        'MSE_test_linear_avg': test_results[1],
        'MSE_test_dB_avg': test_results[2],
        'x_out_test': test_results[3],
        'args': args
    }, args.results_dir + 'test_results.pt')
    
    print(f"Test results saved to: {args.results_dir}test_results.pt")


if __name__ == '__main__':
    main()
