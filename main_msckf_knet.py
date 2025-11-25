"""
Main script for MSCKF-KalmanNet Fusion

This script demonstrates how to use the MSCKF-KalmanNet fusion for 
visual-inertial odometry with learned Kalman gain estimation.

The pipeline:
1. Initialize MSCKF-KalmanNet with camera and IMU parameters
2. Build KalmanNet model (using existing Adaptive KNet architecture)
3. Optionally use Hypernetwork for adaptive noise handling
4. Process IMU and visual measurements
5. Compare performance with standard MSCKF

Usage:
    python main_msckf_knet.py
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import argparse

# Import MSCKF-KalmanNet components
from msckf_knet import (
    CameraParams,
    IMUState,
    IMUReading,
    NoiseParams,
    MSCKFParams,
    MSCKF,
    MSCKFKalmanNet,
    VIOSystemModel
)

# Import KalmanNet components from existing codebase
from mnets.KNet_mnet import KalmanNetNN
from hnets.hnet import HyperNetwork


class KNetArgs:
    """Arguments for KalmanNet configuration."""
    def __init__(self):
        # Dataset settings
        self.N_E = 1000
        self.N_CV = 100
        self.N_T = 200
        self.T = 100
        self.T_test = 100
        self.randomLength = False
        self.T_max = 1000
        self.T_min = 100
        self.randomInit_train = False
        self.randomInit_cv = False
        self.randomInit_test = False
        self.variance = 100
        self.init_distri = 'normal'
        self.proc_noise_distri = 'normal'
        self.meas_noise_distri = 'normal'
        
        # Training settings
        self.wandb_switch = False
        self.use_cuda = False
        self.mixed_dataset = False
        self.n_steps = 1000
        self.n_batch = 20
        self.n_batch_list = [20]
        self.lr = 1e-3
        self.wd = 1e-4
        self.grid_size_dB = 1
        self.forget_factor = 0.3
        self.max_iter = 100
        self.SoW_conv_error = 1e-3
        self.CompositionLoss = False
        self.alpha = 0.3
        self.RobustScaler = False
        self.UnsupervisedLoss = False
        
        # KalmanNet settings
        self.in_mult_KNet = 5
        self.out_mult_KNet = 40
        self.use_context_mod = False
        self.knet_trainable = False
        
        # HyperNetwork settings
        self.hnet_hidden_size_discount = 100
        self.hnet_arch = 'deconv'


class MSCKFKNetArgs:
    """Arguments for MSCKF-KalmanNet demo."""
    def __init__(self):
        self.use_cuda = False
        self.use_adaptive = False  # Disable adaptive mode by default to reduce memory
        self.sequence_length = 100
        self.demo_mode = True


def get_args():
    """Get configuration arguments."""
    # Use class-based args instead of argparse to avoid conflicts
    args = MSCKFKNetArgs()
    return args


def create_default_camera_params():
    """Create default camera parameters (based on typical VIO setup)."""
    camera = CameraParams(
        # Intrinsic parameters (EuRoC MAV dataset style)
        c_u=367.215,
        c_v=248.375,
        f_u=458.654,
        f_v=457.296,
        b=0.0,  # Monocular
        # Extrinsic: camera to IMU
        q_CI=np.array([0.9999, 0.0, 0.0, 0.0]),  # Identity rotation
        p_C_I=np.array([0.0, 0.0, 0.0])  # No translation
    )
    return camera


def create_default_noise_params():
    """Create default noise parameters."""
    # IMU noise parameters
    gyro_noise_density = 0.00016  # rad/s/sqrt(Hz)
    gyro_random_walk = 0.000022  # rad/s^2/sqrt(Hz)
    accel_noise_density = 0.0028  # m/s^2/sqrt(Hz)
    accel_random_walk = 0.00086  # m/s^3/sqrt(Hz)
    
    # Q_imu: [gyro_noise, gyro_bias_noise, accel_noise, accel_bias_noise]
    Q_imu = np.eye(12)
    Q_imu[0:3, 0:3] = gyro_noise_density ** 2 * np.eye(3)
    Q_imu[3:6, 3:6] = gyro_random_walk ** 2 * np.eye(3)
    Q_imu[6:9, 6:9] = accel_noise_density ** 2 * np.eye(3)
    Q_imu[9:12, 9:12] = accel_random_walk ** 2 * np.eye(3)
    
    # Initial IMU covariance
    initial_imu_covar = np.eye(15)
    initial_imu_covar[0:3, 0:3] = 0.01 * np.eye(3)  # Orientation
    initial_imu_covar[3:6, 3:6] = 0.0001 * np.eye(3)  # Gyro bias
    initial_imu_covar[6:9, 6:9] = 0.1 * np.eye(3)  # Velocity
    initial_imu_covar[9:12, 9:12] = 0.0001 * np.eye(3)  # Accel bias
    initial_imu_covar[12:15, 12:15] = 0.01 * np.eye(3)  # Position
    
    noise_params = NoiseParams(
        u_var_prime=0.01,
        v_var_prime=0.01,
        Q_imu=Q_imu,
        initial_imu_covar=initial_imu_covar
    )
    return noise_params


def create_default_msckf_params():
    """Create default MSCKF parameters."""
    msckf_params = MSCKFParams(
        max_gn_cost_norm=1e6,
        min_rcond=1e-12,
        translation_threshold=0.4,
        redundancy_angle_thresh=0.5,
        redundancy_distance_thresh=0.5,
        min_track_length=3,
        max_track_length=20,
        max_cam_states=20
    )
    return msckf_params


def create_initial_imu_state():
    """Create initial IMU state."""
    imu_state = IMUState(
        p_I_G=np.zeros(3),
        v_I_G=np.zeros(3),
        q_IG=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        b_g=np.zeros(3),
        b_a=np.zeros(3),
        g=np.array([0.0, 0.0, -9.81])
    )
    return imu_state


def generate_synthetic_data(T: int, dt: float = 0.01):
    """
    Generate synthetic IMU and observation data for testing.
    
    Args:
        T: Number of time steps
        dt: Time delta between steps
        
    Returns:
        Tuple of (imu_readings, observations, ground_truth)
    """
    # Generate a simple circular trajectory
    omega_z = 0.1  # Angular velocity around z-axis
    
    imu_readings = []
    observations = []
    ground_truth = {
        'positions': [],
        'velocities': [],
        'orientations': []
    }
    
    # Initial state
    pos = np.zeros(3)
    vel = np.array([1.0, 0.0, 0.0])  # Initial forward velocity
    angle = 0.0
    
    for t in range(T):
        # Ground truth state
        angle = omega_z * t * dt
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Update velocity (rotate)
        vel = R @ np.array([1.0, 0.0, 0.0])
        
        # Update position
        pos = pos + vel * dt
        
        # Store ground truth
        ground_truth['positions'].append(pos.copy())
        ground_truth['velocities'].append(vel.copy())
        ground_truth['orientations'].append(angle)
        
        # Generate IMU reading with noise
        omega = np.array([0.0, 0.0, omega_z]) + 0.001 * np.random.randn(3)
        accel = np.array([0.0, 0.0, 9.81]) + 0.01 * np.random.randn(3)  # Gravity
        
        imu_reading = IMUReading(
            omega=omega,
            a=accel,
            dT=dt
        )
        imu_readings.append(imu_reading)
        
        # Generate observation (simplified feature projection)
        # In real VIO, this would be feature tracks from images
        obs = np.array([pos[0], pos[1]]) + 0.1 * np.random.randn(2)
        observations.append(obs)
        
    return imu_readings, observations, ground_truth


def build_kalmannet_for_vio(sys_model: VIOSystemModel, args) -> KalmanNetNN:
    """
    Build a KalmanNet model configured for VIO.
    
    Args:
        sys_model: VIO system model
        args: Configuration arguments
        
    Returns:
        Configured KalmanNet model
    """
    kalmannet = KalmanNetNN()
    n_params = kalmannet.NNBuild(sys_model, args)
    return kalmannet, n_params


def run_msckf_knet_demo(args):
    """
    Run MSCKF-KalmanNet demonstration.
    
    Args:
        args: Configuration arguments
    """
    print("=" * 60)
    print("MSCKF-KalmanNet Fusion Demo")
    print("=" * 60)
    
    # Setup device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    # Create parameters
    camera = create_default_camera_params()
    noise_params = create_default_noise_params()
    msckf_params = create_default_msckf_params()
    imu_state = create_initial_imu_state()
    
    print("\nInitializing filters...")
    
    # Initialize standard MSCKF for comparison
    msckf_standard = MSCKF()
    msckf_standard.initialize(camera, noise_params, msckf_params, imu_state)
    print("Standard MSCKF initialized")
    
    # Initialize MSCKF-KalmanNet
    msckf_knet = MSCKFKalmanNet(use_adaptive=args.use_adaptive)
    msckf_knet.set_device(device)
    msckf_knet.initialize(camera, noise_params, msckf_params, imu_state)
    print("MSCKF-KalmanNet initialized")
    
    # Build KalmanNet model
    print("\nBuilding KalmanNet model...")
    
    # Create VIO system model
    vio_model = VIOSystemModel(
        T=args.sequence_length,
        T_test=args.sequence_length,
        q2=1.0,
        r2=1.0
    )
    
    # Configure KalmanNet args
    knet_args = KNetArgs()
    knet_args.use_cuda = args.use_cuda
    knet_args.use_context_mod = False
    knet_args.knet_trainable = True
    knet_args.n_batch = 1
    knet_args.in_mult_KNet = 5
    knet_args.out_mult_KNet = 40
    
    # Build KalmanNet
    kalmannet, n_params_knet = build_kalmannet_for_vio(vio_model, knet_args)
    kalmannet = kalmannet.to(device)
    print(f"KalmanNet built with {sum(p.numel() for p in kalmannet.parameters())} parameters")
    
    # Set KalmanNet in MSCKF-KalmanNet
    msckf_knet.set_kalmannet_model(kalmannet)
    
    if args.use_adaptive:
        print("\nBuilding Hypernetwork for adaptive estimation...")
        # SoW_len is 4 based on the existing code (SoW has 4 elements)
        hypernetwork = HyperNetwork(knet_args, SoW_len=4, output_size=n_params_knet)
        hypernetwork = hypernetwork.to(device)
        msckf_knet.set_hypernetwork(hypernetwork)
        print(f"Hypernetwork built with {sum(p.numel() for p in hypernetwork.parameters())} parameters")
        
    # Generate synthetic test data
    print(f"\nGenerating synthetic data ({args.sequence_length} steps)...")
    imu_readings, observations, ground_truth = generate_synthetic_data(
        args.sequence_length, dt=0.01
    )
    
    # Run filters
    print("\nRunning filters...")
    
    positions_standard = []
    positions_knet = []
    
    state_id = 0
    
    for t in range(args.sequence_length):
        imu = imu_readings[t]
        
        # Propagate both filters
        msckf_standard.propagate(imu)
        msckf_knet.propagate(imu)
        
        # Augment camera state (every 5 steps)
        if t % 5 == 0:
            msckf_standard.augment_state(state_id, t * 0.01)
            msckf_knet.augment_state(state_id, t * 0.01)
            state_id += 1
            
        # Store positions
        positions_standard.append(msckf_standard.get_position().copy())
        positions_knet.append(msckf_knet.get_position().copy())
        
        if (t + 1) % 20 == 0:
            print(f"  Processed {t+1}/{args.sequence_length} steps")
            
    print("\nProcessing complete!")
    
    # Compute errors
    gt_positions = np.array(ground_truth['positions'])
    pos_standard = np.array(positions_standard)
    pos_knet = np.array(positions_knet)
    
    error_standard = np.sqrt(np.sum((pos_standard - gt_positions) ** 2, axis=1))
    error_knet = np.sqrt(np.sum((pos_knet - gt_positions) ** 2, axis=1))
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nStandard MSCKF:")
    print(f"  Mean position error: {np.mean(error_standard):.4f} m")
    print(f"  Max position error:  {np.max(error_standard):.4f} m")
    print(f"  Final position error: {error_standard[-1]:.4f} m")
    
    print(f"\nMSCKF-KalmanNet:")
    print(f"  Mean position error: {np.mean(error_knet):.4f} m")
    print(f"  Max position error:  {np.max(error_knet):.4f} m")
    print(f"  Final position error: {error_knet[-1]:.4f} m")
    
    print("\n" + "=" * 60)
    print("Note: In this demo, KalmanNet is randomly initialized.")
    print("For best results, train KalmanNet on VIO data first.")
    print("=" * 60)
    
    return {
        'ground_truth': ground_truth,
        'positions_standard': pos_standard,
        'positions_knet': pos_knet,
        'errors_standard': error_standard,
        'errors_knet': error_knet
    }


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("MSCKF-KalmanNet: Visual-Inertial Odometry with Learned")
    print("                 Kalman Gain Estimation")
    print("=" * 60)
    print(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = get_args()
    
    # Run demo
    results = run_msckf_knet_demo(args)
    
    print("\nDemo completed successfully!")
    return results


if __name__ == '__main__':
    main()
