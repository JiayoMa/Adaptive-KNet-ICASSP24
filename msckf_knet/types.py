"""
MSCKF Types - Python port of msckf_mono/types.h

This module defines the data structures used in MSCKF for visual-inertial odometry.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.spatial.transform import Rotation as ScipyRotation


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w] format."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z] format."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""
    # Intrinsic parameters
    c_u: float = 0.0  # Principal point u
    c_v: float = 0.0  # Principal point v
    f_u: float = 0.0  # Focal length u
    f_v: float = 0.0  # Focal length v
    b: float = 0.0    # Baseline (for stereo)
    
    # Extrinsic parameters (camera to IMU transform)
    q_CI: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Quaternion [w, x, y, z]
    p_C_I: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Position
    
    def __post_init__(self):
        if isinstance(self.q_CI, list):
            self.q_CI = np.array(self.q_CI)
        if isinstance(self.p_C_I, list):
            self.p_C_I = np.array(self.p_C_I)
    
    def get_rotation_CI(self) -> np.ndarray:
        """Get rotation matrix from IMU to Camera frame."""
        return ScipyRotation.from_quat(quat_wxyz_to_xyzw(self.q_CI)).as_matrix()


@dataclass
class CameraState:
    """Camera state at a particular time."""
    p_C_G: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Position in global frame
    q_CG: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Quaternion [w, x, y, z]
    time: float = 0.0
    state_id: int = -1
    last_correlated_id: int = -1
    tracked_feature_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.p_C_G, list):
            self.p_C_G = np.array(self.p_C_G)
        if isinstance(self.q_CG, list):
            self.q_CG = np.array(self.q_CG)
    
    def get_rotation_CG(self) -> np.ndarray:
        """Get rotation matrix from global to camera frame."""
        return ScipyRotation.from_quat(quat_wxyz_to_xyzw(self.q_CG)).as_matrix()


@dataclass
class IMUState:
    """IMU state."""
    # Position, velocity, and orientation
    p_I_G: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Position in global frame
    v_I_G: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Velocity in global frame
    q_IG: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Quaternion [w, x, y, z]
    
    # Biases
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Gyroscope bias
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Accelerometer bias
    
    # Gravity
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    
    # Null space projection variables (for observability constraints)
    p_I_G_null: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_I_G_null: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_IG_null: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    
    def __post_init__(self):
        for attr in ['p_I_G', 'v_I_G', 'q_IG', 'b_g', 'b_a', 'g', 'p_I_G_null', 'v_I_G_null', 'q_IG_null']:
            val = getattr(self, attr)
            if isinstance(val, list):
                setattr(self, attr, np.array(val))
    
    def get_rotation_IG(self) -> np.ndarray:
        """Get rotation matrix from global to IMU frame."""
        return ScipyRotation.from_quat(quat_wxyz_to_xyzw(self.q_IG)).as_matrix()
    
    def copy(self) -> 'IMUState':
        """Create a deep copy of the IMU state."""
        return IMUState(
            p_I_G=self.p_I_G.copy(),
            v_I_G=self.v_I_G.copy(),
            q_IG=self.q_IG.copy(),
            b_g=self.b_g.copy(),
            b_a=self.b_a.copy(),
            g=self.g.copy(),
            p_I_G_null=self.p_I_G_null.copy(),
            v_I_G_null=self.v_I_G_null.copy(),
            q_IG_null=self.q_IG_null.copy()
        )


@dataclass
class IMUReading:
    """IMU measurement."""
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Angular velocity
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))       # Linear acceleration
    dT: float = 0.0  # Time delta
    
    def __post_init__(self):
        if isinstance(self.omega, list):
            self.omega = np.array(self.omega)
        if isinstance(self.a, list):
            self.a = np.array(self.a)


@dataclass
class NoiseParams:
    """Noise parameters for MSCKF."""
    # Measurement noise variances
    u_var_prime: float = 0.0
    v_var_prime: float = 0.0
    
    # IMU noise covariance (12x12)
    Q_imu: np.ndarray = field(default_factory=lambda: np.eye(12))
    
    # Initial IMU covariance (15x15)
    initial_imu_covar: np.ndarray = field(default_factory=lambda: np.eye(15))
    
    def __post_init__(self):
        if isinstance(self.Q_imu, list):
            self.Q_imu = np.array(self.Q_imu)
        if isinstance(self.initial_imu_covar, list):
            self.initial_imu_covar = np.array(self.initial_imu_covar)


@dataclass
class MSCKFParams:
    """MSCKF algorithm parameters."""
    max_gn_cost_norm: float = 1e6
    min_rcond: float = 1e-12
    translation_threshold: float = 0.4
    redundancy_angle_thresh: float = 0.5
    redundancy_distance_thresh: float = 0.5
    min_track_length: int = 3
    max_track_length: int = 20
    max_cam_states: int = 20


@dataclass
class FeatureTrack:
    """Feature track across multiple camera states."""
    feature_id: int = 0
    observations: List[np.ndarray] = field(default_factory=list)  # List of 2D observations
    cam_state_indices: List[int] = field(default_factory=list)    # Corresponding camera state IDs
    initialized: bool = False
    p_f_G: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 3D position in global frame
    
    def __post_init__(self):
        if isinstance(self.p_f_G, list):
            self.p_f_G = np.array(self.p_f_G)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    Quaternion format: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Create omega matrix for quaternion kinematics.
    Used in RK4 integration of quaternion.
    """
    return np.array([
        [0, omega[2], -omega[1], omega[0]],
        [-omega[2], 0, omega[0], omega[1]],
        [omega[1], -omega[0], 0, omega[2]],
        [-omega[0], -omega[1], -omega[2], 0]
    ])


def build_update_quaternion(delta_theta: np.ndarray) -> np.ndarray:
    """
    Build quaternion from small angle rotation.
    
    Args:
        delta_theta: Small rotation angles [3,]
    
    Returns:
        Quaternion [w, x, y, z]
    """
    delta_q = 0.5 * delta_theta
    check_sum = np.sum(delta_q ** 2)
    
    if check_sum > 1:
        w = 1.0
    else:
        w = np.sqrt(1 - check_sum)
    
    q = np.array([w, -delta_q[0], -delta_q[1], -delta_q[2]])
    return quaternion_normalize(q)
