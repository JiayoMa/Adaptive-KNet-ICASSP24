"""
MSCKF-KalmanNet Fusion Module

This module combines Multi-State Constraint Kalman Filter (MSCKF) with KalmanNet,
replacing the traditional EKF in MSCKF with a neural network-based Kalman gain estimator.

Based on:
- MSCKF: https://github.com/JiayoMa/msckf_mono
- KalmanNet: https://github.com/JiayoMa/Adaptive-KNet-ICASSP24
"""

from .types import (
    CameraParams,
    CameraState,
    IMUState,
    IMUReading,
    NoiseParams,
    MSCKFParams,
    FeatureTrack
)
from .msckf import MSCKF
from .msckf_kalmannet import MSCKFKalmanNet
from .vio_system_model import VIOSystemModel

__all__ = [
    'CameraParams',
    'CameraState',
    'IMUState',
    'IMUReading',
    'NoiseParams',
    'MSCKFParams',
    'FeatureTrack',
    'MSCKF',
    'MSCKFKalmanNet',
    'VIOSystemModel'
]
