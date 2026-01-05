"""
MSCKF-AdaptiveKNet Integration Module

This module provides integration between MSCKF (Multi-State Constraint Kalman Filter)
and AdaptiveKNet for Visual-Inertial Odometry.

Key components:
- MSCKFSystemModel: System model for VIO with MSCKF
- MSCKFDimensionAdapter: Handles dimension mismatch between MSCKF and KalmanNet
- AdaptiveKNetMSCKF: Combined wrapper for end-to-end filtering
- Pipeline_MSCKF_AdaptiveKNet: Training and testing pipeline

Example usage:
    from msckf_integration import MSCKFSystemModel, MSCKFDimensionAdapter
    
    # Create MSCKF model
    msckf_model = MSCKFSystemModel(n_poses_max=10)
    
    # Create adapter
    adapter = MSCKFDimensionAdapter(
        msckf_m_max=86, msckf_n=40,
        knet_m=16, knet_n=40,
        adaptation_method='project'
    )
"""

from .dimension_adapter import MSCKFDimensionAdapter, AdaptiveKNetMSCKF
from .pipeline_msckf import Pipeline_MSCKF_AdaptiveKNet

__all__ = [
    'MSCKFDimensionAdapter',
    'AdaptiveKNetMSCKF',
    'Pipeline_MSCKF_AdaptiveKNet',
]

__version__ = '1.0.0'
__author__ = 'MSCKF-AdaptiveKNet Integration'
