# MSCKF-KalmanNet Fusion

This module combines the **Multi-State Constraint Kalman Filter (MSCKF)** for visual-inertial odometry with **KalmanNet** for learned Kalman gain estimation.

## Overview

The MSCKF-KalmanNet fusion replaces the traditional Extended Kalman Filter (EKF) in MSCKF with a neural network-based approach for Kalman gain estimation. This can provide improved performance when:

1. **Model uncertainty**: The system dynamics are partially known or subject to variations
2. **Non-Gaussian noise**: The noise statistics are non-Gaussian or time-varying
3. **Linearization errors**: The EKF linearization errors are significant in highly nonlinear systems

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MSCKF-KalmanNet Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐      ┌──────────────────┐      ┌─────────────┐ │
│  │ IMU Sensors │──────│ IMU Propagation  │──────│ IMU State   │ │
│  │             │      │ (RK4 Integration)│      │ Estimate    │ │
│  └─────────────┘      └──────────────────┘      └─────────────┘ │
│                                                        │         │
│  ┌─────────────┐      ┌──────────────────┐            │         │
│  │   Camera    │──────│ Feature Tracking │            │         │
│  │             │      │ & Multi-State    │            ▼         │
│  └─────────────┘      │ Constraints      │      ┌─────────────┐ │
│                       └──────────────────┘      │  KalmanNet  │ │
│                              │                  │ (Learned KG)│ │
│                              │                  └─────────────┘ │
│                              │                        │         │
│                              ▼                        ▼         │
│                       ┌──────────────────────────────────┐      │
│                       │       Measurement Update          │      │
│                       │  (Replace EKF with KalmanNet)     │      │
│                       └──────────────────────────────────┘      │
│                                      │                           │
│                                      ▼                           │
│                               ┌─────────────┐                    │
│                               │ State Output │                   │
│                               │ (Position,   │                   │
│                               │  Orientation,│                   │
│                               │  Velocity)   │                   │
│                               └─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
msckf_knet/
├── __init__.py           # Module exports
├── types.py              # Data types (CameraParams, IMUState, etc.)
├── msckf.py              # Standard MSCKF implementation
├── msckf_kalmannet.py    # MSCKF with KalmanNet fusion
└── vio_system_model.py   # VIO system model for KalmanNet
```

## Key Components

### 1. Types (`types.py`)

Data structures ported from the C++ msckf_mono implementation:

- `CameraParams`: Camera intrinsic and extrinsic parameters
- `CameraState`: Camera pose at a particular time
- `IMUState`: IMU state (position, velocity, orientation, biases)
- `IMUReading`: IMU measurement (angular velocity, linear acceleration)
- `NoiseParams`: Noise parameters for the filter
- `MSCKFParams`: Algorithm parameters
- `FeatureTrack`: Feature track across multiple frames

### 2. Standard MSCKF (`msckf.py`)

Python implementation of the Multi-State Constraint Kalman Filter:

- **IMU Propagation**: RK4 integration of IMU measurements
- **State Augmentation**: Adding new camera states
- **Feature Tracking**: Managing feature observations
- **Measurement Update**: Standard EKF update with visual constraints

### 3. MSCKF-KalmanNet (`msckf_kalmannet.py`)

Extension that replaces EKF with KalmanNet:

- **Learned Kalman Gain**: Uses neural network to estimate optimal Kalman gain
- **Adaptive Mode**: Optional hypernetwork for noise adaptation
- **Hybrid Update**: KalmanNet for IMU state, EKF for camera states

### 4. VIO System Model (`vio_system_model.py`)

System model compatible with KalmanNet:

- State transition and observation functions
- Process and measurement noise covariances
- Support for variable-size MSCKF state space

## Usage

### Basic Example

```python
import numpy as np
from msckf_knet import (
    CameraParams, IMUState, NoiseParams, MSCKFParams,
    MSCKFKalmanNet, IMUReading
)
from mnets.KNet_mnet import KalmanNetNN

# Create parameters
camera = CameraParams(
    c_u=367.215, c_v=248.375,
    f_u=458.654, f_v=457.296
)
noise_params = NoiseParams(u_var_prime=0.01, v_var_prime=0.01)
msckf_params = MSCKFParams()
imu_state = IMUState()

# Initialize filter
msckf_knet = MSCKFKalmanNet()
msckf_knet.initialize(camera, noise_params, msckf_params, imu_state)

# Set KalmanNet model (pre-trained or trainable)
kalmannet = KalmanNetNN()
# ... build and train KalmanNet ...
msckf_knet.set_kalmannet_model(kalmannet)

# Process IMU measurements
imu_reading = IMUReading(
    omega=np.array([0.0, 0.0, 0.1]),
    a=np.array([0.0, 0.0, 9.81]),
    dT=0.01
)
msckf_knet.propagate(imu_reading)

# Get state estimate
position = msckf_knet.get_position()
orientation = msckf_knet.get_orientation()
```

### Running the Demo

```bash
python main_msckf_knet.py
```

This runs a demonstration with synthetic IMU and visual data, comparing:
- Standard MSCKF with EKF
- MSCKF-KalmanNet with learned Kalman gain

## Training KalmanNet for VIO

To achieve best performance, KalmanNet should be trained on VIO data:

1. **Generate Training Data**: Use a VIO simulator or real datasets (e.g., EuRoC)
2. **Define System Model**: Create `VIOSystemModel` with appropriate parameters
3. **Train KalmanNet**: Use the existing training pipeline from Adaptive-KNet
4. **Deploy**: Load trained weights into `MSCKFKalmanNet`

```python
from msckf_knet.vio_system_model import VIOSystemModel
from pipelines.Pipeline_hknet import Pipeline_hknet

# Create VIO system model
vio_model = VIOSystemModel(T=100, T_test=100, q2=1.0, r2=1.0)

# Build and train KalmanNet
# ... follow Adaptive-KNet training pipeline ...

# Use trained model
msckf_knet.set_kalmannet_model(trained_kalmannet)
```

## Comparison with Standard MSCKF

| Feature | Standard MSCKF | MSCKF-KalmanNet |
|---------|---------------|-----------------|
| Kalman Gain | Analytical (EKF) | Learned (Neural Net) |
| Model Dependence | High | Low |
| Noise Adaptation | Fixed | Adaptive (with HyperNet) |
| Computational Cost | Lower | Higher |
| Training Required | No | Yes |

## References

1. **MSCKF**: Mourikis, A. I., & Roumeliotis, S. I. (2007). A multi-state constraint Kalman filter for vision-aided inertial navigation. In ICRA 2007.

2. **KalmanNet**: Revach, G., et al. (2022). KalmanNet: Neural network aided Kalman filtering for partially known dynamics. In IEEE TSP.

3. **Adaptive KalmanNet**: Ma, J., et al. (2024). Adaptive KalmanNet. In ICASSP 2024.

## License

This code is provided for research purposes. Please refer to the original repositories for license information:
- [Adaptive-KNet-ICASSP24](https://github.com/JiayoMa/Adaptive-KNet-ICASSP24)
- [msckf_mono](https://github.com/JiayoMa/msckf_mono)
