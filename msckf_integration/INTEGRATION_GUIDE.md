# Integration Guide: Using Real MSCKF Data with AdaptiveKNet

This guide explains how to integrate real MSCKF data from [stereo_msckf](https://github.com/uoip/stereo_msckf) with AdaptiveKNet.

## Overview

The stereo_msckf repository provides a Python implementation of MSCKF for stereo visual-inertial odometry. This guide shows how to:

1. Extract state and observation data from stereo_msckf
2. Convert the data to AdaptiveKNet format
3. Train AdaptiveKNet to learn Kalman gains
4. Deploy the trained model back into stereo_msckf

## Step 1: Understanding stereo_msckf Data Structure

The stereo_msckf implementation maintains:

```python
# IMU State (in msckf.py)
imu_state = {
    'orientation': quaternion (4,),      # Camera orientation
    'position': np.array (3,),           # Camera position  
    'velocity': np.array (3,),           # Camera velocity
    'gyro_bias': np.array (3,),          # Gyroscope bias
    'acc_bias': np.array (3,)            # Accelerometer bias
}
# Total: 16 dimensions

# Camera States (sliding window)
cam_states = {
    cam_id: {
        'orientation': quaternion (4,),
        'position': np.array (3,)
    }
}
# Per camera: 7 dimensions
# Total state: 16 + 7*N_cameras

# Observations
features = {
    feature_id: [
        (cam_id, u, v),  # 2D pixel coordinates
        ...
    ]
}
```

## Step 2: Data Extraction from stereo_msckf

Create a data logger to capture MSCKF states and observations:

```python
# data_logger.py
import numpy as np
import pickle

class MSCKFDataLogger:
    """Logger to capture MSCKF data for AdaptiveKNet training"""
    
    def __init__(self):
        self.states = []
        self.observations = []
        self.timestamps = []
        
    def log_update(self, msckf, features):
        """
        Log one MSCKF update step
        
        Args:
            msckf: MSCKF filter object
            features: Feature observations
        """
        # Extract full state vector
        state = self._extract_state(msckf)
        
        # Extract observations
        obs = self._extract_observations(features)
        
        self.states.append(state)
        self.observations.append(obs)
        self.timestamps.append(msckf.time)
        
    def _extract_state(self, msckf):
        """Extract state vector from MSCKF"""
        # IMU state
        imu_state = np.concatenate([
            msckf.imu_state.orientation_null,  # quaternion
            msckf.imu_state.position,
            msckf.imu_state.velocity,
            msckf.imu_state.gyro_bias,
            msckf.imu_state.acc_bias
        ])
        
        # Camera states
        cam_states = []
        for cam_id in sorted(msckf.state_server.camera_states.keys()):
            cam = msckf.state_server.camera_states[cam_id]
            cam_state = np.concatenate([
                cam.orientation_null,
                cam.position
            ])
            cam_states.append(cam_state)
        
        if cam_states:
            full_state = np.concatenate([imu_state] + cam_states)
        else:
            full_state = imu_state
            
        return full_state
    
    def _extract_observations(self, features):
        """Extract observation vector from features"""
        obs_list = []
        for feature in features:
            for cam_id, (u, v) in feature.observations.items():
                obs_list.extend([u, v])
        
        # Pad to fixed size if needed
        obs = np.array(obs_list) if obs_list else np.zeros(40)
        
        # Pad or truncate to fixed observation dimension
        if len(obs) < 40:
            obs = np.pad(obs, (0, 40 - len(obs)))
        else:
            obs = obs[:40]
            
        return obs
    
    def save(self, filename):
        """Save logged data"""
        data = {
            'states': np.array(self.states),        # [T, m]
            'observations': np.array(self.observations),  # [T, n]
            'timestamps': np.array(self.timestamps)
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.states)} timesteps to {filename}")


# Integration into stereo_msckf/msckf.py
class MSCKF:
    def __init__(self, config):
        # ... existing code ...
        self.data_logger = MSCKFDataLogger()  # Add logger
        
    def measurement_update(self, feature):
        # ... existing measurement update code ...
        
        # Log before update (for AdaptiveKNet training)
        if hasattr(self, 'data_logger'):
            self.data_logger.log_update(self, [feature])
        
        # ... rest of update ...
```

## Step 3: Convert Data to PyTorch Format

```python
# convert_msckf_data.py
import torch
import pickle
import numpy as np

def convert_msckf_to_pytorch(msckf_data_file, output_dir='./data_msckf/'):
    """
    Convert stereo_msckf data to AdaptiveKNet format
    
    Args:
        msckf_data_file: Pickle file from MSCKFDataLogger
        output_dir: Directory to save PyTorch tensors
    """
    # Load logged data
    with open(msckf_data_file, 'rb') as f:
        data = pickle.load(f)
    
    states = data['states']        # [T, m]
    observations = data['observations']  # [T, n]
    
    T = states.shape[0]
    m = states.shape[1]
    n = observations.shape[1]
    
    print(f"Loaded sequence: T={T}, m={m}, n={n}")
    
    # Reshape to AdaptiveKNet format: [1, dim, T]
    # (single sequence, with time as last dimension)
    states_torch = torch.from_numpy(states.T).unsqueeze(0).float()  # [1, m, T]
    obs_torch = torch.from_numpy(observations.T).unsqueeze(0).float()  # [1, n, T]
    
    # Initial state
    init_state = torch.from_numpy(states[0]).unsqueeze(0).unsqueeze(2).float()  # [1, m, 1]
    
    # Save
    torch.save(states_torch, output_dir + 'states.pt')
    torch.save(obs_torch, output_dir + 'observations.pt')
    torch.save(init_state, output_dir + 'init_state.pt')
    
    print(f"Saved PyTorch tensors to {output_dir}")
    print(f"  states: {states_torch.shape}")
    print(f"  observations: {obs_torch.shape}")
    print(f"  init_state: {init_state.shape}")
    
    return states_torch, obs_torch, init_state


# Example usage
if __name__ == '__main__':
    # Convert multiple sequences
    sequences = ['seq01.pkl', 'seq02.pkl', 'seq03.pkl']
    
    all_states = []
    all_obs = []
    all_inits = []
    
    for seq_file in sequences:
        states, obs, init = convert_msckf_to_pytorch(seq_file)
        all_states.append(states)
        all_obs.append(obs)
        all_inits.append(init)
    
    # Combine into training dataset
    train_states = torch.cat(all_states, dim=0)    # [N, m, T]
    train_obs = torch.cat(all_obs, dim=0)          # [N, n, T]
    train_inits = torch.cat(all_inits, dim=0)      # [N, m, 1]
    
    torch.save(train_states, './data_msckf/train_states.pt')
    torch.save(train_obs, './data_msckf/train_observations.pt')
    torch.save(train_inits, './data_msckf/train_inits.pt')
    
    print(f"\nCombined training data:")
    print(f"  {train_states.shape[0]} sequences")
    print(f"  State dim: {train_states.shape[1]}")
    print(f"  Obs dim: {train_obs.shape[1]}")
    print(f"  Sequence length: {train_states.shape[2]}")
```

## Step 4: Train AdaptiveKNet on Real Data

Modify `main_msckf_adaptiveknet.py` to use real data:

```python
# Load real data instead of generating synthetic
print("Loading real MSCKF data...")
train_target = torch.load('./data_msckf/train_states.pt')
train_input = torch.load('./data_msckf/train_observations.pt')
train_init = torch.load('./data_msckf/train_inits.pt')

# Similarly for CV and test sets
cv_target = torch.load('./data_msckf/cv_states.pt')
cv_input = torch.load('./data_msckf/cv_observations.pt')
cv_init = torch.load('./data_msckf/cv_inits.pt')

test_target = torch.load('./data_msckf/test_states.pt')
test_input = torch.load('./data_msckf/test_observations.pt')
test_init = torch.load('./data_msckf/test_inits.pt')

# Update dimensions from data
m = train_target.shape[1]  # State dimension
n = train_input.shape[1]   # Observation dimension

print(f"Data loaded: {train_target.shape[0]} train, {cv_target.shape[0]} CV, {test_target.shape[0]} test")
print(f"State dimension: {m}, Observation dimension: {n}")

# ... continue with existing training code ...
```

Then run training:

```bash
python main_msckf_adaptiveknet.py \
    --mode train \
    --use_adapter \
    --adaptation_method project \
    --N_E 100 \
    --N_CV 20 \
    --N_T 30 \
    --n_steps 500 \
    --lr 1e-4 \
    --results_dir ./results_real_msckf/
```

## Step 5: Deploy Trained Model in stereo_msckf

Replace Kalman gain computation in `stereo_msckf/msckf.py`:

```python
# In stereo_msckf/msckf.py

import torch
from msckf_integration.dimension_adapter import MSCKFDimensionAdapter
from mnets.KNet_mnet import KalmanNetNN

class MSCKF:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Load trained AdaptiveKNet
        self.use_adaptiveknet = config.get('use_adaptiveknet', False)
        if self.use_adaptiveknet:
            self._setup_adaptiveknet(config)
    
    def _setup_adaptiveknet(self, config):
        """Setup AdaptiveKNet for Kalman gain computation"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.knet = KalmanNetNN()
        # ... initialize knet with proper dimensions ...
        self.knet.load_state_dict(torch.load(config['knet_model_path'], map_location=device))
        self.knet.eval()
        
        # Setup adapter if needed
        if config.get('use_adapter', True):
            self.adapter = MSCKFDimensionAdapter(
                msckf_m_max=config['m_max'],
                msckf_n=config['n'],
                knet_m=config['knet_m'],
                knet_n=config['knet_n'],
                adaptation_method=config['adaptation_method'],
                device=device
            )
        else:
            self.adapter = None
    
    def kalman_gain_computation(self, H, P, R):
        """
        Compute Kalman gain using AdaptiveKNet instead of traditional method
        
        Args:
            H: Observation matrix [n, m]
            P: State covariance [m, m]
            R: Observation noise covariance [n, n]
            
        Returns:
            K: Kalman gain [m, n]
        """
        if not self.use_adaptiveknet:
            # Traditional computation: K = P @ H.T @ inv(H @ P @ H.T + R)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            return K
        
        # AdaptiveKNet computation
        with torch.no_grad():
            # Convert state and observation to torch tensors
            x_current = self._get_current_state_vector()  # [m]
            y_current = self._get_current_observation()    # [n]
            
            # Add batch dimension
            x_batch = torch.from_numpy(x_current).unsqueeze(0).unsqueeze(2).float()  # [1, m, 1]
            y_batch = torch.from_numpy(y_current).unsqueeze(0).unsqueeze(2).float()  # [1, n, 1]
            
            # Adapt dimensions if using adapter
            if self.adapter is not None:
                x_knet, adapt_info = self.adapter.adapt_state_to_knet(x_batch, len(x_current))
                y_knet = self.adapter.adapt_observation_to_knet(y_batch)
            else:
                x_knet = x_batch
                y_knet = y_batch
                adapt_info = None
            
            # Forward pass through KalmanNet
            self.knet.InitSequence(x_knet, 1)
            self.knet.init_hidden()
            _ = self.knet.forward(y_knet)
            
            # Get Kalman gain
            K_knet = self.knet.KGain  # [1, knet_m, knet_n]
            
            # Adapt back to MSCKF dimension if needed
            if self.adapter is not None:
                K_msckf = self.adapter.adapt_kalman_gain(K_knet, adapt_info)
            else:
                K_msckf = K_knet
            
            # Convert to numpy
            K = K_msckf.squeeze(0).numpy()  # [m, n]
            
        return K
    
    def _get_current_state_vector(self):
        """Extract current state as numpy array"""
        # Concatenate IMU state + camera states
        state = np.concatenate([
            self.imu_state.orientation_null,
            self.imu_state.position,
            self.imu_state.velocity,
            self.imu_state.gyro_bias,
            self.imu_state.acc_bias
        ])
        
        for cam_id in sorted(self.state_server.camera_states.keys()):
            cam = self.state_server.camera_states[cam_id]
            cam_state = np.concatenate([cam.orientation_null, cam.position])
            state = np.concatenate([state, cam_state])
        
        return state
    
    def _get_current_observation(self):
        """Extract current observation as numpy array"""
        # This should return the feature observations being processed
        # Implementation depends on how observations are stored
        # Return fixed-size observation vector (pad if necessary)
        obs = np.zeros(40)  # Placeholder
        # ... fill with actual observations ...
        return obs
```

## Step 6: Configuration

Create a configuration file `config_adaptiveknet.yaml`:

```yaml
# AdaptiveKNet configuration for stereo_msckf

use_adaptiveknet: true
use_adapter: true

# Model paths
knet_model_path: './results_real_msckf/knet_best_model.pt'
adapter_model_path: './results_real_msckf/adapter_best_model.pt'

# Dimensions
m_max: 86  # 16 (IMU) + 7*10 (max cameras)
n: 40      # Observation dimension
knet_m: 16  # Use IMU state only
knet_n: 40

# Adaptation
adaptation_method: 'split'  # or 'project' or 'pad'
```

## Performance Comparison

After integration, compare performance:

```python
# benchmark_msckf.py
import time
import numpy as np

def benchmark_msckf(config, dataset):
    """Compare traditional MSCKF vs AdaptiveKNet-MSCKF"""
    
    results = {
        'traditional': {'time': [], 'error': []},
        'adaptiveknet': {'time': [], 'error': []}
    }
    
    for use_ak in [False, True]:
        config['use_adaptiveknet'] = use_ak
        msckf = MSCKF(config)
        
        for frame in dataset:
            start = time.time()
            msckf.process_frame(frame)
            elapsed = time.time() - start
            
            error = compute_error(msckf.get_pose(), frame.ground_truth)
            
            key = 'adaptiveknet' if use_ak else 'traditional'
            results[key]['time'].append(elapsed)
            results[key]['error'].append(error)
    
    # Print comparison
    print("Performance Comparison:")
    print(f"Traditional MSCKF:")
    print(f"  Avg time: {np.mean(results['traditional']['time'])*1000:.2f} ms")
    print(f"  Avg error: {np.mean(results['traditional']['error']):.4f}")
    
    print(f"AdaptiveKNet-MSCKF:")
    print(f"  Avg time: {np.mean(results['adaptiveknet']['time'])*1000:.2f} ms")
    print(f"  Avg error: {np.mean(results['adaptiveknet']['error']):.4f}")
    print(f"  Speedup: {np.mean(results['traditional']['time'])/np.mean(results['adaptiveknet']['time']):.2f}x")
```

## Troubleshooting

### Issue: Dimension mismatch during deployment

**Solution**: Ensure the state and observation dimensions match between training and deployment. Log the actual dimensions used in stereo_msckf and verify they match the trained model.

### Issue: Poor performance on real data

**Solution**: 
1. Collect more diverse training data
2. Fine-tune on the specific test sequence
3. Adjust noise parameters in training to match real sensor characteristics
4. Use data augmentation during training

### Issue: Numerical instability

**Solution**:
1. Normalize inputs before feeding to KalmanNet
2. Clip Kalman gain values to reasonable ranges
3. Use gradient clipping during training

## Summary

This integration provides:
- ✅ Reduced computational complexity (O(n²) vs O(n³))
- ✅ Faster Kalman gain computation
- ✅ Learned gains that adapt to data characteristics
- ✅ Easy deployment into existing MSCKF codebase

The key is proper data logging, format conversion, and careful integration of the learned model into the original MSCKF update step.
