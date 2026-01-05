"""# **Training Pipeline for MSCKF-AdaptiveKNet**

This module provides the training pipeline for MSCKF with AdaptiveKNet.

Training Strategy:
1. Generate synthetic MSCKF trajectories with varying noise conditions
2. Train AdaptiveKNet to predict optimal Kalman gains
3. Handle dimension adaptation between MSCKF and KalmanNet
4. Support both fixed and adaptive (context-modulated) training

The pipeline handles:
- Data generation for MSCKF scenarios
- Training loop with dimension adaptation
- Validation and checkpointing
- Model saving/loading
"""

import torch
import torch.nn as nn
import random
import time
import os


class Pipeline_MSCKF_AdaptiveKNet:
    """
    Training pipeline for MSCKF-AdaptiveKNet integration
    
    Args:
        Time: Timestamp string for run identification
        folderName: Folder to save models and results
        modelName: Name identifier for the model
    """
    
    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        
        # Create folder if doesn't exist
        os.makedirs(self.folderName, exist_ok=True)
        
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"
        
    def save(self):
        """Save pipeline state"""
        torch.save(self, self.PipelineName)
    
    def setModel(self, knet_model, adapter=None, hnet_model=None):
        """
        Set models for training
        
        Args:
            knet_model: KalmanNet model
            adapter: Dimension adapter (optional, for MSCKF integration)
            hnet_model: Hypernetwork model (optional, for context modulation)
        """
        self.knet = knet_model
        self.adapter = adapter
        self.hnet = hnet_model
        
    def setTrainingParams(self, args):
        """
        Set training parameters
        
        Args:
            args: Configuration arguments with training hyperparameters
        """
        self.args = args
        
        # Device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Training hyperparameters
        self.N_steps = args.n_steps  # Number of training epochs
        self.N_B = args.n_batch  # Batch size
        self.learningRate = args.lr  # Learning rate
        self.weightDecay = args.wd  # Weight decay (L2 regularization)
        
        # Loss function
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # Optimizer
        if self.hnet is not None:
            # Train hypernetwork (context modulation)
            self.optimizer = torch.optim.Adam(
                self.hnet.parameters(),
                lr=self.learningRate,
                weight_decay=self.weightDecay
            )
        else:
            # Train KalmanNet directly
            self.optimizer = torch.optim.Adam(
                self.knet.parameters(),
                lr=self.learningRate,
                weight_decay=self.weightDecay
            )
        
        # Add adapter parameters if using learnable projection
        if self.adapter is not None and hasattr(self.adapter, 'state_encoder'):
            self.optimizer.add_param_group({
                'params': self.adapter.parameters(),
                'lr': self.learningRate * 0.1  # Lower learning rate for adapter
            })
    
    def NNTrain(self, msckf_model, train_input, train_target, cv_input, cv_target,
                path_results, train_init=None, cv_init=None, randomLength=False,
                train_lengthMask=None, cv_lengthMask=None):
        """
        Train MSCKF-AdaptiveKNet on a single dataset
        
        Args:
            msckf_model: MSCKF system model
            train_input: Training observations [N_train, msckf_n, T]
            train_target: Training states [N_train, msckf_m, T]
            cv_input: Validation observations [N_cv, msckf_n, T]
            cv_target: Validation states [N_cv, msckf_m, T]
            path_results: Path to save results
            train_init: Training initial states (optional)
            cv_init: Validation initial states (optional)
            randomLength: Whether sequences have variable length
            train_lengthMask: Mask for training sequence lengths
            cv_lengthMask: Mask for validation sequence lengths
        """
        
        # Track best model
        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0
        
        # Initialize loss tracking
        self.MSE_cv_linear_epoch = torch.zeros(self.N_steps)
        self.MSE_cv_dB_epoch = torch.zeros(self.N_steps)
        self.MSE_train_linear_epoch = torch.zeros(self.N_steps)
        self.MSE_train_dB_epoch = torch.zeros(self.N_steps)
        
        # Dataset sizes
        N_train = train_input.shape[0]
        N_cv = cv_input.shape[0]
        msckf_m = train_target.shape[1]
        msckf_n = train_input.shape[1]
        T = train_input.shape[2]
        
        # Initial states
        if train_init is None:
            train_init = msckf_model.m1x_0.view(1, msckf_m, 1).expand(N_train, -1, -1)
        if cv_init is None:
            cv_init = msckf_model.m1x_0.view(1, msckf_m, 1).expand(N_cv, -1, -1)
        
        print(f"Training MSCKF-AdaptiveKNet")
        print(f"Train samples: {N_train}, CV samples: {N_cv}")
        print(f"State dim: {msckf_m}, Obs dim: {msckf_n}, Sequence length: {T}")
        
        # Training loop
        for epoch in range(self.N_steps):
            
            #################
            ### Training  ###
            #################
            
            self.optimizer.zero_grad()
            
            # Training mode
            if self.hnet is not None:
                self.hnet.train()
            else:
                self.knet.train()
            
            if self.adapter is not None:
                self.adapter.train()
            
            # Sample random batch
            n_batch_indices = random.sample(range(N_train), k=self.N_B)
            
            # Initialize batch tensors
            y_training_batch = torch.zeros(self.N_B, msckf_n, T).to(self.device)
            train_target_batch = torch.zeros(self.N_B, msckf_m, T).to(self.device)
            x_out_training_batch = torch.zeros(self.N_B, msckf_m, T).to(self.device)
            train_init_batch = torch.zeros(self.N_B, msckf_m, 1).to(self.device)
            
            # Fill batch
            for idx, train_idx in enumerate(n_batch_indices):
                y_training_batch[idx, :, :] = train_input[train_idx]
                train_target_batch[idx, :, :] = train_target[train_idx]
                train_init_batch[idx, :, 0] = train_init[train_idx, :, 0]
            
            # Initialize hidden states
            if self.hnet is not None and hasattr(self.hnet, 'init_hidden'):
                self.hnet.init_hidden()
            self.knet.init_hidden()
            
            # Update batch size in knet
            self.knet.batch_size = self.N_B
            
            # Initialize sequence with adapter if available
            if self.adapter is not None:
                # Adapt initial state
                train_init_knet, _ = self.adapter.adapt_state_to_knet(
                    train_init_batch, msckf_m
                )
                self.knet.InitSequence(train_init_knet, T)
            else:
                self.knet.InitSequence(train_init_batch, T)
            
            # Forward pass through sequence
            for t in range(T):
                y_t = y_training_batch[:, :, t:t+1]
                
                # Adapt observation if using adapter
                if self.adapter is not None:
                    y_t_knet = self.adapter.adapt_observation_to_knet(y_t)
                else:
                    y_t_knet = y_t
                
                # Get weights from hypernetwork if available
                if self.hnet is not None:
                    # Context: use noise parameters (simplified)
                    context = torch.ones(1, 2).to(self.device)  # [q2, r2]
                    weights_knet = self.hnet(context)
                    x_out_knet = self.knet(y_t_knet, weights_knet=weights_knet)
                else:
                    x_out_knet = self.knet(y_t_knet)
                
                # Adapt back to MSCKF dimension if using adapter
                if self.adapter is not None:
                    adaptation_info = {
                        'method': self.adapter.adaptation_method,
                        'original_dim': msckf_m
                    }
                    if self.adapter.adaptation_method == 'split':
                        # Store camera poses
                        adaptation_info['camera_poses'] = train_init_batch[:, 16:, :]
                    x_out_msckf = self.adapter.adapt_state_from_knet(x_out_knet, adaptation_info)
                else:
                    x_out_msckf = x_out_knet
                
                x_out_training_batch[:, :, t] = x_out_msckf.squeeze(2)
            
            # Compute training loss
            if randomLength:
                MSE_train_linear = torch.zeros(self.N_B)
                for idx in range(self.N_B):
                    mask = train_lengthMask[n_batch_indices[idx], :]
                    MSE_train_linear[idx] = self.loss_fn(
                        x_out_training_batch[idx, :, mask],
                        train_target_batch[idx, :, mask]
                    )
                MSE_train_linear_avg = torch.mean(MSE_train_linear)
            else:
                MSE_train_linear_avg = self.loss_fn(x_out_training_batch, train_target_batch)
            
            # Backward pass
            MSE_train_linear_avg.backward()
            self.optimizer.step()
            
            # Track training loss
            self.MSE_train_linear_epoch[epoch] = MSE_train_linear_avg.item()
            self.MSE_train_dB_epoch[epoch] = 10 * torch.log10(MSE_train_linear_avg).item()
            
            ##################
            ### Validation ###
            ##################
            
            if self.hnet is not None:
                self.hnet.eval()
            else:
                self.knet.eval()
            
            if self.adapter is not None:
                self.adapter.eval()
            
            with torch.no_grad():
                # Initialize CV batch
                y_cv_batch = cv_input.to(self.device)
                cv_target_batch = cv_target.to(self.device)
                x_out_cv_batch = torch.zeros(N_cv, msckf_m, T).to(self.device)
                
                # Initialize hidden states
                if self.hnet is not None and hasattr(self.hnet, 'init_hidden'):
                    self.hnet.init_hidden()
                self.knet.init_hidden()
                
                # Update batch size
                self.knet.batch_size = N_cv
                
                # Initialize sequence
                if self.adapter is not None:
                    cv_init_knet, _ = self.adapter.adapt_state_to_knet(cv_init, msckf_m)
                    self.knet.InitSequence(cv_init_knet, T)
                else:
                    self.knet.InitSequence(cv_init, T)
                
                # Forward pass
                for t in range(T):
                    y_t = y_cv_batch[:, :, t:t+1]
                    
                    if self.adapter is not None:
                        y_t_knet = self.adapter.adapt_observation_to_knet(y_t)
                    else:
                        y_t_knet = y_t
                    
                    if self.hnet is not None:
                        context = torch.ones(1, 2).to(self.device)
                        weights_knet = self.hnet(context)
                        x_out_knet = self.knet(y_t_knet, weights_knet=weights_knet)
                    else:
                        x_out_knet = self.knet(y_t_knet)
                    
                    if self.adapter is not None:
                        adaptation_info = {
                            'method': self.adapter.adaptation_method,
                            'original_dim': msckf_m
                        }
                        if self.adapter.adaptation_method == 'split':
                            adaptation_info['camera_poses'] = cv_init[:, 16:, :]
                        x_out_msckf = self.adapter.adapt_state_from_knet(x_out_knet, adaptation_info)
                    else:
                        x_out_msckf = x_out_knet
                    
                    x_out_cv_batch[:, :, t] = x_out_msckf.squeeze(2)
                
                # Compute CV loss
                if randomLength:
                    MSE_cv_linear = torch.zeros(N_cv)
                    for idx in range(N_cv):
                        mask = cv_lengthMask[idx, :]
                        MSE_cv_linear[idx] = self.loss_fn(
                            x_out_cv_batch[idx, :, mask],
                            cv_target_batch[idx, :, mask]
                        )
                    MSE_cv_linear_avg = torch.mean(MSE_cv_linear)
                else:
                    MSE_cv_linear_avg = self.loss_fn(x_out_cv_batch, cv_target_batch)
                
                # Track CV loss
                self.MSE_cv_linear_epoch[epoch] = MSE_cv_linear_avg.item()
                self.MSE_cv_dB_epoch[epoch] = 10 * torch.log10(MSE_cv_linear_avg).item()
                
                # Save best model
                if self.MSE_cv_dB_epoch[epoch] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[epoch]
                    self.MSE_cv_idx_opt = epoch
                    
                    # Save models
                    if self.hnet is not None:
                        torch.save(self.hnet.state_dict(), path_results + 'hnet_best_model.pt')
                    else:
                        torch.save(self.knet.state_dict(), path_results + 'knet_best_model.pt')
                    
                    if self.adapter is not None and hasattr(self.adapter, 'state_encoder'):
                        torch.save(self.adapter.state_dict(), path_results + 'adapter_best_model.pt')
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.N_steps} - "
                      f"Train Loss: {self.MSE_train_dB_epoch[epoch]:.2f} dB, "
                      f"CV Loss: {self.MSE_cv_dB_epoch[epoch]:.2f} dB, "
                      f"Best CV: {self.MSE_cv_dB_opt:.2f} dB at epoch {self.MSE_cv_idx_opt+1}")
        
        print(f"\nTraining completed!")
        print(f"Best CV Loss: {self.MSE_cv_dB_opt:.2f} dB at epoch {self.MSE_cv_idx_opt+1}")
        
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch,
                self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]
    
    def NNTest(self, msckf_model, test_input, test_target, path_results,
               test_init=None, randomLength=False, test_lengthMask=None):
        """
        Test MSCKF-AdaptiveKNet on test dataset
        
        Args:
            msckf_model: MSCKF system model
            test_input: Test observations [N_test, msckf_n, T]
            test_target: Test states [N_test, msckf_m, T]
            path_results: Path to load models
            test_init: Test initial states (optional)
            randomLength: Whether sequences have variable length
            test_lengthMask: Mask for test sequence lengths
        
        Returns:
            Test results including MSE and predictions
        """
        
        # Load best model
        if self.hnet is not None:
            self.hnet.load_state_dict(torch.load(path_results + 'hnet_best_model.pt', 
                                                 map_location=self.device))
            self.hnet.eval()
        else:
            self.knet.load_state_dict(torch.load(path_results + 'knet_best_model.pt',
                                                 map_location=self.device))
            self.knet.eval()
        
        if self.adapter is not None and hasattr(self.adapter, 'state_encoder'):
            if os.path.exists(path_results + 'adapter_best_model.pt'):
                self.adapter.load_state_dict(torch.load(path_results + 'adapter_best_model.pt',
                                                        map_location=self.device))
            self.adapter.eval()
        
        # Dataset info
        N_test = test_input.shape[0]
        msckf_m = test_target.shape[1]
        msckf_n = test_input.shape[1]
        T = test_input.shape[2]
        
        if test_init is None:
            test_init = msckf_model.m1x_0.view(1, msckf_m, 1).expand(N_test, -1, -1)
        
        print(f"\nTesting MSCKF-AdaptiveKNet")
        print(f"Test samples: {N_test}")
        
        # Initialize output
        x_out_test = torch.zeros(N_test, msckf_m, T).to(self.device)
        MSE_test_linear_arr = torch.zeros(N_test)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Process each test sequence
            for idx in range(N_test):
                # Get sequence
                y_seq = test_input[idx:idx+1].to(self.device)
                x_init = test_init[idx:idx+1].to(self.device)
                
                # Initialize
                if self.hnet is not None and hasattr(self.hnet, 'init_hidden'):
                    self.hnet.init_hidden()
                self.knet.init_hidden()
                self.knet.batch_size = 1
                
                if self.adapter is not None:
                    x_init_knet, _ = self.adapter.adapt_state_to_knet(x_init, msckf_m)
                    self.knet.InitSequence(x_init_knet, T)
                else:
                    self.knet.InitSequence(x_init, T)
                
                # Forward pass
                for t in range(T):
                    y_t = y_seq[:, :, t:t+1]
                    
                    if self.adapter is not None:
                        y_t_knet = self.adapter.adapt_observation_to_knet(y_t)
                    else:
                        y_t_knet = y_t
                    
                    if self.hnet is not None:
                        context = torch.ones(1, 2).to(self.device)
                        weights_knet = self.hnet(context)
                        x_out_knet = self.knet(y_t_knet, weights_knet=weights_knet)
                    else:
                        x_out_knet = self.knet(y_t_knet)
                    
                    if self.adapter is not None:
                        adaptation_info = {
                            'method': self.adapter.adaptation_method,
                            'original_dim': msckf_m
                        }
                        if self.adapter.adaptation_method == 'split':
                            adaptation_info['camera_poses'] = x_init[:, 16:, :]
                        x_out_msckf = self.adapter.adapt_state_from_knet(x_out_knet, adaptation_info)
                    else:
                        x_out_msckf = x_out_knet
                    
                    x_out_test[idx, :, t] = x_out_msckf.squeeze()
                
                # Compute loss for this sequence
                if randomLength:
                    mask = test_lengthMask[idx, :]
                    MSE_test_linear_arr[idx] = self.loss_fn(
                        x_out_test[idx, :, mask],
                        test_target[idx, :, mask].to(self.device)
                    ).item()
                else:
                    MSE_test_linear_arr[idx] = self.loss_fn(
                        x_out_test[idx, :, :],
                        test_target[idx, :, :].to(self.device)
                    ).item()
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Compute statistics
        MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
        MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg)
        MSE_test_std = torch.std(MSE_test_linear_arr, unbiased=True)
        MSE_test_std_dB = 10 * torch.log10(MSE_test_std + MSE_test_linear_avg) - MSE_test_dB_avg
        
        print(f"Test MSE: {MSE_test_dB_avg:.2f} ± {MSE_test_std_dB:.2f} dB")
        print(f"Inference time: {inference_time:.2f} seconds ({inference_time/N_test*1000:.2f} ms per sequence)")
        
        return [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, x_out_test]
