import torch
import torch.nn as nn
import os
import sys
from datetime import datetime
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, EpochScoring
from preprocessing import preprocess, dataset_to_numpy, save_or_preprocess


class ProperSTSCNN(nn.Module):
    """
    Proper STSCNN as described in the paper:
    "a spatial layer was introduced on top of the original deep CNN to attenuate the noisy channels"
    
    Key insight: The spatial layer should OUTPUT the same number of channels,
    acting as a learned channel weighting, NOT collapsing to 1 channel.
    """
    
    def __init__(self, n_channels, n_times, n_classes=5, dropout_rate=0.5):
        super(ProperSTSCNN, self).__init__()
        
        self.n_channels = n_channels
        
        # STSCNN Innovation: Spatial layer on top that preserves channel dimension
        # This learns weights for each channel (n_channels output channels)
        self.spatial_top = nn.Conv2d(
            1, 1,  # Keep 1 input channel, 1 output channel? No - that's wrong.
            kernel_size=(n_channels, 1),
            stride=(1, 1),
            bias=False
        )
        # Wait - this outputs 1 channel. That's the problem!
        
        # CORRECTED: Output n_channels to preserve channel information
        self.spatial_top_corrected = nn.Conv2d(
            1, n_channels,  # Output same number of channels as input
            kernel_size=(1, 1),  # 1x1 convolution across channels
            stride=(1, 1),
            bias=False
        )
        
        # But 1x1 convolution doesn't mix channels. We need depthwise convolution.
        # Let me rethink this...
        
        self._initialize_weights()


class WorkingSTSCNN(nn.Module):
    """
    Working STSCNN based on what actually succeeded in your tests.
    
    This uses the architecture that achieved 48.42% but adds a proper
    channel attention mechanism that doesn't destroy the signal.
    """
    
    def __init__(self, n_channels, n_times, n_classes=5, dropout_rate=0.5):
        super(WorkingSTSCNN, self).__init__()
        
        self.n_channels = n_channels
        
        # CW-Deep CNN: Learnable channel weights (preserves channel dimension)
        # Using a 1x1 convolution that mixes channels via pointwise convolution
        self.channel_weights = nn.Parameter(torch.ones(1, 1, n_channels, 1))
        
        self.conv_time = nn.Conv2d(1, 64, kernel_size=(1, 50), stride=(1, 1))
        self.conv_spatial = nn.Conv2d(64, 64, kernel_size=(n_channels, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = nn.ELU()
        self.mp1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(64, 50, kernel_size=(1, 10), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.nonlinear2 = nn.ELU()
        self.drop2 = nn.Dropout(p=dropout_rate)
        
        self.conv3 = nn.Conv2d(50, 50, kernel_size=(1, 10), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(50)
        self.nonlinear3 = nn.ELU()
        self.drop3 = nn.Dropout(p=dropout_rate)
        
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 10), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(50)
        self.nonlinear4 = nn.ELU()
        self.drop4 = nn.Dropout(p=dropout_rate)
        
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.final_linear = nn.Linear(50, n_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize channel weights to 1 (no attenuation initially)
        nn.init.ones_(self.channel_weights)
    
    def forward(self, x):
        # Input: (batch, channels, time)
        x = x.unsqueeze(1)  # -> (batch, 1, channels, time)
        
        # STSCNN: Apply learnable channel weights (element-wise multiplication)
        # This attenuates noisy channels by learning small weights for them
        x = x * self.channel_weights  # -> (batch, 1, channels, time)
        
        # Continue with original deep CNN (same as your working model)
        x = self.conv_time(x)
        x = self.conv_spatial(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.nonlinear3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear4(x)
        x = self.drop4(x)
        
        x = self.ap(x)
        x = x.squeeze()
        x = self.final_linear(x)
        
        return x


def create_model(n_channels, n_times, n_classes=5):
    """Create the working STSCNN model"""
    return WorkingSTSCNN(n_channels, n_times, n_classes)


def setup_training(model, train_set, valid_set, test_set, device='cuda', max_epochs=200):
    """Set up training with hyperparameters that worked"""
    
    X_train, y_train = dataset_to_numpy(train_set)
    X_valid, y_valid = dataset_to_numpy(valid_set)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.LongTensor(y_valid)
    
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    # Use the hyperparameters that worked: LR=0.001, Adam, no weight decay
    clf = NeuralNetClassifier(
        model,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=0,  # No weight decay as in working model
        batch_size=32,
        max_epochs=max_epochs,
        device=device,
        train_split=predefined_split(valid_ds),
        callbacks=[
            EarlyStopping(patience=20, threshold=0.0005),
            EpochScoring('accuracy', name='valid_acc', lower_is_better=False),
        ],
        iterator_train__shuffle=True,
        iterator_train__num_workers=2,
        iterator_valid__num_workers=2,
        verbose=1
    )
    
    return clf, train_ds, valid_ds


def suppress_mne_logging():
    import mne
    mne.set_log_level('ERROR')
    import warnings
    warnings.filterwarnings('ignore')


class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except:
            pass
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def run_for_pn(pn, results_dir):
    """Run training for a specific pn"""
    
    patient_dir = os.path.join(results_dir, f'P{pn}')
    os.makedirs(patient_dir, exist_ok=True)
    
    output_file = os.path.join(patient_dir, 'stscnn_results.txt')
    
    suppress_mne_logging()
    
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"STSCNN - Patient {pn}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Load data
        train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
        
        X_train, y_train = dataset_to_numpy(train_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        n_channels = X_train.shape[1]
        n_times = X_train.shape[2]
        
        print(f"Input: {n_channels} channels x {n_times} time points")
        print(f"Training: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Create model
        model = create_model(n_channels, n_times)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        
        # Setup training
        clf, train_ds, valid_ds = setup_training(model, train_set, valid_set, test_set)
        
        # Train
        print("\nStarting training...")
        clf.fit(train_ds, y=None)
        
        # Evaluate
        test_acc = clf.score(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        print(f"\n{'='*40}")
        print(f"TEST ACCURACY: {test_acc:.4f}")
        print(f"{'='*40}")
        
        # Save model
        model_path = os.path.join(patient_dir, 'stscnn_model.pt')
        torch.save(model.state_dict(), model_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Also save the channel weights to see which channels are important
        channel_weights = model.channel_weights.detach().cpu().numpy()
        weights_path = os.path.join(patient_dir, 'channel_weights.npy')
        np.save(weights_path, channel_weights)
        print(f"Channel weights saved to: {weights_path}")
        
        return True, test_acc
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
        
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == '__main__':
    import multiprocessing
    import numpy as np
    multiprocessing.freeze_support()
    
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    results_dir = '../STSCNN results/FinalWorking2'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run on patient 2 first
    #pn_values = [3,4,7,16,17,18,19,20,29,35,41]
    pn_values = [32]
    
    print("=" * 80)
    print("Working STSCNN - Using architecture that achieved 48% baseline")
    print("With added learnable channel weights (the STSCNN innovation)")
    print("=" * 80)
    
    all_results = {}
    
    for pn in pn_values:
        print(f"\n{'='*80}")
        print(f"Running Patient {pn}")
        print(f"{'='*80}")
        
        success, acc = run_for_pn(pn, results_dir)
        all_results[f"P{pn}"] = acc if success else None
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for key, acc in all_results.items():
        if acc is not None:
            print(f"{key}: {acc:.4f}")
        else:
            print(f"{key}: FAILED")