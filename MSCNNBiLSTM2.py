import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from datetime import datetime
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import preprocess, dataset_to_numpy, save_or_preprocess


# ============================================================================
# SEEGNet components (as provided)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block as used in SEEG-Net"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        batch, channels, time = x.shape
        squeeze = self.global_avg_pool(x).view(batch, channels)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch, channels, 1)
        return x * excitation


class AggregationLayer(nn.Module):
    """Aggregation Layer from SEEG-Net with residual SE block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        shortcut = self.shortcut(x)
        out = out + shortcut
        return F.relu(out)


class ConvBlock(nn.Module):
    """Single convolution block with batch norm, activation, and optional pooling"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, use_pool=True, dropout=0.3):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(2) if use_pool else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.use_pool = use_pool
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.use_pool:
            x = self.pool(x)
        x = self.dropout(x)
        return x


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN from SEEG-Net with consistent temporal dimensions"""
    def __init__(self, n_channels, n_times, 
                 kernel_sizes=[50, 400, 1250],
                 n_filters=32,
                 dropout=0.3):
        super().__init__()
        
        # Scale kernels for 2000Hz sampling (paper used 5000Hz)
        scale_factor = 2000 / 5000
        adjusted_kernels = [max(3, int(k * scale_factor)) for k in kernel_sizes]
        
        print(f"MultiScaleCNN: Using kernel sizes {adjusted_kernels} (scaled from {kernel_sizes})")
        
        self.n_times = n_times
        
        # Branch 1: Small kernel
        self.branch1 = nn.Sequential(
            ConvBlock(n_channels, n_filters, adjusted_kernels[0], use_pool=True, dropout=dropout),
            ConvBlock(n_filters, n_filters * 2, adjusted_kernels[0] // 2, use_pool=True, dropout=dropout),
            ConvBlock(n_filters * 2, n_filters * 2, adjusted_kernels[0] // 4, use_pool=False, dropout=dropout),
        )
        
        # Branch 2: Medium kernel
        self.branch2 = nn.Sequential(
            ConvBlock(n_channels, n_filters, adjusted_kernels[1], use_pool=True, dropout=dropout),
            ConvBlock(n_filters, n_filters * 2, adjusted_kernels[1] // 2, use_pool=True, dropout=dropout),
            ConvBlock(n_filters * 2, n_filters * 2, adjusted_kernels[1] // 4, use_pool=False, dropout=dropout),
        )
        
        # Branch 3: Wide kernel
        self.branch3 = nn.Sequential(
            ConvBlock(n_channels, n_filters, adjusted_kernels[2], use_pool=True, dropout=dropout),
            ConvBlock(n_filters, n_filters * 2, adjusted_kernels[2] // 2, use_pool=True, dropout=dropout),
            ConvBlock(n_filters * 2, n_filters * 2, adjusted_kernels[2] // 4, use_pool=False, dropout=dropout),
        )
        
        # Adaptive pooling to ensure consistent time dimension (400 -> 200 -> 100)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)
        self.output_channels = n_filters * 2 * 3  # 32*2*3 = 192
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b1 = self.adaptive_pool(b1)
        b2 = self.adaptive_pool(b2)
        b3 = self.adaptive_pool(b3)
        out = torch.cat([b1, b2, b3], dim=1)
        return out


class BiLSTMWithAttention(nn.Module):
    """BiLSTM with Attention Mechanism (BiLSTM-AM) from SEEG-Net"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention_weights = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_context = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        u_t = torch.tanh(self.attention_weights(lstm_out))
        attention_scores = self.attention_context(u_t)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return context, attention_weights


class SEEGNet(nn.Module):
    """Complete SEEG-Net implementation for motor movement classification"""
    def __init__(self, n_channels, n_times, n_classes=5,
                 cnn_filters=32, lstm_hidden=128, dropout=0.4):
        super().__init__()
        
        self.mscnn = MultiScaleCNN(
            n_channels=n_channels,
            n_times=n_times,
            kernel_sizes=[50, 400, 1250],
            n_filters=cnn_filters,
            dropout=dropout
        )
        
        self.aggregation = AggregationLayer(
            in_channels=self.mscnn.output_channels,
            out_channels=self.mscnn.output_channels
        )
        
        self.bilstm_attention = BiLSTMWithAttention(
            input_size=self.mscnn.output_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes)
        )
        
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.mscnn(x)                 # (batch, 192, 100)
        x = self.aggregation(x)           # (batch, 192, 100)
        x = x.permute(0, 2, 1)            # (batch, 100, 192)
        context, _ = self.bilstm_attention(x)   # (batch, hidden*2)
        logits = self.classifier(context)
        return logits


def create_model(n_channels, n_times, n_classes=5, model_type='seegnet', **kwargs):
    """Factory function to create SEEGNet model."""
    return SEEGNet(n_channels, n_times, n_classes,
                   cnn_filters=32, lstm_hidden=128, dropout=0.4)


def setup_training(model, train_set, valid_set, test_set, device='cuda', max_epochs=150):
    """Same training setup as original"""
    X_train, y_train = dataset_to_numpy(train_set)
    X_valid, y_valid = dataset_to_numpy(valid_set)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.LongTensor(y_valid)
    
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    n_train_samples = len(X_train)
    
    if n_train_samples >= 6000:
        lr = 0.001
        weight_decay = 1e-4
    else:
        lr = 0.0008
        weight_decay = 1e-5
    
    clf = NeuralNetClassifier(
        model,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=32,
        max_epochs=max_epochs,
        device=device,
        train_split=predefined_split(valid_ds),
        callbacks=[
            EarlyStopping('valid_acc', patience=20, threshold=0.0005, lower_is_better=False),
            EpochScoring('accuracy', name='valid_acc', lower_is_better=False),
            LRScheduler(policy=ReduceLROnPlateau, mode='min', patience=7, factor=0.5)
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
        try: self.log.write(message)
        except: pass
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def run_for_pn(pn, results_dir, model_type='seegnet'):
    """Run training for a specific patient using SEEGNet"""
    patient_dir = os.path.join(results_dir, f'P{pn}')
    os.makedirs(patient_dir, exist_ok=True)
    
    output_file = os.path.join(patient_dir, f'{model_type}.txt')
    suppress_mne_logging()
    
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"SEEGNet - Patient {pn}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
        X_train, y_train = dataset_to_numpy(train_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        n_channels = X_train.shape[1]
        n_times = X_train.shape[2]
        n_train_samples = len(X_train)
        
        print(f"Input: {n_channels} channels x {n_times} time points")
        print(f"Training: {n_train_samples} samples")
        print(f"Test: {len(X_test)} samples")
        
        model = create_model(n_channels, n_times, n_classes=5, model_type=model_type)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        
        clf, train_ds, valid_ds = setup_training(model, train_set, valid_set, test_set)
        
        print("\nStarting training...")
        clf.fit(train_ds, y=None)
        
        test_acc = clf.score(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        print(f"\n{'='*40}")
        print(f"TEST ACCURACY: {test_acc:.4f}")
        print(f"{'='*40}")
        
        model_path = os.path.join(patient_dir, f'{model_type}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
        print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    results_dir = '../SEEGNet2 results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Same patients as in your table
    pn_values = [2, 3, 4, 10, 13, 17, 19, 29, 32, 41]
    
    print("=" * 80)
    print("SEEGNet Implementation")
    print(f"Results directory: {results_dir}")
    print("=" * 80)
    
    all_results = {}
    for pn in pn_values:
        print(f"\n{'='*80}")
        print(f"Running Patient {pn}")
        print(f"{'='*80}")
        success, acc = run_for_pn(pn, results_dir, model_type='seegnet')
        all_results[f"P{pn}"] = acc if success else None
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - SEEGNet")
    print("=" * 80)
    for pn in pn_values:
        acc = all_results.get(f"P{pn}")
        if acc is not None:
            print(f"Patient {pn}: {acc:.4f} ({acc*100:.1f}%)")
        else:
            print(f"Patient {pn}: FAILED")
    
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"SEEGNet Results\nRun date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        for key, acc in all_results.items():
            f.write(f"{key}: {acc:.4f}\n" if acc else f"{key}: FAILED\n")
    
    print(f"\nSummary saved to: {summary_path}")