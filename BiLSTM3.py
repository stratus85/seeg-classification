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


class MultiHeadSelfAttention(nn.Module):
    """
    Proper multi-head self-attention for time series.
    This is the attention mechanism that actually works.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.3):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        Returns: (batch, seq_len, hidden_size) - same shape
        """
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attention_weights.mean(dim=1)  # Average over heads for visualization


class ImprovedBiLSTMWithAttention(nn.Module):
    """
    Improved Bi-LSTM with attention that actually works.
    
    Key improvements:
    1. Attention AFTER LSTM (not before)
    2. Residual connection around attention
    3. Layer normalization for stability
    4. Dropout at correct positions
    """
    
    def __init__(self, n_channels, n_times, n_classes=5, 
                 hidden_size=64,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.4,
                 use_attention=True):
        super(ImprovedBiLSTMWithAttention, self).__init__()
        
        self.use_attention = use_attention
        
        # Aggressive channel reduction (critical for SEEG)
        reduced_channels = min(32, max(16, n_channels // 6))
        self.channel_reduction = nn.Sequential(
            nn.Conv1d(n_channels, reduced_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(reduced_channels),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        # Bi-LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=reduced_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer norm after LSTM
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head self-attention (only if enabled)
        if use_attention:
            self.attention = MultiHeadSelfAttention(
                hidden_size=hidden_size * 2,
                num_heads=num_heads,
                dropout=dropout
            )
            self.attention_dropout = nn.Dropout(dropout)
        
        # Global pooling options
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, channels, time)
        
        # Channel reduction
        x = self.channel_reduction(x)  # (batch, reduced, time)
        
        # Permute for LSTM
        x = x.permute(0, 2, 1)  # (batch, time, reduced)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        
        # Layer norm
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention (if enabled)
        if self.use_attention:
            attended_out, attention_weights = self.attention(lstm_out)
            # Residual connection
            lstm_out = lstm_out + self.attention_dropout(attended_out)
        
        # Global pooling over time
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden*2, time)
        pooled = self.global_pool(lstm_out).squeeze(-1)  # (batch, hidden*2)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

def create_model(n_channels, n_times, n_classes=5, model_type='improved', use_attention=True):
    """Factory function with improved defaults"""
    return ImprovedBiLSTMWithAttention(n_channels, n_times, n_classes, use_attention=use_attention)


def setup_training(model, train_set, valid_set, test_set, device='cuda', max_epochs=150):
    """Improved training setup with better regularization"""
    
    X_train, y_train = dataset_to_numpy(train_set)
    X_valid, y_valid = dataset_to_numpy(valid_set)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.LongTensor(y_valid)
    
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    n_train_samples = len(X_train)
    
    # Optimized hyperparameters
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


def run_for_pn(pn, results_dir, model_type='improved', use_attention=True):
    """Run training for a specific pn"""
    
    patient_dir = os.path.join(results_dir, f'P{pn}')
    os.makedirs(patient_dir, exist_ok=True)
    
    attention_str = "with_attention" if use_attention else "no_attention"
    output_file = os.path.join(patient_dir, f'{model_type}_{attention_str}.txt')
    
    suppress_mne_logging()
    
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"Improved Bi-LSTM - Patient {pn}")
        print(f"Model: {model_type}, Attention: {use_attention}")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Load data
        train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
        
        X_train, y_train = dataset_to_numpy(train_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        n_channels = X_train.shape[1]
        n_times = X_train.shape[2]
        n_train_samples = len(X_train)
        
        print(f"Input: {n_channels} channels x {n_times} time points")
        print(f"Training: {n_train_samples} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Create model
        model = create_model(
            n_channels, n_times, n_classes=5,
            model_type=model_type,
            use_attention=use_attention
        )
        
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
        model_path = os.path.join(patient_dir, f'{model_type}_{attention_str}.pt')
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
    
    results_dir = '../BiLSTM results/improved_attention3'
    os.makedirs(results_dir, exist_ok=True)
    
    # Test configurations on patients where attention showed promise
    # P13 and P41 saw benefit, P17 and P19 were close
    pn_values = [2,3,4,10,13,17,19,29,32,41]  # Focus on 2kHz patients
    
    # Compare with and without attention
    configs = [
        {'model_type': 'improved', 'use_attention': True, 'name': 'with_attention'},
        {'model_type': 'improved', 'use_attention': False, 'name': 'no_attention'},
        #{'model_type': 'optimized_baseline', 'use_attention': False, 'name': 'optimized_baseline'},
    ]
    
    print("=" * 80)
    print("Improved Bi-LSTM with Proper Attention Mechanism")
    print(f"Results directory: {results_dir}")
    print("=" * 80)
    print("\nKey improvements:")
    print("1. Multi-head self-attention (not simple linear)")
    print("2. Residual connection around attention")
    print("3. Layer normalization for stability")
    print("4. Better channel reduction (Conv1d with kernel=5)")
    print("5. Deeper LSTM (2 layers for baseline)")
    print("6. Adaptive learning rate scheduling")
    print("=" * 80)
    
    all_results = {}
    
    for pn in pn_values:
        for config in configs:
            print(f"\n{'='*80}")
            print(f"Running Patient {pn}, {config['name']}")
            print(f"{'='*80}")
            
            success, acc = run_for_pn(
                pn, results_dir,
                model_type=config['model_type'],
                use_attention=config['use_attention']
            )
            
            key = f"P{pn}_{config['name']}"
            all_results[key] = acc if success else None
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Improved Bi-LSTM")
    print("=" * 80)
    
    # Group by patient
    for pn in pn_values:
        print(f"\nPatient {pn}:")
        for config in configs:
            key = f"P{pn}_{config['name']}"
            acc = all_results.get(key)
            if acc is not None:
                print(f"  {config['name']}: {acc:.4f} ({acc*100:.1f}%)")
            else:
                print(f"  {config['name']}: FAILED")
    
    # Save summary
    summary_path = os.path.join(results_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Improved Bi-LSTM Results\n")
        f.write(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        for key, acc in all_results.items():
            if acc is not None:
                f.write(f"{key}: {acc:.4f}\n")
            else:
                f.write(f"{key}: FAILED\n")
    
    print(f"\nSummary saved to: {summary_path}")