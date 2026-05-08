import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
import hdf5storage
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import save_or_preprocess, dataset_to_numpy

def get_electrode_groups(patient_id, base_path='../Resources/EleCTX_Files_2018_10_26'):
    """
    Determine which channels belong to which electrode using electrode names.
    Returns: list of lists, where each inner list contains channel indices for one electrode
    """
    filepath = os.path.join(base_path, f'P{patient_id}', 'electrodes_Final_Norm.mat')

    if patient_id == 41:
        filepath = os.path.join(base_path, f'P{patient_id}', 'electrodes_Final_Anatomy_wm_All.mat')
    
    if not os.path.exists(filepath):
        print(f"Warning: Electrode file not found at {filepath}")
        return None
    
    mat_data = hdf5storage.loadmat(filepath)
    elec_info = mat_data['elec_Info_Final_wm'][0]
    
    # Get the name field which contains electrode labels like 'A1', 'A2', 'B1', 'B2', etc.
    name_cell = elec_info['name'][0]
    
    # Group by electrode letter (A, B, C, etc.)
    electrode_groups = {}
    
    for channel_idx, name in enumerate(name_cell):
        # Extract the electrode letter (first character) from names like 'A1', 'A2', 'B1', etc.
        if len(name) > 0:
            # Extract electrode letter (first character)
            electrode_letter = name[0][0][0]

            if electrode_letter not in electrode_groups:
                electrode_groups[electrode_letter] = []
            electrode_groups[electrode_letter].append(channel_idx)
    
    # Convert to list of lists and sort by electrode letter
    groups = [electrode_groups[letter] for letter in sorted(electrode_groups.keys())]
    
    # Print summary
    print(f"Found {len(groups)} electrodes:")
    for i, (letter, group) in enumerate(sorted(electrode_groups.items())):
        print(f"  Electrode {letter}: {len(group)} channels (indices {group[0]}-{group[-1]})")
    print(f"Total channels: {sum(len(g) for g in groups)}")
    
    return groups


class ElectrodeAwareChannelReducer(nn.Module):
    """
    Reduce channels WITHIN each electrode separately using group convolution.
    This preserves the fact that channels on the same electrode are highly correlated.
    """
    def __init__(self, n_channels, electrode_groups, output_channels_per_electrode=4):
        super().__init__()
        
        self.electrode_groups = electrode_groups
        self.n_electrodes = len(electrode_groups)
        self.output_channels = self.n_electrodes * output_channels_per_electrode
        
        # Create a 1D convolution for each electrode
        self.electrode_convs = nn.ModuleList()
        
        for group in electrode_groups:
            n_ch_in_group = len(group)
            # Conv1d: input channels = n_ch_in_group, output = output_channels_per_electrode
            conv = nn.Conv1d(n_ch_in_group, output_channels_per_electrode, 
                            kernel_size=3, padding=1)
            self.electrode_convs.append(conv)
        
        # Temporal smoothing after electrode processing
        self.temporal_smooth = nn.Sequential(
            nn.Conv1d(self.output_channels, self.output_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.output_channels),
            nn.ELU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        batch_size, _, n_times = x.shape
        
        # Process each electrode group separately
        electrode_outputs = []
        
        for conv, group in zip(self.electrode_convs, self.electrode_groups):
            # Extract channels for this electrode
            electrode_data = x[:, group, :]  # (batch, n_ch_in_group, time)
            # Apply convolution
            out = conv(electrode_data)  # (batch, output_channels_per_electrode, time)
            electrode_outputs.append(out)
        
        # Concatenate all electrodes
        x = torch.cat(electrode_outputs, dim=1)  # (batch, output_channels, time)
        
        # Temporal smoothing
        x = self.temporal_smooth(x)
        
        return x

class SimpleTemporalAttention(nn.Module):
    """Single-head temporal attention - minimal parameters"""
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, time, hidden)
        weights = torch.tanh(self.attention(x))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        context = (x * weights).sum(dim=1)
        return context, weights

class ElectrodeAwareBiLSTM(nn.Module):
    """
    BiLSTM with electrode-aware channel reduction.
    This keeps the successful BiLSTM architecture but adds electrode knowledge.
    """
    def __init__(self, n_channels, n_times, n_classes=5,
                 electrode_groups=None,
                 output_channels_per_electrode=4,
                 lstm_hidden=64,
                 lstm_layers=2,
                 dropout=0.4):
        super().__init__()
        
        # Electrode-aware channel reduction
        if electrode_groups is not None:
            self.channel_reducer = ElectrodeAwareChannelReducer(
                n_channels, electrode_groups, output_channels_per_electrode
            )
            lstm_input_size = self.channel_reducer.output_channels
        else:
            # Fallback: simple 1x1 conv if no electrode info
            self.channel_reducer = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=1),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            lstm_input_size = 32
        
        # BiLSTM (same as your working version)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        
        # Simple attention (replaces multi-head)
        self.attention = SimpleTemporalAttention(lstm_hidden * 2, dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes)
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
        
        # Channel reduction (electrode-aware)
        x = self.channel_reducer(x)  # (batch, reduced_channels, time)
        
        # Permute for LSTM
        x = x.permute(0, 2, 1)  # (batch, time, reduced_channels)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        
        # Layer norm
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention
        context, _ = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Classification
        logits = self.classifier(context)
        
        return logits

# ============================================================================
# Training Functions (same as your working version)
# ============================================================================

def train_model(model, train_loader, valid_loader, device, epochs=150, lr=0.0005):
    """Train the model - using your successful hyperparameters"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    best_valid_acc = 0
    patience_counter = 0
    patience_limit = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += batch_y.size(0)
                valid_correct += predicted.eq(batch_y).sum().item()
        
        valid_acc = 100. * valid_correct / valid_total
        
        scheduler.step(valid_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Valid Acc: {valid_acc:.2f}% | Valid Loss: {valid_loss:.2f}%")
    
    model.load_state_dict(best_model_state)
    return model, best_valid_acc

def test_model(model, test_loader, device):
    """Test the model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ============================================================================
# Main Execution
# ============================================================================

def run_patient(patient_id, results_base_dir):
    """Run electrode-aware BiLSTM for a single patient"""
    
    patient_dir = os.path.join(results_base_dir, f'P{patient_id}')
    os.makedirs(patient_dir, exist_ok=True)
    
    # Redirect output
    log_file = os.path.join(patient_dir, 'training_log.txt')
    original_stdout = sys.stdout
    log_f = open(log_file, 'w')
    sys.stdout = log_f
    
    print("="*60)
    print(f"Electrode-Aware BiLSTM - Patient {patient_id}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Load data
        print("\nLoading data...")
        train_set, valid_set, test_set = save_or_preprocess(patient_id, cache_dir='../Preprocessed')
        
        X_train, y_train = dataset_to_numpy(train_set)
        X_valid, y_valid = dataset_to_numpy(valid_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        n_channels = X_train.shape[1]
        n_times = X_train.shape[2]
        n_train = len(X_train)
        
        print(f"Channels: {n_channels}, Time points: {n_times}")
        print(f"Train: {n_train}, Valid: {len(X_valid)}, Test: {len(X_test)}")
        
        # Get electrode groups
        print("\nLoading electrode information...")
        electrode_groups = get_electrode_groups(patient_id)
        
        if electrode_groups is None:
            print("WARNING: No electrode info found. Using fallback channel reduction.")
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        model = ElectrodeAwareBiLSTM(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=5,
            electrode_groups=electrode_groups,
            output_channels_per_electrode=4,  # Each electrode -> 4 features
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.4
        )
        model = model.to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        
        # Data loaders
        batch_size = 64 if n_train >= 6000 else 32
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
                                 batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid)), 
                                 batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
                                batch_size=batch_size, shuffle=False)
        
        # Learning rate (use your successful values)
        lr = 0.0008 if n_train >= 6000 else 0.0005
        
        # Train
        print("\nStarting training...")
        model, best_valid_acc = train_model(model, train_loader, valid_loader, device, 
                                           epochs=150, lr=lr)
        
        # Test
        test_acc = test_model(model, test_loader, device)
        
        print("\n" + "="*60)
        print(f"RESULTS - Patient {patient_id}")
        print("="*60)
        print(f"Best validation accuracy: {best_valid_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        print("="*60)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(patient_dir, f'model_p{patient_id}.pt'))
        model_path = os.path.join(patient_dir, 'gcnlstm.pt')

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
        log_f.close()
        print(f"Results saved to {patient_dir}")

# ============================================================================
# Run All Patients
# ============================================================================

if __name__ == '__main__':
    patients = [2,3,4,10,13,17,19,29,32,41]
    results_dir = '../GCN-LSTM results/test1.2'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*70)
    print("Electrode-Aware BiLSTM with Simple Attention")
    print("="*70)
    
    results = {}
    for patient_id in patients:
        print(f"\n{'='*70}")
        print(f"Patient {patient_id}")
        print(f"{'='*70}")
        
        success, acc = run_patient(patient_id, results_dir)
        if success:
            results[patient_id] = acc
            print(f"Patient {patient_id}: {acc:.2f}%")
        else:
            print(f"Patient {patient_id}: FAILED")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for pid, acc in results.items():
        print(f"P{pid}: {acc:.2f}%")
    
    if results:
        avg = np.mean(list(results.values()))
        print(f"\nAverage: {avg:.2f}%")