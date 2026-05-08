import torch
import torch.nn as nn
import os
import sys
from datetime import datetime
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skorch.callbacks import EarlyStopping, EpochScoring
from preprocessing import save_or_preprocess, dataset_to_numpy

class ResidualSEEGNet(nn.Module):
    """
    Residual Network for SEEG classification as described in the paper.
    
    Architecture:
    - Block 1: Initial convolution to expand 2D input to 3D cube
    - Block 2: Residual block with identity connection (two 3×3 conv layers)
    - Block 3: Residual block with 1×1 convolutional connection
    - Block 4: Global pooling and linear layer to 1×5 output
    
    Input shape: (batch_size, channels, time_points)
    Output shape: (batch_size, 5) - 5 gesture classes
    """
    def __init__(self, n_channels, n_times, n_classes=5, n_filters=32):
        super(ResidualSEEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_filters = n_filters
        
        # Define all layers except the final FC
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.block2 = RegularizedResidualBlock(
            in_channels=n_filters,
            out_channels=n_filters * 2,
            use_identity=True,
            stride=2
        )
        
        self.block3 = RegularizedResidualBlock(
            in_channels=n_filters * 2,
            out_channels=n_filters * 4,
            use_identity=False,
            stride=2
        )
        
        # Block 4: Third residual block - maintain channels, no downsampling
        # Dimensions match (128 → 128), so identity skip will be used
        self.block4 = RegularizedResidualBlock(
            in_channels=n_filters * 4,
            out_channels=n_filters * 4,
            use_identity=True,       # Identity skip (dimensions match, stride=1)
            stride=1,
        )
        
        # Block 5: Fourth residual block - maintain channels, no downsampling
        # Dimensions match (128 → 128), so identity skip will be used
        self.block5 = RegularizedResidualBlock(
            in_channels=n_filters * 4,
            out_channels=n_filters * 4,
            use_identity=True,       # Identity skip (dimensions match, stride=1)
            stride=1,
        )

        self.block6 = RegularizedResidualBlock(
            in_channels=n_filters * 4,
            out_channels=n_filters * 4,
            use_identity=True,       # Identity skip (dimensions match, stride=1)
            stride=1,
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate FC input size dynamically
        self._initialize_fc_layer()
            
    def _initialize_fc_layer(self):
        """Initialize FC layer with correct input size"""
        with torch.no_grad():
            # Create dummy input
            dummy = torch.zeros(1, 1, self.n_channels, self.n_times)
            
            # Forward through feature extractor
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.global_pool(x)
            
            # Get flattened size
            fc_input_size = x.view(1, -1).size(1)
            
            print(f"Automatically detected FC input size: {fc_input_size}")
            
            # Create FC layer
            self.fc = nn.Linear(fc_input_size, self.n_classes)
    
    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block as described in the paper.
    
    Composed of two 3×3 convolutional layers with BatchNorm and ReLU.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_identity: If True, use identity connection (if dimensions match)
                     If False, use 1×1 convolution for the skip connection
        stride: Stride for the first convolution (controls downsampling)
    """
    
    def __init__(self, in_channels, out_channels, use_identity=True, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.use_identity = use_identity
        
        # First 3×3 convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3×3 convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if not use_identity or in_channels != out_channels or stride != 1:
            # Use 1×1 convolution for the skip connection
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=1, stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        identity = self.skip(identity)
        
        # Add skip connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out

class RegularizedResidualBlock(nn.Module):
    """
    Regularized Residual Block with multiple techniques to prevent overfitting.
    
    Regularization features:
    1. Dropout2d after first activation (spatial dropout)
    2. Label smoothing compatible (via loss function, not in block)
    3. Optional weight standardization
    4. Optional spectral normalization
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_identity: If True, use identity connection (if dimensions match)
                     If False, use 1×1 convolution for the skip connection
        stride: Stride for the first convolution (controls downsampling)
        dropout_rate: Dropout probability (0 = no dropout)
        use_weight_standardization: If True, use WS conv instead of regular conv
    """
    
    def __init__(self, in_channels, out_channels, use_identity=True, stride=1, 
                 dropout_rate=0.5, use_weight_standardization=False):
        super(RegularizedResidualBlock, self).__init__()
        
        self.use_identity = use_identity
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.use_weight_standardization = use_weight_standardization
        
        # Choose convolution type
        conv_class = nn.Conv2d
        
        # First 3×3 convolution (may downsample)
        self.conv1 = conv_class(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3×3 convolution
        self.conv2 = conv_class(
            out_channels, out_channels, 
            kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout after first activation (spatial dropout works well for EEG)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if not use_identity or in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                conv_class(
                    in_channels, out_channels, 
                    kernel_size=1, stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Apply dropout after first activation
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        identity = self.skip(identity)
        
        # Add skip connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out

def create_model(n_channels, n_times, n_classes=5, n_filters=32):
    """
    Factory function to create the residual SEEG model.
    
    Args:
        n_channels: Number of input channels
        n_times: Number of time points
        n_classes: Number of output classes
        n_filters: Base number of filters
    
    Returns:
        ResidualSEEGNet model
    """
    return ResidualSEEGNet(n_channels, n_times, n_classes, n_filters)


# Training setup function
def setup_training(model, train_set, valid_set, test_set, device='cuda', max_epochs=200):
    """
    Set up training with skorch wrapper.
    
    Args:
        model: PyTorch model
        train_set: Braindecode WindowsDataset for training
        valid_set: Braindecode WindowsDataset for validation
        test_set: Braindecode WindowsDataset for testing
        device: 'cuda' or 'cpu'
        max_epochs: Maximum number of training epochs
    
    Returns:
        Skorch NeuralNetClassifier ready for training
    """
    
    # Extract data from WindowsDataset
    X_train, y_train = dataset_to_numpy(train_set)
    X_valid, y_valid = dataset_to_numpy(valid_set)

    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.LongTensor(y_valid)
    
    # Create dataset objects
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(X_valid, y_valid)
    
    # Create skorch wrapper
    clf = NeuralNetClassifier(
        model,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=1e-6,
        batch_size=32,
        max_epochs=max_epochs,
        device=device,
        train_split=predefined_split(valid_ds),
        callbacks=[
            EarlyStopping(patience=20),
            EpochScoring('accuracy', name='valid_acc', lower_is_better=False),
        ],
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__num_workers=4,
        verbose=1
    )
    
    return clf, train_ds, valid_ds

class Tee:
    """Class to redirect stdout to both file and console"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def run_for_pn(pn, results_dir):
    """
    Run training for a specific pn value and save output to file.
    Returns True if successful, False if error occurred.
    """
    # Create output file path
    output_file = os.path.join(results_dir, f'pn_{pn}.txt')
    
    # Redirect stdout to both console and file
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"Starting run for pn = {pn}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Preprocess data
        train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
        
        # Convert datasets to numpy
        X_train, y_train = dataset_to_numpy(train_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        n_channels = X_train.shape[1]
        n_times = X_train.shape[2]
        
        print(f"Input shape: {n_channels} channels × {n_times} time points")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create model
        model = create_model(n_channels, n_times, n_classes=5)
        
        # Setup training
        clf, train_ds, valid_ds = setup_training(model, train_set, valid_set, test_set)
        
        # Train
        print("Starting training...")
        clf.fit(train_ds, y=None)
        
        # Evaluate
        test_acc = clf.score(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        print(f"Test accuracy: {test_acc:.4f}")
        
        print("=" * 80)
        print(f"Finished run for pn = {pn}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print("=" * 80)
        print(f"ERROR for pn = {pn}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=" * 80)
        return False
        
    finally:
        # Restore original stdout and close the tee
        sys.stdout = original_stdout
        tee.close()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set, ignore
        pass
    
    # Define results directory
    results_dir = '../ResNet results/Overnight4redo'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Define pn values to run (2 to 23, skipping problematic ones)
    # pn_values = [pn for pn in range(2, 31) if pn not in [6, 11, 12, 15, 27, 28]]
    pn_values = [32]
    # Also skip any others that don't work
    # Add more here if needed: if pn not in [12, 15, x, y, z]
    
    print("=" * 80)
    print(f"Starting batch processing for pn values: {pn_values}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Total runs: {len(pn_values)}")
    print("=" * 80)
    
    # Track results
    successful = []
    failed = []
    
    # Run for each pn value
    for pn in pn_values:
        print(f"\n{'=' * 80}")
        print(f"Processing pn = {pn}...")
        print(f"{'=' * 80}")
        
        success = run_for_pn(pn, results_dir)
        
        if success:
            successful.append(pn)
        else:
            failed.append(pn)
        
        print(f"\nCompleted pn = {pn} (Success: {success})")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    print(f"Total successful: {len(successful)}/{len(pn_values)}")
    print(f"Results saved in: {results_dir}")
    print("=" * 80)