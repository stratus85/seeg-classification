import numpy as np
import os
import matplotlib.pyplot as plt

def load_channel_weights(file_path):
    """
    Load channel weights from a .npy file.
    
    Args:
        file_path (str): Path to the .npy file.
    
    Returns:
        numpy.ndarray: Array of channel weights.
    """
    # Resolve the relative path from the script's location (optional)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, file_path)
    
    try:
        # Load the .npy file
        weights = np.load(absolute_path)
        print(f"Successfully loaded channel weights from:\n{absolute_path}")
        print(f"Shape: {weights.shape}")
        print(f"Data type: {weights.dtype}")
        return weights
    except FileNotFoundError:
        print(f"Error: File not found at {absolute_path}")
        raise
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        raise

if __name__ == "__main__":
    # Path relative to the script's location
    npy_path = "../STSCNN results/FinalWorking/P41/channel_weights.npy"
    
    # Load and extract weights
    channel_weights = load_channel_weights(npy_path)
    
    # Print the weights (or process further)
    print("\nChannel weights array:")
    print(channel_weights)
    
    # Extract the 1D array of shape (189,)
    weights_1d = channel_weights[0, 0, :, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 190), weights_1d, marker='o', linestyle='-', markersize=3)
    plt.ylim(0,1.5)
    plt.xlabel('Channel Index')
    plt.ylabel('Weight Value')
    plt.title('Channel Weights (189 channels) for dataset 10')
    plt.grid(True)
    plt.show()