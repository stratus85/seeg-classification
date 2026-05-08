import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from datetime import datetime
from scipy.io import loadmat
import hdf5storage

# ============================================================================
# Helper Functions for Electrode Information
# ============================================================================

def get_electrode_groups(patient_id, base_path='../Resources/EleCTX_Files_2018_10_26'):
    """
    Determine which channels belong to which electrode using electrode names.
    Returns: list of lists, where each inner list contains channel indices for one electrode
    """
    filepath = os.path.join(base_path, f'P{patient_id}', 'electrodes_Final_Norm.mat')
    
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