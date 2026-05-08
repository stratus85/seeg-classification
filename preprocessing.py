import pywt
import numpy as np
import mne
import hdf5storage
from sklearn.preprocessing import StandardScaler
from braindecode.datasets import create_from_mne_epochs
from chn_settings import get_channel_setting

def wavelet_denoise(data, wavelet='db4', level=None, method='soft', threshold_scale='universal'):
    """
    Apply wavelet denoising to each channel independently
    
    Parameters:
    -----------
    data : ndarray, shape (n_channels, n_times)
        Input data to denoise
    wavelet : str
        Wavelet type to use (e.g., 'db4', 'sym8', 'coif5')
    level : int or None
        Decomposition level (None = maximum possible)
    method : str
        Thresholding method: 'soft' or 'hard'
    threshold_scale : str
        Threshold scaling: 'universal', 'sqrt', or 'level-dependent'
    
    Returns:
    --------
    denoised : ndarray, same shape as data
        Denoised data
    """
    
    n_channels, n_times = data.shape
    
    # Determine max decomposition level if not specified
    if level is None:
        level = pywt.dwt_max_level(n_times, pywt.Wavelet(wavelet).dec_len)
        level = min(level, 8)  # Cap at 8 levels to avoid over-decomposition
    
    denoised = np.zeros_like(data)
    
    for ch in range(n_channels):
        # Get channel data
        signal = data[ch, :]
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Estimate noise standard deviation from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        if threshold_scale == 'universal':
            # Universal threshold: sigma * sqrt(2*log(n))
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        elif threshold_scale == 'sqrt':
            # sqrt-based threshold
            threshold = sigma * np.sqrt(2 * np.log(len(signal))) / 2
        elif threshold_scale == 'level-dependent':
            # Level-dependent threshold (more aggressive at finer scales)
            n_levels = len(coeffs) - 1
            thresholds = []
            for i in range(1, n_levels + 1):
                # Finer scales get smaller threshold (preserve more details)
                level_thresh = sigma * np.sqrt(2 * np.log(len(signal))) / (2 ** (i/2))
                thresholds.append(level_thresh)
        
        # Apply thresholding to detail coefficients (skip approximation)
        if threshold_scale == 'level-dependent':
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], thresholds[i-1], mode=method)
        else:
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode=method)
        
        # Reconstruct signal
        denoised[ch, :] = pywt.waverec(coeffs, wavelet)[:n_times]
    
    return denoised

import pickle
import os

def save_or_preprocess(pn, cache_dir='../Preprocessed'):
    """
    Load preprocessed data from cache or run preprocessing if not cached.
    
    Args:
        pn: Patient number
        cache_dir: Directory to store cached preprocessing results
    
    Returns:
        train_set, valid_set, test_set
    """
    
    mne.set_log_level('ERROR')

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache file path
    cache_file = os.path.join(cache_dir, f'pn_{pn}_preprocessed.pkl')
    
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading cached preprocessing for pn={pn} from {cache_file}")
        with open(cache_file, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        return train_set, valid_set, test_set
    
    # If not cached, run preprocessing
    print(f"No cache found for pn={pn}. Running preprocessing...")
    train_set, valid_set, test_set = preprocess(pn)
    
    # Save to cache
    print(f"Saving cached preprocessing to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((train_set, valid_set, test_set), f)
    
    return train_set, valid_set, test_set

def preprocess(pn):

    UseChn,TrigChn, fs = get_channel_setting(pn)

    loadPath = f"../Resources/EleCTX_Files_2018_10_26/P{pn}/P{pn}_H1_1_Raw.mat"
    mat1 = hdf5storage.loadmat(loadPath)
    data1 = mat1['Data']
    del mat1

    loadPath = f"../Resources/EleCTX_Files_2018_10_26/P{pn}/P{pn}_H1_2_Raw.mat"
    mat2 = hdf5storage.loadmat(loadPath)
    data2 = mat2['Data']
    del mat2
    """
    loadPath = '../Resources/EleCTX_Files_2018_10_26/P2/P2_H1_3_Raw.mat'
    mat3 = hdf5storage.loadmat(loadPath)
    data3 = mat3['Data']
    del mat3
    """
    seegChn1 = data1[UseChn,:]
    stimChn1 = data1[TrigChn,:]

    seegChn2 = data2[UseChn,:]
    stimChn2 = data2[TrigChn,:]
    """
    seegChn3 = data3[UseChn,:]
    stimChn3 = data3[TrigChn,:]
    """
    seegChn = np.concatenate((seegChn1,seegChn2),axis=1)
    stimChn = np.concatenate((stimChn1,stimChn2),axis=1)

    scaler = StandardScaler()
    scaler.fit(seegChn)
    seegChn = scaler.transform(seegChn)

    seegChn64 = seegChn.astype(np.float64)
    seegChn_band_filt = mne.filter.filter_data(
        seegChn64,
        sfreq=fs,
        l_freq=0.5,
        h_freq=200.0,
        method='iir',
        iir_params=dict(order=4, ftype='butter')
    )

    seegChn_notch_filt = mne.filter.notch_filter(
        seegChn_band_filt,
        Fs=fs,
        freqs=50,
        method='iir',
        iir_params=dict(order=32, ftype='butter')
    )

    seegChn_notch_filt2 = mne.filter.notch_filter(
        seegChn_notch_filt,
        Fs=fs,
        freqs=100,
        method='iir',
        iir_params=dict(order=16, ftype='butter')
    )

    seegChn_notch_filt3 = mne.filter.notch_filter(
        seegChn_notch_filt2,
        Fs=fs,
        freqs=150,
        method='iir',
        iir_params=dict(order=16, ftype='butter')
    )

    seegChn_denoised = wavelet_denoise(
        seegChn_notch_filt3,
        wavelet='db4',        # Daubechies 4 (good for EEG/sEEG)
        level=5,              # Fixed decomposition level
        method='soft',        # Soft thresholding (more aggressive)
        threshold_scale='universal'  # Universal threshold
    )

    seegChn = seegChn_denoised

    # Get trigger channel indices
    stim_names = ["stim0", "stim1", "stim2", "stim3", "stim4"]

    if pn == 41:
        stimChn[0:5, :] = (stimChn[0:5, :] > 6000).astype(float)
    else:
        stimChn[0:5, :] = (stimChn[0:5, :] > 1e6).astype(float)

    # Combine SEEG data and triggers
    data = np.concatenate((seegChn, stimChn), axis=0)

    # stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
    chn_names=np.append(["seeg"]*len(UseChn),stim_names)
    print(data.shape)
    print(chn_names.shape)
    chn_types=np.append(["seeg"]*len(UseChn),["stim", "stim", "stim", "stim", "stim"])
    info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)

    mne.set_log_level('ERROR')
    raw = mne.io.RawArray(data, info)

    # gesture/events type: 1,2,3,4,5
    # minimum duration depends on sampling frequency, potentially change each dataset
    events0 = mne.find_events(raw, stim_channel='stim0', min_duration=4, verbose=False)
    events1 = mne.find_events(raw, stim_channel='stim1', min_duration=4, verbose=False)
    events2 = mne.find_events(raw, stim_channel='stim2', min_duration=4, verbose=False)
    events3 = mne.find_events(raw, stim_channel='stim3', min_duration=4, verbose=False)
    events4 = mne.find_events(raw, stim_channel='stim4', min_duration=4, verbose=False)

    # Assign proper class labels (0-4)
    events0[:, 2] = 0
    events1[:, 2] = 1
    events2[:, 2] = 2
    events3[:, 2] = 3
    events4[:, 2] = 4

    raw=raw.pick(["seeg"])
    # epoch from 0s to 4s with only  movement data.
    epoch0 = mne.Epochs(raw, events0, tmin=0, tmax=4, baseline=None, verbose=False)
    epoch1 = mne.Epochs(raw, events1, tmin=0, tmax=4, baseline=None, verbose=False)
    epoch2 = mne.Epochs(raw, events2, tmin=0, tmax=4, baseline=None, verbose=False)
    epoch3 = mne.Epochs(raw, events3, tmin=0, tmax=4, baseline=None, verbose=False)
    epoch4 = mne.Epochs(raw, events4, tmin=0, tmax=4, baseline=None, verbose=False)

    list_of_epochs = [epoch0, epoch1, epoch2, epoch3, epoch4]

    # 10 trials/epoch * 5 epochs or limb movements = 50 trials = 50 datasets
    # trials spread out unevenly across epochs
    # 1 dataset can be slided into multiple (depends on wind_size and stride) windows.
    windows_datasets = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=400,
        window_stride_samples=100,
        drop_last_window=False
    )

    # train/valid/test split based on description column
    desc = windows_datasets.description
    desc = desc.rename(columns={0: 'split'})

    # Get number of trials per epoch dynamically
    trials_per_epoch = epoch1.events.shape[0]
    total_trials = trials_per_epoch * 5  # 5 classes
    print(f"Trials per epoch: {trials_per_epoch}")
    print(f"Total trials: {total_trials}")

    import random
    random.seed(42)

    val_test_num = int(trials_per_epoch/10)  # Now 6 val and 6 test trials per class (since you tripled the data)
    # Or keep it proportional: val_test_num = int(trials_per_epoch * 0.2)  # 20% for validation+test

    # Select random trial indices (same for all classes)
    # Critical!!!
    random_index = random.sample(range(trials_per_epoch), val_test_num * 2)
    random_index.sort()  # sort in place (more efficient than sorted(random_index))

    # Create indices for all classes
    # For class i, trials are at positions: i*trials_per_epoch to (i+1)*trials_per_epoch - 1
    val_index = [rand + i * trials_per_epoch for i in range(5) for rand in random_index[:val_test_num]]
    test_index = [rand + i * trials_per_epoch for i in range(5) for rand in random_index[val_test_num:]]

    # Training indices are all the rest
    train_index = [item for item in range(total_trials) if item not in val_index + test_index]

    # Assign splits
    desc.iloc[val_index] = 'validate'
    desc.iloc[test_index] = 'test'
    desc.iloc[train_index] = 'train'

    # Verify correct number of trials per split
    expected_per_class = val_test_num
    total_expected = expected_per_class * 5

    assert desc[desc['split'] == 'validate'].size == total_expected, \
        f"Validation set has {desc[desc['split'] == 'validate'].size} trials, expected {total_expected}"
    assert desc[desc['split'] == 'test'].size == total_expected, \
        f"Test set has {desc[desc['split'] == 'test'].size} trials, expected {total_expected}"

    windows_datasets.set_description(desc)
    splitted = windows_datasets.split('split')

    train_set = splitted['train']
    valid_set = splitted['validate']
    test_set = splitted['test']

    return train_set, valid_set, test_set

def dataset_to_numpy(dataset):
    """Convert braindecode dataset to numpy"""
    X_list, y_list = [], []
    for i in range(len(dataset)):
        X, y, _ = dataset[i]
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list), np.array(y_list)

def load_electrode_coordinates(patient_id, base_path='../Resources/EleCTX_Files_2018_10_26'):
    """Load electrode coordinates from MATLAB file"""
    filepath = os.path.join(base_path, f'P{patient_id}', 'electrodes_Final_Norm.mat')
    
    if not os.path.exists(filepath):
        print(f"Warning: Electrode file not found at {filepath}")
        return None
    
    mat_data = hdf5storage.loadmat(filepath)
    elec_info = mat_data['elec_Info_Final_wm'][0]
    pos_cell = elec_info['pos'][0]
    
    coordinates = []
    for i in range(len(pos_cell)):
        coords = pos_cell[i].flatten()
        coordinates.append((coords[0], coords[1], coords[2]))
    
    return coordinates


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