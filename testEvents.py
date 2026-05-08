import pywt
import numpy as np
import mne
import hdf5storage
from sklearn.preprocessing import StandardScaler
from braindecode.datasets import create_from_mne_epochs
from chn_settings import get_channel_setting

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

    # Get trigger channel indices
    stim_names = ["stim0", "stim1", "stim2", "stim3", "stim4"]
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

preprocess(10)