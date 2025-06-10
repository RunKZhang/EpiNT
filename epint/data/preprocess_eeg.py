import mne
import math
import numpy as np
from scipy.interpolate import CubicSpline

from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def estimate_line_freq(instance):
    psd = instance.compute_psd(fmin=40, fmax=70, method='welch', picks='all')
    psd_values = psd.get_data()
    freqs = psd.freqs
    if isinstance(instance, mne.io.BaseRaw):
        line_freq = freqs[np.argmax(psd_values.mean(axis=0))]
    elif isinstance(instance, mne.Epochs):
        line_freq = freqs[np.argmax(psd_values.mean(axis=(0, 1)))]
    else:
        raise ValueError('instance should be either mne.io.Raw or mne.Epochs')

    # post-process the line_freq
    if int(math.ceil(line_freq)) == 50 or int(math.floor(line_freq)) == 50:
        line_freq = 50
    elif int(math.ceil(line_freq)) == 60 or int(math.floor(line_freq)) == 60:
        line_freq = 60
    else:
        print(f"Warning: the estimated line noise frequency is {line_freq} Hz, not 50 or 60 Hz")
        line_freq = None

    return line_freq

def preprocess_func(raw, equal_sfreq = True, compress_chn = False, normalize = False, scalp_ieeg_flag = False):
    # scalp_ieeg_flag: True for scalp EEG, False for iEEG
    print(f'raw data length: {raw.times[-1]}')
    units = 'uV'
    print(f'using units: {units}')

    # raw_data = raw.get_data()
    # print(f'sample mean: {np.abs(raw_data.mean())}, sample std: {raw_data.std()}, sample range: {raw_data.min()} to {raw_data.max()}')
    if equal_sfreq:
        # equalize the eeg and ieeg sfreq, 
        global_sfreq = 1024
        duration = 3
    else:
        # scalp to 256 Hz, iEEG to 1024 Hz
        if scalp_ieeg_flag:
            global_sfreq = 256
            duration = 12
        else:
            global_sfreq = 1024
            duration = 3

    if raw.times[-1] < duration:
        print('-------The data is shorter than the duration, skip-------')
        return None, global_sfreq, len(raw.ch_names)
    else:
        print(f'-------The data is longer than the duration, process-------')
        # remove line noise and harmonics from raw
        line_freq = estimate_line_freq(raw)
        print(f"Estimated line noise frequency: {line_freq} Hz")

        # Up to the 9th harmonic# Up to the 9th harmonic for iEEG, 5th for scalp EEG
        # harmonics = [line_freq * i for i in range(1, 10)] if raw.info['sfreq'] >= 1024 else [line_freq * i for i in range(1, 5)]  
        if line_freq is not None:
            raw = raw.copy().notch_filter(line_freq, verbose=False, notch_widths=1)
        
        # segment into epochs
        epoch = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False).load_data()
        # print(f'Original epoch_shape: {epoch.get_data().shape}')
    
    # resampling and get data
    if epoch.info['sfreq'] > global_sfreq or epoch.info['sfreq'] < global_sfreq:
        epoch = epoch.resample(sfreq=global_sfreq, verbose=False)
        # epoch_np = epoch.get_data(units=units)
        # pass
    else:
        print(f'sampling frequency is {global_sfreq} Hz, no need for equalize')
    # epoch = mne.make_fixed_length_epochs(raw, duration=12, verbose=False).load_data()
    epoch_np = epoch.get_data(units=units)
    global_sfreq = epoch.info['sfreq']

    # make data to one channel
    if compress_chn:
        print('-------Compress the channel number -------')
        chn_num = epoch_np.shape[1]
        epoch_num = epoch_np.shape[0]
        epoch_np = epoch_np.reshape(epoch_num * chn_num, epoch_np.shape[2]) 
    
    # normalize the data
    if normalize:
        print('-------Normalize the data-------')   
        scaler = StandardScaler()
        epoch_np = scaler.fit_transform(epoch_np.copy())

    # pad and crop
    if epoch_np.shape[1] >= int(global_sfreq * duration):
        print('-------The data is longer than or equal to the final duration, crop-------')
        orgi_shape = epoch_np.shape
        print(f'Before modification: {epoch_np.shape}')
        epoch_np = epoch_np[:, :int(global_sfreq * duration)]
        print(f'epoch_np: {epoch_np.shape}, sfreq: {global_sfreq}, final duration: {int(global_sfreq*duration)}, loss: {orgi_shape[1] - int(global_sfreq*duration)}')
        
    elif epoch_np.shape[1] < int(global_sfreq * duration):
        print('-------The data is shorter than the final duration, pad-------')
        pad_width = int(global_sfreq * duration) - epoch_np.shape[1]
        print(f'Before modification: {epoch_np.shape}')
        epoch_np = np.pad(epoch_np, ((0, 0), (0, pad_width)), 'mean')
        print(f'epoch_np shape: {epoch_np.shape}, sfreq: {global_sfreq}, final_duration: {int(global_sfreq*duration)}, pad width: {pad_width}')
    
    print(f'final epoch shape: {epoch_np.shape}')  
    print(f'epoch mean: {np.abs(epoch_np.mean())}, epoch std: {epoch_np.std()}, epoch range: {epoch_np.min()} to {epoch_np.max()}')

    return epoch_np, global_sfreq, chn_num
    