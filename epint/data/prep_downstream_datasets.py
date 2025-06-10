import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
from tqdm import tqdm
import os
import pandas as pd
import mne
import math

from torch.utils.data import Dataset, DataLoader
from .chb_mit_data_handler import Handler

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

class FNUSA_MAYO:
    # IED/HFO detection
    def __init__(self, ds_name, hd5dataset, datasets_config):
        if ds_name == 'fnusa':
            # self.eval_id = [0, 1, 2, 3, 4] # fnusa's evaluation patient id (BrainWave)
            self.path = datasets_config['FNUSA']
        elif ds_name == 'mayo':
            # self.eval_id = [0, 1, 2, 3, 4, 5] # mayo's evaluation patient id (BrainWave)
            self.path = datasets_config['Mayo']
        
        self.ds_name = ds_name

        if self.path[-1] != '/':
            self.path += '/'
        df = pd.read_csv(self.path + 'segments.csv')

        self.df = self.remove_powerline_noise_class(df.copy())

        self.hd5dataset = hd5dataset
        self.hd5dataset.create_group(ds_name)

    def remove_powerline_noise_class(self, df):
        # remove powerline noise
        df = df[df['category_id']!=0]
        df['category_id'] = df['category_id'] - 1
        df = df.reset_index(drop=True)

        # remove artifact noise 
        df = df[df['category_id']!=0]
        df['category_id'] = df['category_id'] - 1
        df = df.reset_index(drop=True)

        return df

    def _process_data(self):
        # 0: pathology, 1: physiology
        sfreq = 5000
        new_sfreq = 1024
        length = 3 # 3 seconds
        patient_list = self.df['patient_id'].unique()
        # print(patient_list)
        for i in patient_list:
            patient_df = self.df[self.df['patient_id']==i]
            arr, label = [], []
            # print(patient_df)
            for segment_id in patient_df['segment_id']:
                data = sio.loadmat(self.path+'{}'.format(segment_id))['data'].squeeze()
                target = patient_df[patient_df['segment_id']==segment_id]['category_id'].values[0] # 0: pathology, 1: [hysiology]

                # resample
                resampled_data = signal.resample(data, int(new_sfreq/sfreq*len(data)), axis=0)

                arr.append(resampled_data)
                label.append(target)
            
            arr = np.array(arr).astype(np.float32)
            label = np.array(label)
            label = label[:, np.newaxis].astype(int)
           
            print(arr.shape, label.shape)
           
            self.hd5dataset[self.ds_name].create_group(f'patient_{str(i)}')
            self.hd5dataset[self.ds_name][f'patient_{str(i)}'].create_dataset('data', data=arr)
            self.hd5dataset[self.ds_name][f'patient_{str(i)}'].create_dataset('label', data=label)


class CUK_IMHANS:
    # IED detection
    def __init__(self, hd5dataset, datasets_config):
        self.path = datasets_config['CUK_IMHANS']
        self.ds_name = 'CUK_IMHANS'

        self.hd5dataset = hd5dataset
        self.hd5dataset.create_group(self.ds_name)

    def time_to_seconds(self, hh, mm, ss):
        # Function to convert time to seconds
        return hh * 3600 + mm * 60 + ss
    
    def _process_data(self):
        # True(1) is IED, False(0) is not non-IED
        new_sfreq = 256
        duration = 12
        annot_path = os.path.join(self.path, 'Annotations', 'Epileptic')
        raw_data_path = os.path.join(self.path, 'RawData', 'Epileptic')
        for file in os.listdir(raw_data_path):
            annot_df = pd.read_csv(os.path.join(annot_path, file.split('.')[0]+'.csv'))
            raw = mne.io.read_raw(os.path.join(raw_data_path, file), preload=True, verbose=False)

            # line noise
            # line_freq = estimate_line_freq(raw)
            # print(f"Estimated line noise frequency: {line_freq} Hz")    
            # if line_freq is not None:
            #     raw = raw.copy().notch_filter(line_freq, verbose=False, notch_widths=1)

            annot_df['start_time_seconds'] = annot_df.apply(lambda row: self.time_to_seconds(row['Start hh'], row['Start mm'], row['Start ss']), axis=1)
            annot_df['end_time_seconds'] = annot_df.apply(lambda row: self.time_to_seconds(row['End hh'], row['End mm'], row['End ss']), axis=1)
            
            # print(raw.times[-1])
            epochs = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False)
            epoch_data = epochs.get_data(units = 'uV')
            epoch_start_time = np.array([i*duration for i in range(epoch_data.shape[0])])
            epoch_end_time = np.array([(i+1)*duration-1/new_sfreq for i in range(epoch_data.shape[0])])
            event_start_time = np.array(annot_df['start_time_seconds'].values)
            event_end_time = np.array(annot_df['end_time_seconds'].values)
            
            overlap = (epoch_start_time[:, None] < event_end_time) & (event_start_time < epoch_end_time[:, None]) # magic candy
            epoch_labels = np.any(overlap, axis=1).astype(int)  # True if any event overlaps with the epoch

            n_channels = epoch_data.shape[1]
            expanded_labels = np.repeat(epoch_labels[:, np.newaxis], n_channels, axis=1)

            epoch_data = epoch_data.reshape(epoch_data.shape[0] * epoch_data.shape[1], epoch_data.shape[2]).astype(np.float32)
            expanded_labels = expanded_labels.reshape(expanded_labels.shape[0] * expanded_labels.shape[1], 1).astype(int)
            
            print(epoch_data.shape, expanded_labels.shape)

            self.hd5dataset[self.ds_name].create_group(file.split('.')[0])
            self.hd5dataset[self.ds_name][file.split('.')[0]].create_dataset('data', data=epoch_data)
            self.hd5dataset[self.ds_name][file.split('.')[0]].create_dataset('label', data=expanded_labels)


class Zenodo_neonatal:
    # Seizure detection
    def __init__(self, hd5dataset, datasets_config):
        self.path = datasets_config['Zenodo_neonatal']
        self.ds_name = 'Zenodo_neonatal'

        self.hd5dataset = hd5dataset
        self.hd5dataset.create_group(self.ds_name)
    
    def _process_data(self):
        # True(1) is seizure, False(0) is not seizure
        # True(1) is not seizure, False(0) is seizure
        new_sfreq = 256
        duration = 12
        annot_A = pd.read_csv(os.path.join(self.path, 'annotations_2017_A_fixed.csv')).fillna(False).astype(bool)
        annot_B = pd.read_csv(os.path.join(self.path, 'annotations_2017_B.csv')).fillna(False).astype(bool)
        annot_C = pd.read_csv(os.path.join(self.path, 'annotations_2017_C.csv')).fillna(False).astype(bool)
        inter_annot = annot_A | annot_B | annot_C
        for i in inter_annot.columns:
            edf_path = os.path.join(self.path,  f'eeg{i}.edf')
            raw = mne.io.read_raw(edf_path, preload=True, verbose=False)     

            # line noise
            line_freq = 50
            raw = raw.copy().notch_filter(line_freq, verbose=False, notch_widths=1)

            # resample
            raw = raw.copy().resample(new_sfreq)
            epoch = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False)
            epoch_data = epoch.get_data(units = 'uV')
            
            # get the label and reduce it to epoch level
            data_duration = epoch_data.shape[0] * duration 
            label = inter_annot[i].values[:data_duration]
            label = label.reshape(-1, duration)
            label = np.any(label, axis=1).astype(int) # True if any event overlaps with the epoch
            label = 1 - label # Invert the labels

            n_channels = epoch_data.shape[1]
            expanded_labels = np.repeat(label[:, np.newaxis], n_channels, axis=1)

            # make to one channel
            epoch_data = epoch_data.reshape(epoch_data.shape[0] * epoch_data.shape[1], epoch_data.shape[2]).astype(np.float32)
            expanded_labels = expanded_labels.reshape(expanded_labels.shape[0] * expanded_labels.shape[1], 1).astype(int)

            print(epoch_data.shape, expanded_labels.shape)

            self.hd5dataset[self.ds_name].create_group(f'eeg{i}')
            self.hd5dataset[self.ds_name][f'eeg{i}'].create_dataset('data', data=epoch_data)
            self.hd5dataset[self.ds_name][f'eeg{i}'].create_dataset('label', data=expanded_labels)
            
class CHB_MIT:
    # seizure prediction
    def __init__(self, hd5dataset, datasets_config):
        self.path = datasets_config['CHB_MIT']
        self.ds_name = 'CHB_MIT'

        self.hd5dataset = hd5dataset
        self.hd5dataset.create_group(self.ds_name)

    def _process_data(self):
        new_sfreq = 256
        duration = 12
        handler = Handler(self.path)
        patients = handler.get_patients()
        for patient in patients:
            flag = True
            edf_files = handler.get_patient_edf(patient)
            seizures = handler.get_seizure_data(patient)
            seizures_with = seizures[seizures["number_of_seizures"] > 0]
            seizures_without = seizures[seizures["number_of_seizures"] == 0]
            
            from ast import literal_eval
            seizure_samples = []
            preictal_samples = []
            pre_preictal_samples = []
            for edf_file in seizures_with["file_name"]:
                seizure_time = literal_eval(seizures_with[seizures_with["file_name"] == edf_file]["start_end_times"].values[0])
                if edf_file in ['chb12_27.edf', 'chb12_28.edf', 'chb12_29.edf']:
                    continue
                
                raw = mne.io.read_raw(os.path.join(self.path, patient, edf_file), verbose=False, preload=True)
                
                # line noise
                line_freq = 60
                raw = raw.copy().notch_filter(line_freq, verbose=False, notch_widths=1)
            
                # get seizure data
                if seizure_time[0][1] - seizure_time[0][0] > duration:
                    raw_seiz = raw.copy().crop(seizure_time[0][0], seizure_time[0][1])
                    
                else:
                    end = seizure_time[0][0] + duration
                    raw_seiz = raw.copy().crop(seizure_time[0][0], end)
                
                print(os.path.join(self.path, patient, edf_file))

                # get chns and resample
                raw_seiz = raw_seiz.copy().pick(handler.get_features()).resample(new_sfreq)

                epoch_data_seiz = mne.make_fixed_length_epochs(raw_seiz, duration=duration, verbose=False).get_data(units = 'uV')
                
                seizure_samples.append(epoch_data_seiz)

                # get preictal data
                if seizure_time[0][0] - 8*60 >= 0:                    
                    raw_pre = raw.copy().crop(seizure_time[0][0]-8*60, seizure_time[0][0])
                    raw_pre = raw_pre.copy().pick(handler.get_features()).resample(new_sfreq)
                    epoch_data_preictal = mne.make_fixed_length_epochs(raw_pre, duration=duration, verbose=False).get_data(units = 'uV')
                   
                    preictal_samples.append(epoch_data_preictal)
        
            epoch_data_seiz = np.concatenate(seizure_samples, axis=0)
            epoch_data_preictal = np.concatenate(preictal_samples, axis=0)
 
            # make labels
            n_channels = epoch_data_seiz.shape[1]
            expanded_labels_seiz = np.full((epoch_data_seiz.shape[0], n_channels), 0) # seizure is 0
            expanded_labels_preictal = np.full((epoch_data_preictal.shape[0], n_channels), 1) # preictal is 1
            
            # make to one channel
            epoch_data_seiz = epoch_data_seiz.reshape(epoch_data_seiz.shape[0] * epoch_data_seiz.shape[1], epoch_data_seiz.shape[2]).astype(np.float32)
            epoch_data_preictal = epoch_data_preictal.reshape(epoch_data_preictal.shape[0] * epoch_data_preictal.shape[1], epoch_data_preictal.shape[2]).astype(np.float32)
            array = np.concatenate([epoch_data_seiz, epoch_data_preictal], axis=0)


            # also labels
            expanded_labels_seiz = expanded_labels_seiz.reshape(expanded_labels_seiz.shape[0] * expanded_labels_seiz.shape[1], 1).astype(int)
            expanded_labels_preictal = expanded_labels_preictal.reshape(expanded_labels_preictal.shape[0] * expanded_labels_preictal.shape[1], 1).astype(int)
            labels = np.concatenate([expanded_labels_seiz, expanded_labels_preictal], axis=0)

            print(array.shape, labels.shape)

            self.hd5dataset[self.ds_name].create_group(patient)
            self.hd5dataset[self.ds_name][patient].create_dataset('data', data=array)
            self.hd5dataset[self.ds_name][patient].create_dataset('label', data=labels)
            
        

class SOZ_LOC:
    # soz localization
    def __init__(self, hd5dataset, datasets_config):
        self.path = datasets_config['iEEG_bids_figshare']
        self.ds_name = 'iEEG_bids_figshare'

        self.hd5dataset = hd5dataset
        self.hd5dataset.create_group(self.ds_name)

    def _process_data(self):
        new_sfreq = 1024
        duration = 3
        root_path = os.path.join(self.path, 'ieeg_ieds_bids')
        patient_list = [subject for subject in os.listdir(root_path) if subject.startswith('sub-')]
        for patient in patient_list:
            patient_dir = os.path.join(root_path, patient, 'ieeg')
            chn_tsv = [file for file in os.listdir(patient_dir) if file.endswith('channels.tsv')][0]
            chn_df = pd.read_csv(os.path.join(patient_dir, chn_tsv), sep='\t')
            chn_orders = chn_df['name'].values
            chn_label = chn_df['soz_region'].values
            print(chn_orders, chn_label)

            # get the data
            edf_file = [file for file in os.listdir(patient_dir) if file.endswith('.edf')][0]
            raw = mne.io.read_raw(os.path.join(patient_dir, edf_file), preload=True, verbose=False)

            # line noise
            line_freq = estimate_line_freq(raw)
            print(f"Estimated line noise frequency: {line_freq} Hz")    
            if line_freq is not None:
                raw = raw.copy().notch_filter(line_freq, verbose=False, notch_widths=1)
            
            raw = raw.copy().reorder_channels(chn_orders)
            raw = raw.copy().resample(new_sfreq)

            epochs = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False)
            epoch_data = epochs.get_data(units = 'uV')

            # expand the label
            expanded_labels = np.repeat(chn_label[np.newaxis, :], epoch_data.shape[0], axis=0)
            
            # make to one channel
            epoch_data = epoch_data.reshape(epoch_data.shape[0] * epoch_data.shape[1], epoch_data.shape[2]).astype(np.float32)
            expanded_labels = expanded_labels.reshape(expanded_labels.shape[0] * expanded_labels.shape[1], 1).astype(int)

            print(expanded_labels.shape, epoch_data.shape)

            self.hd5dataset[self.ds_name].create_group(patient)
            self.hd5dataset[self.ds_name][patient].create_dataset('data', data=epoch_data)
            self.hd5dataset[self.ds_name][patient].create_dataset('label', data=expanded_labels)



            
            