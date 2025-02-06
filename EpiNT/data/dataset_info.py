import os
import json
import mne
import math
import pandas as pd
import numpy as np

class openneuron:
    def __init__(self, ds_name) -> None:
        root_dir = '/your/path/openneuron'
        self.ds_name = ds_name
        self.root_dir = os.path.join(root_dir, ds_name)


    def get_patients(self):
        files = os.listdir(self.root_dir)
        patient_list = []
        for file in files:
            if file.startswith('sub'):
                patient_list.append(file)
        return patient_list

    def get_durations(self, patient):
        patient_dir = os.path.join(self.root_dir, patient)
        patient_durations = []
        if self.ds_name in ['ds003029', 'ds003498', 'ds003844', 'ds003876', 'ds004100', 'ds004752', 'ds005398']:
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.json'):
                        with open(os.path.join(root, file)) as f:
                            data = json.load(f)
                            if 'RecordingDuration' in data.keys():
                                patient_durations.append(data['RecordingDuration'])
        elif self.ds_name in ['ds003555']:
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.edf'):
                        raw = mne.io.read_raw_edf(os.path.join(root, file))
                        duration = raw.times[-1]
                        patient_durations.append(duration)
        return patient_durations
    
    def get_types(self, patient):
        patient_dir = os.path.join(self.root_dir, patient)
        patient_types = []
        if self.ds_name in ['ds003029', 'ds003498', 'ds003844', 'ds003876', 'ds004100', 'ds004752', 'ds005398']:
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.json'):
                        with open(os.path.join(root, file)) as f:
                            data = json.load(f)
                            if "ECOGChannelCount" in data.keys() and data['ECOGChannelCount'] > 0:
                                patient_types.append('ECOG')
                            if "SEEGChannelCount" in data.keys() and data['SEEGChannelCount'] > 0:
                                patient_types.append('SEEG')
                            if "EEGChannelCount" in data.keys() and data['EEGChannelCount'] > 0:
                                patient_types.append('EEG')
        elif self.ds_name in ['ds003555']:
            patient_types.append('EEG')
        
        return patient_types


    def get_sfreq(self, patient):
        patient_dir = os.path.join(self.root_dir, patient)
        patient_sfreqs = []
        if self.ds_name in ['ds003029', 'ds003498', 'ds003844', 'ds003876', 'ds004100', 'ds004752', 'ds005398']:
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.json'):
                        with open(os.path.join(root, file)) as f:
                            data = json.load(f)
                            if "SamplingFrequency" in data.keys():
                                patient_sfreqs.append(data['SamplingFrequency'])
        
        elif self.ds_name in ['ds003555']:
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.edf'):
                        raw = mne.io.read_raw_edf(os.path.join(root, file))
                        sfreq = raw.info['sfreq']
                        patient_sfreqs.append(sfreq)
        
        return patient_sfreqs

    def loop(self):
        patient_list = self.get_patients()
        ds_level_stat = {
            'Number of Patients': len(patient_list),
            'Durations': [],
            'Types': [],
            'Sampling Frequency': [],
        }
        for patient in patient_list:
            # get durations and transform to hours
            patient_durations = self.get_durations(patient)
            duration_sum = np.array(patient_durations).sum() / 3600
            ds_level_stat['Durations'].append(duration_sum)
            
            # get types
            patient_types = self.get_types(patient)
            ds_level_stat['Types'].extend(patient_types)
            
            # get sampling frequency
            patient_sfreqs = self.get_sfreq(patient)
            for idx, sfreq in enumerate(patient_sfreqs):
                # post processing
                if math.ceil(sfreq) in [200, 250, 256, 500, 512, 1000, 1024, 2000, 2048]:
                    patient_sfreqs[idx] = math.ceil(sfreq)
                elif math.floor(sfreq) in [200, 250, 256, 500, 512, 1000, 1024, 2000, 2048]:
                    patient_sfreqs[idx] = math.floor(sfreq)
            ds_level_stat['Sampling Frequency'].extend(patient_sfreqs)

        
        ds_level_stat['Durations'] = np.array(ds_level_stat['Durations']).sum()
        ds_level_stat['Sampling Frequency'] = list(set(ds_level_stat['Sampling Frequency'])) # need change
        ds_level_stat['Types'] = list(set(ds_level_stat['Types'])) # need change
            
        return ds_level_stat

class TUEP:
    def __init__(self) -> None:
        self.root_dir = '/your/path/TUEP/tuh_eeg_epilepsy/v2.0.1'
    
    def get_patients(self):
        epilepsy_dir = os.path.join(self.root_dir, '00_epilepsy')
        patients = os.listdir(epilepsy_dir)
        return patients
    
    def loop(self):
        patient_list = self.get_patients()
        ds_level_stat = {
            'Number of Patients': len(patient_list),
            'Durations': [],
            'Types': ['EEG'],
            'Sampling Frequency': [],
        }

        for patient in patient_list:
            patient_dir = os.path.join(self.root_dir, '00_epilepsy', patient)
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.edf'):
                        raw = mne.io.read_raw_edf(os.path.join(root, file))
                        duration = raw.times[-1]
                        ds_level_stat['Durations'].append(duration / 3600)
                        ds_level_stat['Sampling Frequency'].append(int(raw.info['sfreq']))
        
        # post processing
        ds_level_stat['Durations'] = np.array(ds_level_stat['Durations']).sum()
        ds_level_stat['Sampling Frequency'] = list(set(ds_level_stat['Sampling Frequency']))
        return ds_level_stat

class TUSZ:
    def __init__(self) -> None:
        self.root_dir = '/your/path/TUSZ/'
    
    def get_patients(self):
        train_patients = os.listdir(os.path.join(self.root_dir, 'edf', 'train'))
        eval_patients = os.listdir(os.path.join(self.root_dir, 'edf', 'eval'))
        dev_patients = os.listdir(os.path.join(self.root_dir, 'edf', 'dev'))

        patients = train_patients + eval_patients + dev_patients
        return patients
    
    def loop(self):
        patient_list = self.get_patients()
        ds_level_stat = {
            'Number of Patients': len(patient_list),
            'Durations': [],
            'Types': ['EEG'],
            'Sampling Frequency': [],
        }

        for patient in patient_list:
            for dataset in ['train', 'eval', 'dev']:
                patient_dir = os.path.join(self.root_dir, 'edf', dataset, patient)
                for root, dirs, files in os.walk(patient_dir):
                    for file in files:
                        if file.endswith('.edf'):
                            raw = mne.io.read_raw_edf(os.path.join(root, file))
                            duration = raw.times[-1]
                            ds_level_stat['Durations'].append(duration / 3600)
                            ds_level_stat['Sampling Frequency'].append(int(raw.info['sfreq']))
        
        # post processing
        ds_level_stat['Durations'] = np.array(ds_level_stat['Durations']).sum()
        ds_level_stat['Sampling Frequency'] = list(set(ds_level_stat['Sampling Frequency']))
        return ds_level_stat
    
class Siena:
    def __init__(self) -> None:
        self.root_dir = '/your/path/Siena/physionet.org/files/siena-scalp-eeg/1.0.0'
    
    def loop(self):
        df = pd.read_csv(os.path.join(self.root_dir, 'subject_info.csv'))
        # print(df.columns)
        # print(df[' rec_time_minutes'])
        # print(df['rec_time_minutes'])
        ds_level_stat = {
            'Number of Patients': len(df['patient_id']),
            'Durations': df[' rec_time_minutes'].sum() / 60,
            'Types': ['EEG'],
            'Sampling Frequency': [512],
        }

        return ds_level_stat

class JHCH_JHMCHH:
    def __init__(self) -> None:
        self.root_dir = '/your/path/JHCH_JHMCHH/'
    
    def loop(self):
        ds_level_stat = {
            'Number of Patients': 11+6,
            'Durations': [],
            'Types': ['EEG'],
            'Sampling Frequency': [],
        }

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.edf'):
                    raw = mne.io.read_raw_edf(os.path.join(root, file), encoding='latin1')
                    duration = raw.times[-1]
                    ds_level_stat['Durations'].append(duration / 3600)
                    ds_level_stat['Sampling Frequency'].append(int(raw.info['sfreq']))
        
        # post processing
        ds_level_stat['Durations'] = np.array(ds_level_stat['Durations']).sum()
        ds_level_stat['Sampling Frequency'] = list(set(ds_level_stat['Sampling Frequency']))
        
        return ds_level_stat

class Mendeley:
    def __init__(self) -> None:
        self.root_dir = '/your/path/Mendeley/'

    def loop(self):
        ds_level_stat = {
            'Number of Patients': 6,
            'Durations': [],
            'Types': ['EEG'],
            'Sampling Frequency': [500],
        }

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.edf'):
                    raw = mne.io.read_raw_edf(os.path.join(root, file))
                    duration = raw.times[-1]
                    ds_level_stat['Durations'].append(duration / 3600)
                    ds_level_stat['Sampling Frequency'].append(int(raw.info['sfreq']))
        
        # post processing
        ds_level_stat['Durations'] = np.array(ds_level_stat['Durations']).sum()
        ds_level_stat['Sampling Frequency'] = list(set(ds_level_stat['Sampling Frequency']))

        return ds_level_stat