import os
import pandas as pd
import mne
import numpy as np
import json

from scipy.io import loadmat
from EpiNT.data.utils import tuh_montage
from .dataset_list import DATASETS_PATH, DATASETS_OUTPUT_PATH
from .preprocess_eeg import preprocess_func, concat_session

class PretrainDatasetBase:
    def __init__(self, dataset_name, hd5dataset):
        self.summary_df = pd.DataFrame(columns=['Patient_id', 'Session', 'Seconds', 'Type', 'Sampling Frequency', 'Channel Number'])
        self.dataset_name = dataset_name
        self.path = DATASETS_PATH[dataset_name]
        self.hd5dataset = hd5dataset
    
    def _process_data(self, duration=1, equal_sfreq=True):
        raise NotImplementedError

    def _save_to_hd5(self, patient_id, session,seconds, Type, global_sfreq, epoch_np, channel_num, scalp_ieeg_flag):
        if epoch_np is not None:
            # channel_num = epoch_np.shape[1]

            # for epoch, do Hd5Dataset, put each (epoch_num, chn_num, sequence) into a dataset
            # grp  = self.hd5dataset.addGroup(f'{patient_id}_{session}')       
            # dset = self.hd5dataset.addDataset(grp, f'{patient_id}_{session}', epoch_np.astype(np.float32))
            # grp.attrs['sfreq'] =  global_sfreq
            # grp.attrs['chn_num'] = channel_num
            self.hd5dataset.createDataset(f'{patient_id}_{session}', epoch_np.astype(np.float32))
            eeg_type = 'EEG' if scalp_ieeg_flag else 'iEEG'
            self.hd5dataset.addAttributes(self.hd5dataset[f'{patient_id}_{session}'], 'type', eeg_type)
            # self.hd5dataset.__f[f'{patient_id}_{session}'].attrs['type'] = eeg_type
            # save the summary
            self.summary_df.loc[len(self.summary_df)] = [patient_id, session, seconds, Type, global_sfreq, channel_num]

    
class OpenNeuro(PretrainDatasetBase):
    # DS00xxxx use this class
    def __init__(self, dataset_name, hd5dataset):
        super().__init__(dataset_name, hd5dataset)
        self.files_list = []
        for dirpath, _, filenames in os.walk(self.path):
            for filename in filenames:
                if filename.endswith('eeg.edf') or filename.endswith('eeg.vhdr'):
                    self.files_list.append(os.path.join(dirpath, filename))      
    
    def _convert_units(self, raw, unit):
        # convert to proper unit, some data's unit is not correct
        if self.dataset_name == 'DS003555':
            # its value is in uV, converts to V
            raw_data = raw.get_data()
            if unit[0] == 'uV':
                raw_data = raw_data / 1e6
                info = raw.info
                raw = mne.io.RawArray(raw_data, info)
        
        elif self.dataset_name == 'DS004752':
            raw_data = raw.get_data()
            raw_data = raw_data / 1e6
            info = raw.info
            raw = mne.io.RawArray(raw_data, info)

        return raw
    
    def _open_neuron_chn_mapping(self, raw, file_path):
        # Chn mapping, for selecting the good channels
        if file_path.split('/')[-1] in ['sub-umf005_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        'sub-umf005_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf']:
            # print('I am here')
            chn_mapping = {chn: chn.replace(' ', '') for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-NIH1_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH8_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH7_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH11_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH6_ses-extraoperative_task-interictal_run-01_ieeg.edf']:
            chn_mapping = {chn: chn.replace('-G2', '') for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn.replace('EEG ', '') for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-NIH2_ses-extraoperative_task-interictal_run-01_ieeg.edf',]:
            chn_mapping = {chn: chn.replace('EEG POL ', '') for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn.replace('-'+chn.split('-')[1], '') for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn for chn in raw.ch_names}         
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-PY18N002_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        ]:
            chn_mapping = {chn: chn.split('-Ref')[0].split(' ')[1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn for chn in raw.ch_names}
            chn_mapping['Fp1'] = 'FP1'
            chn_mapping['Fp2'] = 'FP2'
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-PY19N023_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        ]:
            raw = raw.drop_channels(['POL-0', 'POL-1'])
            chn_mapping = {chn: chn.split('-Ref')[0].split(' ')[1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn for chn in raw.ch_names}
            chn_mapping['Fp1'] = 'FP1'
            chn_mapping['Fp2'] = 'FP2'
            raw = raw.rename_channels(chn_mapping)
            # print(raw.ch_names)
        if file_path.split('/')[-1] in ['sub-NIH3_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-pt1_ses-extraoperative_task-interictalawake_run-02_ieeg.edf',
                                        'sub-pt1_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf',
                                        'sub-pt1_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        'sub-pt1_ses-extraoperative_task-interictalasleep_run-02_ieeg.edf',
                                        'sub-jh103_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        'sub-jh103_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf',
                                        'sub-NIH10_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-PY19N012_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH9_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-PY19N026_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH5_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-PY18N013_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-pt3_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf',
                                        'sub-pt3_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        'sub-pt3_ses-extraoperative_task-interictalasleep_run-02_ieeg.edf',
                                        'sub-pt2_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf',
                                        'sub-pt2_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        'sub-pt2_ses-extraoperative_task-interictalasleep_run-02_ieeg.edf',
                                        'sub-pt2_ses-extraoperative_task-interictalawake_run-02_ieeg.edf']:
            chn_mapping = {chn: chn.split(' ')[-1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-PY19N015_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-jh105_ses-extraoperative_task-interictalasleep_run-01_ieeg.edf',
                                        'sub-jh105_ses-extraoperative_task-interictalawake_run-01_ieeg.edf',
                                        ]:
            raw = raw.drop_channels(['POL-0', 'POL-1'])
            chn_mapping = {chn: chn.split(' ')[-1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-PY18N007_ses-extraoperative_task-interictal_run-01_ieeg.edf',
                                        'sub-NIH4_ses-extraoperative_task-interictal_run-01_ieeg.edf']:
            chn_mapping = {chn: chn.split(' ')[-1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
            chn_mapping = {chn: chn.split('-R')[0] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)
        if file_path.split('/')[-1] in ['sub-PY18N015_ses-extraoperative_task-interictal_run-01_ieeg.edf']:
            raw = raw.drop_channels(['POL-0', 'POL-1'])
            chn_mapping = {chn: chn.split('-Ref')[0].split(' ')[1] for chn in raw.ch_names}
            raw = raw.rename_channels(chn_mapping)

        return raw

    def _process_data(self, equal_sfreq, compress_chn):
        if self.dataset_name in ['DS003555']:
            # scalp
            scalp_ieeg_flag = True
        elif self.dataset_name in ['DS003029', 'DS003498', 'DS003844', 'DS003876', 'DS004100', 'DS004752', 'DS005398']:
            # iEEG
            scalp_ieeg_flag = False
        else:
            raise ValueError('Dataset name is not supported')

        for file_path in self.files_list:
            print(file_path)
            subject_name = file_path.split('/')[-4]
            patient_id = self.dataset_name + '_' + file_path.split('/')[-4]
            print(patient_id)
            session = file_path.split('/')[-1].split(subject_name)[1].split('.')[0][1:]
            Type = file_path.split('/')[-2]

            if file_path.endswith('.vhdr'):
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            elif file_path.endswith('.edf'):
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            seconds = raw.times[-1]
            
            # Chn mapping, for selecting the good channels
            raw = self._open_neuron_chn_mapping(raw, file_path)
            if 'ieeg.' in file_path:
                chn_tsv = pd.read_csv(file_path.split('ieeg.')[0] +'channels.tsv', sep='\t')
            elif 'eeg.' in file_path:
                chn_tsv = pd.read_csv(file_path.split('eeg.')[0] +'channels.tsv', sep='\t')
            if 'status' in chn_tsv.columns:
                good_chns = chn_tsv[chn_tsv['status'] == 'good']['name'].tolist()
                raw = raw.pick(good_chns)
            if 'units' in chn_tsv.columns:
                # conver units
                unit = chn_tsv['units'].unique()
                raw = self._convert_units(raw, unit)    
                
            # drop the dead channels
            useless_chns = []
            for chn in raw.ch_names:
                if chn.startswith('$'):
                    useless_chns.append(chn)
            raw = raw.drop_channels(useless_chns)
            
            # raw_data = raw.get_data()
            # print(f'sample mean: {np.abs(raw_data.mean())}, sample std: {raw_data.std()}, sample range: {raw_data.min()} to {raw_data.max()}')
            # preprocess the raw data
            epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                              equal_sfreq=equal_sfreq, 
                                                              compress_chn=compress_chn, 
                                                              scalp_ieeg_flag=scalp_ieeg_flag)
            
            self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag)

        # save the summary
        self.hd5dataset.store_dataframe(self.summary_df)

        return self.summary_df

class Normal_EDF_EEG(PretrainDatasetBase):
    # TUEP, Siena, Zenodo_neonatal, iEEG_bids_figshare, CHB_MIT
    def __init__(self, dataset_name, hd5dataset):
        super().__init__(dataset_name, hd5dataset)
        self.edf_files = []
        for dirpath, _, filenames in os.walk(self.path):
            for filename in filenames:
                if filename.endswith('.edf'):
                    self.edf_files.append(os.path.join(dirpath, filename))

    
    def _process_data(self, equal_sfreq, compress_chn):
        if self.dataset_name == 'TUEP':
            scalp_ieeg_flag = True # EEG
            for file_path in self.edf_files:
                if '00_epilepsy' in file_path:
                    # only process epilepsy data
                    print(file_path)
                    patient_id = self.dataset_name + '_' + file_path.split('/')[-4]
                    session = file_path.split('/')[-3]+'_'+file_path.split('/')[-1].split('_')[-1].split('.')[0]
                    montage = file_path.split('/')[-2]
                    print(patient_id, session, montage)
                    
                    # read the raw data
                    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                    seconds = raw.times[-1]
                    Type = 'EEG'
                    anode, cathode, ch_name = tuh_montage(montage)
                    raw = mne.set_bipolar_reference(raw, anode, cathode, ch_name, drop_refs=True, verbose=False)
                    raw =raw.pick(ch_name)  
                    
                    # preprocess the raw data
                    epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                                      equal_sfreq=equal_sfreq,  
                                                                      compress_chn=compress_chn,
                                                                      scalp_ieeg_flag=scalp_ieeg_flag)
                    
                    self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag)  
        
        elif self.dataset_name == 'TUSZ':
            scalp_ieeg_flag = True # EEG
            for file_path in self.edf_files:
                print(file_path)
                patient_id = self.dataset_name + '_' + file_path.split('/')[-4]
                session = file_path.split('/')[-3]+'_'+file_path.split('/')[-1].split('_')[-1].split('.')[0]
                montage = file_path.split('/')[-2]
                print(patient_id, session, montage)

                # read the raw data
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                seconds = raw.times[-1]
                Type = 'EEG'
                anode, cathode, ch_name = tuh_montage(montage)
                raw = mne.set_bipolar_reference(raw, anode, cathode, ch_name, drop_refs=True, verbose=False)
                raw =raw.pick(ch_name)  
                
                # preprocess the raw data
                epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                                  equal_sfreq=equal_sfreq, 
                                                                  compress_chn=compress_chn,
                                                                  scalp_ieeg_flag=scalp_ieeg_flag)
                
                self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag) 

        elif self.dataset_name == 'Siena':
            scalp_ieeg_flag = True # EEG
            for file_path in self.edf_files:
                print(file_path)
                patient_id = self.dataset_name + '_' +file_path.split('/')[-2]
                session = file_path.split('/')[-1].split('.')[0].split('-')[1]
                
                raw = mne.io.read_raw(file_path, preload=True, verbose=False)
                pick_chns = [chn_name for chn_name in raw.ch_names if chn_name.startswith('EEG')]
                raw = raw.copy().pick(pick_chns)
                seconds = raw.times[-1]
                Type = 'EEG'
                
                # preprocess the raw data
                epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                                  equal_sfreq=equal_sfreq, 
                                                                  compress_chn=compress_chn,
                                                                  scalp_ieeg_flag=scalp_ieeg_flag)
                
                self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag)
        

        elif self.dataset_name == 'JHCH_JHMCHH':
            scalp_ieeg_flag = True # EEG
            for file_path in self.edf_files:
                patient_id = self.dataset_name + '_' + file_path.split('/')[-4] + '_' + file_path.split('/')[-2]
                session = file_path.split('/')[-1].split('.')[0]
                Type = 'EEG'
                
                # raw file reading
                raw = mne.io.read_raw(file_path, preload=True, verbose=False, encoding='latin1')
                chns = [chn for chn in raw.ch_names if chn.startswith('EEG')]
                raw = raw.pick(chns)
                seconds = raw.times[-1]
                                
                # preprocess the raw data
                epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                                  equal_sfreq=equal_sfreq, 
                                                                  compress_chn=compress_chn, 
                                                                  scalp_ieeg_flag=scalp_ieeg_flag)
                
                self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag)

        elif self.dataset_name == 'Mendeley':
            scalp_ieeg_flag = True # EEG
            for file_path in self.edf_files:
                patient_id = self.dataset_name + '_' + file_path.split('/')[-1].split('_')[0]
                session = file_path.split('/')[-1].split('.')[0].split('_')[1]
                Type = 'EEG'

                raw = mne.io.read_raw(file_path, preload=True, verbose=False)
                seconds = raw.times[-1]
                chns = [chn for chn in raw.ch_names if chn.startswith('EEG')]
                raw = raw.pick(chns)
                print(raw.ch_names)

                # preprocess the raw data
                epoch_np, global_sfreq, chn_num = preprocess_func(raw, 
                                                                  equal_sfreq=equal_sfreq, 
                                                                  compress_chn=compress_chn, 
                                                                  scalp_ieeg_flag=scalp_ieeg_flag)
                
                self._save_to_hd5(patient_id, session, seconds, Type, global_sfreq, epoch_np, chn_num, scalp_ieeg_flag)

        # save the summary
        self.hd5dataset.store_dataframe(self.summary_df)

        return self.summary_df
        
