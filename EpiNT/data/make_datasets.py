import h5py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info, TensorDataset
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from .sampler import ImbalancedDatasetSampler


class OneSampleDataset(IterableDataset):
    def __init__(self, file_path, validation_ratio=0.2, train=True):
        self.file_path = file_path
        self.validation_ratio = validation_ratio
        self.is_training = train
        
        # read vali keys
        vali_key_path = self.file_path.replace('.hdf5', '_vali_keys.txt')
        with open(vali_key_path, 'r') as f:
            vali_keys = f.read().splitlines()

        # prepare data to be loaded
        with h5py.File(file_path, 'r') as h5_file:
            self.dset_list = list(h5_file.keys())
            self.val_dset = vali_keys
            self.train_dset = [dset for dset in self.dset_list if dset not in self.val_dset]
            
            self.dset_list = self.train_dset if self.is_training else self.val_dset
            self.total_samples = sum(h5_file[dset].shape[0] for dset in self.dset_list)
    
    def __len__(self):
        return self.total_samples
    
    def __iter__(self):
        """Iterator that yields batches from the HDF5 file, supporting multi-worker processing."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, no splitting required
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        self.h5_file = h5py.File(self.file_path, 'r')
        # Split group list among workers
        dset_subset = self.dset_list[worker_id::num_workers]

        for dset_name in dset_subset:
            dataset = self.h5_file[dset_name]
            for sample in dataset:
                yield sample


def create_dataloaders(file_path, batch_size, train_ratio=0.8, num_workers=4):
    train_dataset = OneSampleDataset(file_path, validation_ratio=1-train_ratio, train=True)
    val_dataset = OneSampleDataset(file_path, validation_ratio=1-train_ratio, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader


class DownStreamDataset(Dataset):
    def __init__(self, file_path, seizure_task, validation_ratio=0.2):
        self.file_path = file_path
        self.seizure_task = seizure_task
        
        # read the validation patients from txt file
        vali_patients_path = self.file_path.replace('.hdf5', f'_{self._get_h5_group()}_vali_keys.txt')
        with open(vali_patients_path, 'r') as f:
            vali_patients = f.read().splitlines()

        with h5py.File(file_path, 'r') as h5_file:
            group = self._get_h5_group()
            self.patient_list = list(h5_file[group].keys())
            self.vali_patients = vali_patients
            self.train_patients = [patient for patient in self.patient_list if patient not in self.vali_patients]

            print(f"Total patients: {self.patient_list}")
            print(f"Train patients: {self.train_patients}")
            print(f"Vali patients: {self.vali_patients}")

            # read data into memory
            self.X_train, self.y_train, self.X_vali, self.y_vali = self._get_data(group, h5_file)
            # self.X_train, self.y_train, self.X_vali, self.y_vali = self._get_data_wo_cross(group, h5_file)
            self.class_weights = compute_class_weight(None, classes = np.unique(self.y_train), y = self.y_train)
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

    def _get_h5_group(self):
        seizure_task_dict = {
            'soz_loc': 'iEEG_bids_figshare',
            'seiz_pred': 'CHB_MIT',
            'seiz_detec': 'Zenodo_neonatal',
            'ied_detec': 'CUK_IMHANS',
            'hfo_ied_detec1': 'fnusa',
            'hfo_ied_detec2': 'mayo'
        }
        return seizure_task_dict[self.seizure_task]
    
    def _scaling(self, data):
        scaling_dict = {
            'soz_loc': 1,
            'seiz_pred': 1,
            'seiz_detec': 1,
            'ied_detec': 1,
            # I am lazy, I do not want to process data again, multiply to make it to uV
            'hfo_ied_detec1': 1e3,
            'hfo_ied_detec2': 1e3 
        }

        return data * scaling_dict[self.seizure_task]
    
    def _get_data(self, group, h5_file):
        X_train = []
        y_train = []
        X_vali = []
        y_vali = []

        for patient in self.train_patients:
            data = h5_file[group][patient]['data'][:]
            label = h5_file[group][patient]['label'][:]
            X_train.append(data)
            y_train.append(label)
        
        for patient in self.vali_patients:
            data = h5_file[group][patient]['data'][:]
            label = h5_file[group][patient]['label'][:]
            X_vali.append(data)
            y_vali.append(label)
    
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)
        X_vali = np.vstack(X_vali)
        y_vali = np.vstack(y_vali)

        # scaling data
        X_train = self._scaling(X_train)
        X_vali = self._scaling(X_vali)

        # print(X_rain.shape, y_train.shape, X_vali.shape, y_vali.shape)
        print(f"Train data shape: {X_train.shape}, {y_train.shape}")
        print(f"Vali data shape: {X_vali.shape}, {y_vali.shape}")
        # return X_train.astype(np.float32), y_train.astype(np.float32).squeeze(), X_vali.astype(np.float32), y_vali.astype(np.float32).squeeze()
        return X_train.astype(np.float32), y_train.astype(int).squeeze(), X_vali.astype(np.float32), y_vali.astype(int).squeeze()

    def _get_dataset(self, train=True):
        if train:
            return TensorDataset(torch.tensor(self.X_train), torch.tensor(self.y_train))
        else:
            return TensorDataset(torch.tensor(self.X_vali), torch.tensor(self.y_vali))
    
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_downstream_dataloaders(file_path, batch_size, seizure_task, train_ratio=0.8, num_workers=4):
    downstream_dataset = DownStreamDataset(file_path, seizure_task, validation_ratio=1-train_ratio)
    train_dataset = downstream_dataset._get_dataset(train=True)
    val_dataset = downstream_dataset._get_dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              sampler = ImbalancedDatasetSampler(train_dataset),
                              pin_memory=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            sampler=None,
                            pin_memory=True, num_workers=num_workers, drop_last=True)
    return train_loader, val_loader, downstream_dataset.class_weights