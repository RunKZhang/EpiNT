import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
from yaml import CLoader as Loader
from yaml import dump, load
from epint.data.utils import h5Dataset
from epint.data.prep_downstream_datasets import *

def print_hdf5_structure(f, indent=0):
    """Recursively print the structure of an HDF5 file with clear hierarchy."""
    for key in f.keys():
        item = f[key]
        prefix = "  " * indent  # Create an indent based on the level
        if isinstance(item, h5py.Group):
            print(f"{prefix}- {key} (Group)")
            print_hdf5_structure(item, indent + 1)  # Recursive call for sub-groups
        else:
            print(f"{prefix}- {key} (Dataset: {item.shape})")

if __name__ == "__main__":
    dataset_config_path = '../configs/dataset_configs.yaml'
    with open(dataset_config_path, 'rb') as f:
        dataset_config = load(f, Loader=Loader)

    # pretrain Dataset
    hdf5_name = 'downstream_dataset'
    file_path = os.path.join(dataset_config['DATASETS_OUTPUT_PATH'], f'{hdf5_name}.hdf5')
    downstream_dataset = h5py.File(file_path, 'a')
    # FNUSA_MAYO('fnusa', downstream_dataset, dataset_config)._process_data()
    # FNUSA_MAYO('mayo', downstream_dataset, dataset_config)._process_data()
    # CUK_IMHANS(downstream_dataset, dataset_config)._process_data()
    # Zenodo_neonatal(downstream_dataset, dataset_config)._process_data()
    CHB_MIT(downstream_dataset, dataset_config)._process_data()
    # SOZ_LOC(downstream_dataset, dataset_config)._process_data()
    downstream_dataset.close()

    
    