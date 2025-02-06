import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
from EpiNT.data.utils import h5Dataset
from EpiNT.data.dataset_list import DATASETS_OUTPUT_PATH
from EpiNT.data.prep_downstream_datasets import *

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
    # pretrain Dataset
    hdf5_name = 'downstream_wo_sfreq_equal_cpres_chn'
    file_path = os.path.join(DATASETS_OUTPUT_PATH, f'{hdf5_name}.hdf5')
    downstream_dataset = h5py.File(file_path, 'a')
    FNUSA_MAYO('fnusa', downstream_dataset)._process_data()
    FNUSA_MAYO('mayo', downstream_dataset)._process_data()
    CUK_IMHANS(downstream_dataset)._process_data()
    Zenodo_neonatal(downstream_dataset)._process_data()
    CHB_MIT(downstream_dataset)._process_data()
    SOZ_LOC(downstream_dataset)._process_data()
    downstream_dataset.close()
    # print(DATASETS_PATH['TUSZ'])
    
    # Print the structure of the HDF5 file
    with h5py.File(file_path, 'r') as f:
        print_hdf5_structure(f)

    # Get .hd5 file size in terabytes (1 TB = 1024^4 bytes)
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 ** 3)
    print(f"File size: {file_size} bytes ({file_size_mb:.2f} GB)")

    
    