import os
import sys
import h5py
import shutil
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EpiNT.data.prep_pretrain_datasets import *
from EpiNT.data.utils import h5Dataset
from EpiNT.data.dataset_list import DATASETS_OUTPUT_PATH

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


def remove_patient_groups(original_file, prefix):
    """
    Deletes groups with a specified prefix from an HDF5 file, repacking the file to free disk space.

    Parameters:
    - original_file (str): Path to the original HDF5 file.
    - prefix (str): Prefix of group names to delete.
    - temp_file (str): Temporary file name for the repacked HDF5 file.
    """
    temp_file = original_file.split('.')[0]+'_temp.hdf5'

    # Create a new HDF5 file that only includes groups without the specified prefix
    try:
        with h5py.File(original_file, 'r') as src, h5py.File(temp_file, 'w') as dst:
            for group_name in src:
                if not group_name.startswith(prefix):
                    src.copy(group_name, dst)  # Copy only the groups that do not match the prefix
            
        # Replace the original file with the repacked file
        os.replace(temp_file, original_file)
        print(f"Successfully deleted groups with prefix '{prefix}' and freed up space.")

    except Exception as e:
        # Cleanup in case of failure
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Failed to delete and repack HDF5 file: {e}")

def reorganize_hdf5(path, batch_size=2048, chunk_batch=256, chunk_length=6144):
    """
    Reorganize the HDF5 file to speed up data loading by reordering the datasets.

    Parameters:
    - path (str): Path to the HDF5 file.
    """
    temp_file = path.split('.')[0]+'_reorg.hdf5'

    with h5py.File(path, 'r') as src, h5py.File(temp_file, 'w') as dst:
        groups = list(src.keys())
        random.shuffle(groups)

        current_batch = []
        idx = 0
        for group_name in groups:
            print(group_name)
            dataset = src[group_name][group_name]
            num_samples = dataset.shape[0]
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                current_batch.append(dataset[start_idx:end_idx])

                # If current_batch size reaches or exceeds batch_size, yield it
                total_samples = sum(batch.shape[0] for batch in current_batch)
                if total_samples >= batch_size:
                    # Concatenate accumulated samples to form a single batch
                    batch = np.concatenate(current_batch, axis=0, dtype=np.float32)[:batch_size]
                    
                    # Remove used samples and prepare for the next batch
                    remaining_samples = np.concatenate(current_batch, axis=0, dtype=np.float32)[batch_size:]
                    current_batch = [remaining_samples] if remaining_samples.size > 0 else []
                    
                    # save batch
                    dst.create_dataset(f'subset_{idx}', data=batch, dtype=np.float32, chunks=(chunk_batch, chunk_length))
                    # print(f'subset_{idx}, batch size: {batch.shape}, batch dtype: {batch.dtype}')
                    idx += 1
        
        # By default, the last batch is droped
        
    return temp_file
    
if __name__ == "__main__":
    # pretrain Dataset
    hdf5_name = 'pretrain_wo_sfreq_cpres_chn'
    # pretrain_dataset = h5Dataset(hdf5_name, os.path.join(DATASETS_OUTPUT_PATH, 'pretrain'))
    pretrain_dataset = h5Dataset(hdf5_name, DATASETS_OUTPUT_PATH)
    equal_sfreq = False # whether to resample the data to the same sampling frequency
    compress_chn = True # whether to compress the channel number
    summary_df = OpenNeuro('DS003029', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS003498', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS003555', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS003844', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS003876', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS004100', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS004752', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = OpenNeuro('DS005398', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = Normal_EDF_EEG('TUEP', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = Normal_EDF_EEG('TUSZ', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = Normal_EDF_EEG('Siena', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)     
    summary_df = Normal_EDF_EEG('JHCH_JHMCHH', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    summary_df = Normal_EDF_EEG('Mendeley', pretrain_dataset)._process_data(equal_sfreq=equal_sfreq, compress_chn=compress_chn)
    pretrain_dataset.save()
    
    # output hdf5 file path
    # hdf5_file_path =os.path.join(DATASETS_OUTPUT_PATH, 'pretrain', hdf5_name+'.hdf5')
    hdf5_file_path =os.path.join(DATASETS_OUTPUT_PATH, hdf5_name+'.hdf5')
    
    # delete unwanted patients
    # remove_patient_groups(hdf5_file_path, 'iEEG')

    # check the structure of the HDF5 file
    # with h5py.File(hdf5_file_path, 'r') as f:
    #     print_hdf5_structure(f)
    
    # re-organize the HDF5 file to speed up data loading
    # random.seed(666)
    # reorg_file = reorganize_hdf5(hdf5_file_path, batch_size=2048, chunk_batch=128, chunk_length=int(duration * global_sfreq))

    # Get .hd5 file size in terabytes (1 TB = 1024^4 bytes)
    file_size = os.path.getsize(hdf5_file_path)
    file_size_mb = file_size / (1024 ** 4)
    print(f"File size: {file_size} bytes ({file_size_mb:.2f} TB)")
    
    # file_size = os.path.getsize(reorg_file)
    # file_size_mb = file_size / (1024 ** 4)
    # print(f"File size: {file_size} bytes ({file_size_mb:.2f} TB)")
