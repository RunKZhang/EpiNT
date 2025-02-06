import os
import sys
import argparse
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import h5py

def dataset_statistics(path, num_bins=10):
    results = {
        'mean': [],
        'median': [],
    }
    file = h5py.File(path, 'r')
    dataset_list = list(file.keys())
    dataset_list = [name for name in dataset_list if name.startswith('DS')]
    for dset in dataset_list:
        data = file[dset][:]
        
        # Compute mean and median
        mean = np.mean(data)
        median = np.median(data)
        
        # Generate histogram bins
        results['mean'].append(mean)
        results['median'].append(median)

        print(f'dataset name: {dset}, data mean: {np.abs(mean)}, data meadian: {np.abs(median)}, data range: {data.min()} to {data.max()}, std: {data.std()}')

        # for sample in data:
        #     sample = sample * 1e-6
        #     print(f'sample mean: {np.abs(sample.mean())}, sample std: {sample.std()}, sample range: {sample.min()} to {sample.max()}')
        # break
    return results

def downstream_dataset(path):
    file = h5py.File(path, 'r')
    group_list = list(file.keys())
    for grp in group_list:
        for dset in file[grp]:
            # print(f'group name: {grp}, dataset name: {dset}')
            data = file[grp][dset]['data'][:]
            print(f'group name: {grp}, dataset name: {dset}, data shape: {data.shape}, data range: {data.min()} to {data.max()}, data std: {data.std()}')
            # for sample in data:
            #     sample = sample * 1e-6
            #     print(f'sample mean: {np.abs(sample.mean())}, sample std: {sample.std()}, sample range: {sample.min()} to {sample.max()}')
            # break


if __name__ == '__main__':
    # path = '/home/ZRK/ZRK_ssd2/ZRK/Engineering_Server/Epileptogenic/dataset/pretrain_wo_sfreq_cpres_chn.hdf5'
    path = '/home/ZRK/ZRK_ssd2/ZRK/Engineering_Server/Epileptogenic/dataset/downstream_w_sfreq_equal_1024_cpres_chn.hdf5'
    # calculate group mean
    # results = dataset_statistics(path)
    downstream_dataset(path)
    