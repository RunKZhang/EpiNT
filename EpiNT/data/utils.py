import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from .constants import TCP_AR_01, TCP_AR_A_03, TCP_LE_02

def tuh_montage(montage):
    if montage == '01_tcp_ar':
        return TCP_AR_01['ANODE'], TCP_AR_01['CATHNODE'], TCP_AR_01['CH_NAME']
    elif montage == '03_tcp_ar_a':
        return TCP_AR_A_03['ANODE'], TCP_AR_A_03['CATHODE'], TCP_AR_A_03['CH_NAME']
    elif montage == '02_tcp_le':
        return TCP_LE_02['ANODE'], TCP_LE_02['CATHODE'], TCP_LE_02['CH_NAME']
    else:
        raise ValueError(f"Montage {montage} not supported.")

class h5Dataset:
    def __init__(self, name:str, path:Path) -> None:
        self.__name = name
        # file_path = os.path.join(path, f'{name}.hdf5')
        self.__f = h5py.File(os.path.join(path, f'{name}.hdf5'), 'a')
        self.__csv_path = os.path.join(path, f'{name}.csv')
        if os.path.exists(self.__csv_path):
            self.df = pd.read_csv(self.__csv_path)
        else:
            self.df = pd.DataFrame(columns=['Patient_id', 'Session', 'Seconds', 'Type', 'Sampling Frequency', 'Channel Number'])
            self.df.to_csv(self.__csv_path, index=False)

    def __getitem__(self, key: str):
        """Enable subscripting to access datasets or groups."""
        return self.__f[key]
    
    def addGroup(self, grpName:str):
        return self.__f.create_group(grpName)
    
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, 
                #    chunks = False
                   ):
        return grp.create_dataset(dsName, data=arr, 
                                #   chunks=chunks
                                  )
    def createDataset(self, dsName:str, arr:np.array):
        return self.__f.create_dataset(dsName, data=arr)
    
    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue

    def save(self):
        self.__f.close()
    
    def store_dataframe(self, provide_df: pd.DataFrame):
        """Store the provided DataFrame as a .csv file in the same directory."""
        if not provide_df.isin(self.df).all(axis=None):
            new_df = pd.concat([self.df, provide_df], ignore_index=True)
            new_df.to_csv(self.__csv_path, index=False)


    @property
    def name(self):
        return self.__name
    

