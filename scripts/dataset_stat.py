import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EpiNT.data.dataset_info import openneuron, TUEP, TUSZ, Siena, JHCH_JHMCHH, Mendeley
from prettytable import PrettyTable

if __name__ == "__main__":
    column_names = ['Dataset', 'Number of Patients', 'Recording Duration (hours)', 'Sampling Frequency', 'Types']
    showtable = PrettyTable(column_names)

    ds_list = []
    for ds_name in ['ds003029', 'ds003498', 'ds003555', 'ds003844', 'ds003876', 'ds004100', 'ds004752', 'ds005398']:
        ds = openneuron(ds_name).loop()
        ds_list.append(ds)
        showtable.add_row([ds_name, ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])

    # TUEP
    ds = TUEP().loop()
    ds_list.append(ds)
    showtable.add_row(['TUEP', ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])

    # # TUSZ
    ds = TUSZ().loop()
    ds_list.append(ds)
    showtable.add_row(['TUSZ', ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])
        
    # Siena
    ds = Siena().loop()
    ds_list.append(ds)
    showtable.add_row(['Siena', ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])

    # JH
    ds = JHCH_JHMCHH().loop()
    ds_list.append(ds)
    showtable.add_row(['JHCH_JHMCHH', ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])

    # Mendeley
    ds = Mendeley().loop()
    ds_list.append(ds)
    showtable.add_row(['Mendeley', ds['Number of Patients'], ds['Durations'], ds['Sampling Frequency'], ds['Types']])
    
    print(showtable)