import numpy as np
import pandas as pd

#setting
file_dset = 'Data Fklim BMKG 2023.csv'
wmo2kota = {}
wmo2kota[96017] = 'Kota Banda Aceh'
wmo2kota[96753] = 'Kota Bogor'
wmo2kota[96687] = 'Kota Banjarmasin'
wmo2kota[97184] = 'Kota Makassar'
wmo2kota[97692] = 'Kota Jayapura'

df = pd.read_csv(file_dset)
#print(df['WMO ID'].unique())

all_keys = list(dict.keys(wmo2kota))
for iter_keys in range(len(all_keys)):
    single_keys = all_keys[iter_keys]
    single_wmo_data = df[df['WMO ID'] == single_keys]

    #get emit probability
    
    print(single_wmo_data)

    break