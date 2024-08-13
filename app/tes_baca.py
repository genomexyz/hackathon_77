import pandas as pd
import numpy as np

baca = pd.read_csv('lokasi.csv')
nama_area = list(baca['nama'])
lat_area = np.array(baca['lat'])
lon_area = np.array(baca['lon'])
print(lat_area)