import numpy as np
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from glob import glob
from imread import imread, imsave

#setting
lat_low = -19.98
lat_high = 20
lon_low = 90
lon_high = 149.98
plot_gsmap_dir = 'gsmap_plot'
plot_overlay = 'overlay'
data_gsmap = 'data'
file_sample = 'gsmap_gauge.20240705.0.1d.daily.00Z-23Z.dat.gz'
levels = [0.1, 5, 10, 20, 20.1]
XX = ''
YY = ''


def search_index(mat, coordinate_target):
    for i in range(len(mat)):
        if mat[i] > coordinate_target and i != 0:
            if abs(coordinate_target-mat[i]) < abs(coordinate_target-mat[i-1]):
                return i
            else:
                return i-1
    return 0

def trailing_zero(val):
    if val < 10:
        return '0'+str(val)
    else:
        return str(val)

def ploting_data(xx, yy, data, filename, title):
    global XX, YY
    m = Basemap(projection='merc',llcrnrlon=lon_low,llcrnrlat=lat_low,urcrnrlon=lon_high,urcrnrlat=lat_high, resolution='h')

    #XX, YY = m(xx, yy)
    if isinstance(XX, str) and isinstance(YY, str):
        XX, YY = m(xx, yy)
    plt.tight_layout()

    m.drawcountries(linewidth=0.25, color='blue')
    m.drawcoastlines(linewidth=0.25, color='blue')
    meridians = np.arange(0.,351.,10.)
    parallels = np.arange(-40.,40,10)
    m.drawmeridians(meridians, labels=[1, 0, 0, 1], fontsize=6, linewidth=0.5)
    m.drawparallels(parallels, labels=[0, 1, 1, 0], fontsize=6, linewidth=0.5)

    cs = m.contourf(XX, YY, data, levels)
    proxy = [plt.Rectangle((1, 1), 2, 2, fc=pc.get_facecolor()[0]) for pc in cs.collections]
    plt.legend(proxy, ["Slight", "Moderate", "Heavy", "Extreme"])
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.close()

lon_mat_target = np.linspace(0.05,359.95,3600)
lat_mat_target = np.linspace(59.95,-59.95,1200)

lon_mat_target = np.linspace(0.05,359.95,3600)
lat_mat_target = np.linspace(59.95,-59.95,1200)
lat_mat_target = np.flip(lat_mat_target)

X, Y = np.meshgrid(lon_mat_target, lat_mat_target)

lon_low_idx = search_index(lon_mat_target, lon_low)
lon_high_idx = search_index(lon_mat_target, lon_high)
lat_low_idx = search_index(lat_mat_target, lat_low)
lat_high_idx = search_index(lat_mat_target, lat_high)

#check existing overlay plot
all_files = glob(plot_overlay+'/*.dat.gz.png')
dt_files = []
for iter_file in range(len(all_files)):
    single_file = all_files[iter_file]
    try:
        datetime_single_file = datetime.strptime(single_file, plot_overlay+'/'+'gsmap_gauge.%Y%m%d.0.1d.daily.00Z-23Z.dat.gz.png')
    except ValueError:
        continue
    dt_files.append(datetime_single_file)

all_files = glob(data_gsmap+'/*.dat.gz')
for iter_file in range(len(all_files)):
    single_file = all_files[iter_file]
    datetime_single_file = datetime.strptime(single_file, data_gsmap+'/'+'gsmap_gauge.%Y%m%d.0.1d.daily.00Z-23Z.dat.gz')

    #if already plot, skip
    if datetime_single_file in dt_files:
        continue
    
    gz = gzip.GzipFile(single_file,'rb')
    gsmap_data = np.frombuffer(gz.read(),dtype=np.float32)
    gsmap_data = gsmap_data.reshape((1200,3600))
    gsmap_data = gsmap_data[::-1]
    gsmap_data = gsmap_data[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
    gsmap_data = np.copy(gsmap_data)
    gsmap_data[gsmap_data == -99] = np.nan

    X = X[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
    Y = Y[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]

    ploting_data(X, Y, gsmap_data, plot_overlay+'/gsmap_%s%s%s.%s%s.png'%(datetime_single_file.year, trailing_zero(datetime_single_file.month), 
        trailing_zero(datetime_single_file.day), trailing_zero(datetime_single_file.hour), trailing_zero(datetime_single_file.minute)), 'gsmap %s-%s-%s\
        %s:%s'%(datetime_single_file.year, trailing_zero(datetime_single_file.month), 
        trailing_zero(datetime_single_file.day), trailing_zero(datetime_single_file.hour), trailing_zero(datetime_single_file.minute)))