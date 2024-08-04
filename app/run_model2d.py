import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import gzip
from scipy.interpolate import griddata
import pickle
from imread import imread, imsave

#setting
basetime_dir = 'data_model/gfs'
data_gsmap = 'data'
dir_overlay = 'overlay'
len_day_predict = 3
lat_low = -25.0
lat_high = 40.0
lon_low = 70.0
lon_high = 170.0
threshold_hujan = 0.5

def search_index(mat, coordinate_target):
    for i in range(len(mat)):
        if mat[i] > coordinate_target and i != 0:
            if abs(coordinate_target-mat[i]) < abs(coordinate_target-mat[i-1]):
                return i
            else:
                return i-1
    return 0

def regrid_data(param_old, x_old, y_old, x_new, y_new):
    target_points = np.array([x_new.ravel(), y_new.ravel()]).T
    source_points = np.array([x_old.ravel(), y_old.ravel()]).T
    #regridded_data = griddata(source_points, np.reshape(param_old, (-1)), target_points, method='linear')
    regridded_data = griddata(source_points, np.reshape(param_old, (-1)), target_points, method='nearest')
    regridded_data = regridded_data.reshape(x_new.shape)
    return regridded_data

def determine_quadrant(value, percentiles):
    # Ensure percentiles are sorted
    percentiles = sorted(percentiles)
    
    for i, p in enumerate(percentiles):
        if value < p:
            return i
    return len(percentiles)

lon_mat_target = np.linspace(0.05,359.95,3600)
lat_mat_target = np.linspace(59.95,-59.95,1200)

lon_mat_target = np.linspace(0.05,359.95,3600)
lat_mat_target = np.linspace(59.95,-59.95,1200)
lat_mat_target = np.flip(lat_mat_target)

X, Y = np.meshgrid(lon_mat_target, lat_mat_target)

all_basetime = sorted(glob(basetime_dir+'/*'))
#all_basetime_dt = []
#for iter_bt in range(len(all_basetime)):
#    single_bt = all_basetime[iter_bt]
#    single_bt_dt = datetime.strptime(single_bt, basetime_dir+'/'+'%Y%m%d')
#    all_basetime_dt.append(single_bt_dt)
#
#idx_sort = np.argsort(all_basetime_dt)
#print(idx_sort)
#all_basetime_dt = all_basetime_dt[idx_sort]
#print(all_basetime_dt)

used_basetime = all_basetime[-1]
used_basetime_dt = datetime.strptime(used_basetime, basetime_dir+'/'+'%Y%m%d')
print(used_basetime)

with open('brain_full.pkl', 'rb') as fp:
    map_emit = pickle.load(fp)

arr_hujans = []
arr_cloud_clover = []

arr_T1000 = []
arr_rh1000 = []
arr_wspd1000 = []
    
arr_T850 = []
arr_rh850 = []
arr_wspd850 = []

arr_T700 = []
arr_rh700 = []
arr_wspd700 = []

arr_T500 = []
arr_rh500 = []
arr_wspd500 = []
lat = []
lon = []
lat2d = []
lon2d = []
for iter_file in range(1, 73):
    if iter_file < 10:
        iter_file_str = '00%s'%(iter_file)
    elif iter_file >= 10 and iter_file < 100:
        iter_file_str = '0%s'%(iter_file)
    else:
        iter_file_str = str(iter_file)
    single_file = 'gfs.t00z.pgrb2.0p25.f%s'%(iter_file_str)
    #cek hujan
    if iter_file % 6 == 0:
        ds = xr.open_dataset(used_basetime+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'accum'})
        if len(lat) == 0 and len(lon) == 0:
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            lon2d, lat2d = np.meshgrid(lon, lat)
        hujan = ds['tp'].values
        arr_hujans.append(hujan)
    
    ds_atmos = xr.open_dataset(used_basetime+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'atmosphere', 'stepType': 'avg'})
    cloud_clover = ds_atmos['tcc'].values
    arr_cloud_clover.append(cloud_clover)

    ds_isobaric = xr.open_dataset(used_basetime+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
    t1000_model = ds_isobaric['t'].sel(isobaricInhPa=1000).values
    t850_model = ds_isobaric['t'].sel(isobaricInhPa=850).values
    t700_model = ds_isobaric['t'].sel(isobaricInhPa=700).values
    t500_model = ds_isobaric['t'].sel(isobaricInhPa=500).values
        
    arr_T1000.append(t1000_model)
    arr_T850.append(t850_model)
    arr_T700.append(t700_model)
    arr_T500.append(t500_model)

    r1000_model = ds_isobaric['r'].sel(isobaricInhPa=1000).values
    r850_model = ds_isobaric['r'].sel(isobaricInhPa=850).values
    r700_model = ds_isobaric['r'].sel(isobaricInhPa=700).values
    r500_model = ds_isobaric['r'].sel(isobaricInhPa=500).values

    arr_rh1000.append(r1000_model)
    arr_rh850.append(r850_model)
    arr_rh700.append(r700_model)
    arr_rh500.append(r500_model)

    u1000_model = ds_isobaric['u'].sel(isobaricInhPa=1000).values
    u850_model = ds_isobaric['u'].sel(isobaricInhPa=850).values
    u700_model = ds_isobaric['u'].sel(isobaricInhPa=700).values
    u500_model = ds_isobaric['u'].sel(isobaricInhPa=500).values

    v1000_model = ds_isobaric['v'].sel(isobaricInhPa=1000).values
    v850_model = ds_isobaric['v'].sel(isobaricInhPa=850).values
    v700_model = ds_isobaric['v'].sel(isobaricInhPa=700).values
    v500_model = ds_isobaric['v'].sel(isobaricInhPa=500).values

    wspd_1000 = (u1000_model**2 + v1000_model**2)**0.5
    wspd_850 = (u850_model**2 + v850_model**2)**0.5
    wspd_700 = (u700_model**2 + v700_model**2)**0.5
    wspd_500 = (u500_model**2 + v500_model**2)**0.5

    arr_wspd1000.append(wspd_1000)
    arr_wspd850.append(wspd_850)
    arr_wspd700.append(wspd_700)
    arr_wspd500.append(wspd_500)

#finalize hujan
arr_hujans = np.array(arr_hujans)
#print('cek arr hujan', arr_hujans)
hujan_daily = np.zeros((len_day_predict, len(arr_hujans[0]), len(arr_hujans[0,0])))
for iter_daily in range(len_day_predict):
    hujan_daily[iter_daily, :, :] = np.sum(arr_hujans[4*iter_daily:4*(iter_daily+1)], axis=0)

#finalize cloud
arr_cloud_clover = np.array(arr_cloud_clover)
arr_cloud_clover[np.isnan(arr_cloud_clover)] = 0
cloud_clover_daily = np.zeros((len_day_predict, len(arr_cloud_clover[0]), len(arr_cloud_clover[0,0])))
for iter_daily in range(len_day_predict):
    cloud_clover_daily[iter_daily, :, :] = np.nanmean(arr_cloud_clover[24*iter_daily:24*(iter_daily+1)], axis=0)
cloud_clover_daily[np.isnan(cloud_clover_daily)] = 0
    
#finalize temperature
arr_T1000 = np.array(arr_T1000)
arr_T850 = np.array(arr_T850)
arr_T700 = np.array(arr_T700)
arr_T500 = np.array(arr_T500)

T1000_daily = np.zeros((len_day_predict, len(arr_T1000[0]), len(arr_T1000[0,0])))
T850_daily = np.zeros((len_day_predict, len(arr_T1000[0]), len(arr_T1000[0,0])))
T700_daily = np.zeros((len_day_predict, len(arr_T1000[0]), len(arr_T1000[0,0])))
T500_daily = np.zeros((len_day_predict, len(arr_T1000[0]), len(arr_T1000[0,0])))

for iter_daily in range(len_day_predict):
    T1000_daily[iter_daily, :, :] = np.mean(arr_T1000[24*iter_daily:24*(iter_daily+1)], axis=0)
    T850_daily[iter_daily, :, :] = np.mean(arr_T850[24*iter_daily:24*(iter_daily+1)], axis=0)
    T700_daily[iter_daily, :, :] = np.mean(arr_T700[24*iter_daily:24*(iter_daily+1)], axis=0)
    T500_daily[iter_daily, :, :] = np.mean(arr_T500[24*iter_daily:24*(iter_daily+1)], axis=0)
    
#finalize rh
arr_rh1000 = np.array(arr_rh1000)
arr_rh850 = np.array(arr_rh850)
arr_rh700 = np.array(arr_rh700)
arr_rh500 = np.array(arr_rh500)

rh1000_daily = np.zeros((len_day_predict, len(arr_rh1000[0]), len(arr_rh1000[0,0])))
rh850_daily = np.zeros((len_day_predict, len(arr_rh1000[0]), len(arr_rh1000[0,0])))
rh700_daily = np.zeros((len_day_predict, len(arr_rh1000[0]), len(arr_rh1000[0,0])))
rh500_daily = np.zeros((len_day_predict, len(arr_rh1000[0]), len(arr_rh1000[0,0])))

for iter_daily in range(len_day_predict):
    rh1000_daily[iter_daily, :, :] = np.mean(arr_rh1000[24*iter_daily:24*(iter_daily+1)], axis=0)
    rh850_daily[iter_daily, :, :] = np.mean(arr_rh850[24*iter_daily:24*(iter_daily+1)], axis=0)
    rh700_daily[iter_daily, :, :] = np.mean(arr_rh700[24*iter_daily:24*(iter_daily+1)], axis=0)
    rh500_daily[iter_daily, :, :] = np.mean(arr_rh500[24*iter_daily:24*(iter_daily+1)], axis=0)
    
#finalize wspd
arr_wspd1000 = np.array(arr_wspd1000)
arr_wspd850 = np.array(arr_wspd850)
arr_wspd700 = np.array(arr_wspd700)
arr_wspd500 = np.array(arr_wspd500)

wspd1000_daily = np.zeros((len_day_predict, len(arr_wspd1000[0]), len(arr_wspd1000[0,0])))
wspd850_daily = np.zeros((len_day_predict, len(arr_wspd1000[0]), len(arr_wspd1000[0,0])))
wspd700_daily = np.zeros((len_day_predict, len(arr_wspd1000[0]), len(arr_wspd1000[0,0])))
wspd500_daily = np.zeros((len_day_predict, len(arr_wspd1000[0]), len(arr_wspd1000[0,0])))

for iter_daily in range(len_day_predict):
    wspd1000_daily[iter_daily, :, :] = np.mean(arr_wspd1000[24*iter_daily:24*(iter_daily+1)], axis=0)
    wspd850_daily[iter_daily, :, :] = np.mean(arr_wspd850[24*iter_daily:24*(iter_daily+1)], axis=0)
    wspd700_daily[iter_daily, :, :] = np.mean(arr_wspd700[24*iter_daily:24*(iter_daily+1)], axis=0)
    wspd500_daily[iter_daily, :, :] = np.mean(arr_wspd500[24*iter_daily:24*(iter_daily+1)], axis=0)

#get transprob
used_basetime_dt_prev = used_basetime_dt - timedelta(days=1)
gsmap_file = data_gsmap+'/'+datetime.strftime(used_basetime_dt_prev, 'gsmap_gauge.%Y%m%d.0.1d.daily.00Z-23Z.dat.gz')

#print('cek gsmap file', gsmap_file)

lon_low_idx = search_index(lon_mat_target, lon_low)
lon_high_idx = search_index(lon_mat_target, lon_high)
lat_low_idx = search_index(lat_mat_target, lat_low)
lat_high_idx = search_index(lat_mat_target, lat_high)

gz = gzip.GzipFile(gsmap_file, 'rb')
gsmap_data = np.frombuffer(gz.read(),dtype=np.float32)
gsmap_data = gsmap_data.reshape((1200,3600))
gsmap_data = gsmap_data[::-1]
gsmap_data = gsmap_data[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
gsmap_data = np.copy(gsmap_data)
gsmap_data[gsmap_data == -99] = np.nan

X_used = X[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
Y_used = Y[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]

gsmap_data_regrid = regrid_data(gsmap_data, X_used, Y_used, lon2d, lat2d)
gsmap_data_regrid_1d = gsmap_data_regrid.ravel()

predict_2d = np.zeros((len_day_predict, len(lon2d), len(lon2d[0])))
for iter_step in range(len_day_predict):
    if iter_step == 0:
        previous_rain = gsmap_data_regrid_1d
    single_hujan_daily = hujan_daily[iter_step]
    single_hujan_daily = single_hujan_daily.ravel()

    single_cc_daily = cloud_clover_daily[iter_step]
    single_cc_daily = single_cc_daily.ravel()
    
    single_T1000_daily = T1000_daily[iter_step]
    single_T850_daily = T850_daily[iter_step]
    single_T700_daily = T700_daily[iter_step]
    single_T500_daily = T500_daily[iter_step]

    single_T1000_daily = single_T1000_daily.ravel()
    single_T850_daily = single_T850_daily.ravel()
    single_T700_daily = single_T700_daily.ravel()
    single_T500_daily = single_T500_daily.ravel()

    single_rh1000_daily = rh1000_daily[iter_step]
    single_rh850_daily = rh850_daily[iter_step]
    single_rh700_daily = rh700_daily[iter_step]
    single_rh500_daily = rh500_daily[iter_step]

    single_rh1000_daily = single_rh1000_daily.ravel()
    single_rh850_daily = single_rh850_daily.ravel()
    single_rh700_daily = single_rh700_daily.ravel()
    single_rh500_daily = single_rh500_daily.ravel()

    single_wspd1000_daily = wspd1000_daily[iter_step]
    single_wspd850_daily = wspd850_daily[iter_step]
    single_wspd700_daily = wspd700_daily[iter_step]
    single_wspd500_daily = wspd500_daily[iter_step]

    single_wspd1000_daily = single_wspd1000_daily.ravel()
    single_wspd850_daily = single_wspd850_daily.ravel()
    single_wspd700_daily = single_wspd700_daily.ravel()
    single_wspd500_daily = single_wspd500_daily.ravel()

    single_2d_1d = np.zeros((len(lon) * len(lat)))
    for iter_ravel in range(len(previous_rain)):
        single_grid_gsmap = previous_rain[iter_ravel]
        single_grid_hujan = single_hujan_daily[iter_ravel]
        single_grid_cc = single_cc_daily[iter_ravel]

        single_grid_T1000 = single_T1000_daily[iter_ravel]
        single_grid_T850 = single_T850_daily[iter_ravel]
        single_grid_T700 = single_T700_daily[iter_ravel]
        single_grid_T500 = single_T500_daily[iter_ravel]

        single_grid_rh1000 = single_rh1000_daily[iter_ravel]
        single_grid_rh850 = single_rh850_daily[iter_ravel]
        single_grid_rh700 = single_rh700_daily[iter_ravel]
        single_grid_rh500 = single_rh500_daily[iter_ravel]

        single_grid_wspd1000 = single_wspd1000_daily[iter_ravel]
        single_grid_wspd850 = single_wspd850_daily[iter_ravel]
        single_grid_wspd700 = single_wspd700_daily[iter_ravel]
        single_grid_wspd500 = single_wspd500_daily[iter_ravel]

        single_grid_hujan_q = determine_quadrant(single_grid_hujan, map_emit['obs_percentile'][0])
        single_grid_cc_q = determine_quadrant(single_grid_cc, map_emit['obs_percentile'][1])

        single_grid_T1000_q = determine_quadrant(single_grid_T1000, map_emit['obs_percentile'][2])
        single_grid_T850_q = determine_quadrant(single_grid_T850, map_emit['obs_percentile'][3])
        single_grid_T700_q = determine_quadrant(single_grid_T700, map_emit['obs_percentile'][4])
        single_grid_T500_q = determine_quadrant(single_grid_T500, map_emit['obs_percentile'][5])

        single_grid_rh1000_q = determine_quadrant(single_grid_rh1000, map_emit['obs_percentile'][6])
        single_grid_rh850_q = determine_quadrant(single_grid_rh850, map_emit['obs_percentile'][7])
        single_grid_rh700_q = determine_quadrant(single_grid_rh700, map_emit['obs_percentile'][8])
        single_grid_rh500_q = determine_quadrant(single_grid_rh500, map_emit['obs_percentile'][9])

        single_grid_wspd1000_q = determine_quadrant(single_grid_wspd1000, map_emit['obs_percentile'][10])
        single_grid_wspd850_q = determine_quadrant(single_grid_wspd850, map_emit['obs_percentile'][11])
        single_grid_wspd700_q = determine_quadrant(single_grid_wspd700, map_emit['obs_percentile'][12])
        single_grid_wspd500_q = determine_quadrant(single_grid_wspd500, map_emit['obs_percentile'][13])

        #print('cek gsmap', single_grid_gsmap)
        if single_grid_gsmap > threshold_hujan:
            transprob_yes = map_emit['transition_value'][0,0]
            transprob_no = map_emit['transition_value'][0,1]
        else:
            transprob_yes = map_emit['transition_value'][1,0]
            transprob_no = map_emit['transition_value'][1,1]
        
        prob_yes = transprob_yes * map_emit['emit_label']['yes'][0, single_grid_hujan_q] * map_emit['emit_label']['yes'][1, single_grid_cc_q] * \
        map_emit['emit_label']['yes'][2, single_grid_T1000_q] * map_emit['emit_label']['yes'][3, single_grid_T850_q] * map_emit['emit_label']['yes'][4, single_grid_T700_q] * map_emit['emit_label']['yes'][5, single_grid_T500_q] * \
        map_emit['emit_label']['yes'][6, single_grid_rh1000_q] * map_emit['emit_label']['yes'][7, single_grid_rh850_q] * map_emit['emit_label']['yes'][8, single_grid_rh700_q] * map_emit['emit_label']['yes'][9, single_grid_rh500_q] * \
        map_emit['emit_label']['yes'][10, single_grid_wspd1000_q] * map_emit['emit_label']['yes'][11, single_grid_wspd850_q] * map_emit['emit_label']['yes'][12, single_grid_wspd700_q] * map_emit['emit_label']['yes'][13, single_grid_wspd500_q]


        prob_no = transprob_no * map_emit['emit_label']['no'][0, single_grid_hujan_q] * map_emit['emit_label']['no'][1, single_grid_cc_q] * \
        map_emit['emit_label']['no'][2, single_grid_T1000_q] * map_emit['emit_label']['no'][3, single_grid_T850_q] * map_emit['emit_label']['no'][4, single_grid_T700_q] * map_emit['emit_label']['no'][5, single_grid_T500_q] * \
        map_emit['emit_label']['no'][6, single_grid_rh1000_q] * map_emit['emit_label']['no'][7, single_grid_rh850_q] * map_emit['emit_label']['no'][8, single_grid_rh700_q] * map_emit['emit_label']['no'][9, single_grid_rh500_q] * \
        map_emit['emit_label']['no'][10, single_grid_wspd1000_q] * map_emit['emit_label']['no'][11, single_grid_wspd850_q] * map_emit['emit_label']['no'][12, single_grid_wspd700_q] * map_emit['emit_label']['no'][13, single_grid_wspd500_q]

        real_prob_yes = prob_yes / (prob_yes + prob_no)
        if real_prob_yes > 0.5:
            single_2d_1d[iter_ravel] = 1
        #print('cek prob', real_prob_yes)
    
    single_2d = np.reshape(single_2d_1d, (len(lat), len(lon)))
    predict_2d[iter_step, :, :] = single_2d
    previous_rain = single_2d.ravel()
    #print('cek bin', np.sum(single_2d))
    #break

#save image
citra = np.zeros((len(predict_2d[0]), len(predict_2d[0,0]), 4), dtype=np.uint8)
for iter_p in range(len_day_predict):
    single_predict = predict_2d[iter_p]

    foto_red = np.zeros((len(single_predict), len(single_predict[0])), dtype=np.uint8)
    foto_green = np.zeros((len(single_predict), len(single_predict[0])), dtype=np.uint8)
    foto_blue = np.zeros((len(single_predict), len(single_predict[0])), dtype=np.uint8)
    foto_opacity = np.zeros((len(single_predict), len(single_predict[0])), dtype=np.uint8)

    foto_red[single_predict > 0] = 255
    foto_opacity[single_predict > 0] = 230

    #merge all rgb and opacity
    citra[:,:,0] = foto_red
    citra[:,:,1] = foto_green
    citra[:,:,2] = foto_blue
    citra[:,:,3] = foto_opacity

    imsave(dir_overlay+'/'+'%s.png'%(iter_p), citra)

#investigate in 1d
