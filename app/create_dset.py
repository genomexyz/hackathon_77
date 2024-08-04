import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import gzip
from scipy.interpolate import griddata
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#setting
sample_file = '../data/gfs/gfs.t00z.pgrb2.0p25.f018'
basetime_dir = 'data_model/gfs'
data_gsmap = 'data'
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

#ds = xr.open_dataset(sample_file, engine='cfgrib')
#print(ds)

#check all basetime
all_basetime = sorted(glob(basetime_dir+'/*'))
print(all_basetime)

hujan_dailys = []
gsmap_dailys = []
cloud_clover_dailys = []
T1000_dailys = []
T850_dailys = []
T700_dailys = []
T500_dailys = []

rh1000_dailys = []
rh850_dailys = []
rh700_dailys = []
rh500_dailys = []

wspd1000_dailys = []
wspd850_dailys = []
wspd700_dailys = []
wspd500_dailys = []

for iter_bt in range(len(all_basetime)):
    single_bt = all_basetime[iter_bt]

    single_bt_dt = datetime.strptime(single_bt, basetime_dir+'/'+'%Y%m%d')

    #single_gsmap = data_gsmap+'/'+datetime.strftime(single_bt_dt, 'gsmap_gauge.%Y%m%d.0.1d.daily.00Z-23Z.dat.gz')
    #print(single_gsmap)
    #continue

    #iterate all timestep
    #cnt = 0
    #prev_hujan = []
    #after_hujan = []
    #arr_hujans = np.zeros(len_day_predict*4) # 24 hours / 6 = 4
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
            ds = xr.open_dataset(single_bt+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface', 'stepType': 'accum'})
            if len(lat) == 0 and len(lon) == 0:
                lat = ds['latitude'].values
                lon = ds['longitude'].values
                lon2d, lat2d = np.meshgrid(lon, lat)
            hujan = ds['tp'].values
            arr_hujans.append(hujan)
            #if cnt == 0:
            #    prev_hujan = hujan
            #    after_hujan = hujan
            #else:
            #    prev_hujan = after_hujan
            #    after_hujan = hujan
            #    diff_hujan = after_hujan - prev_hujan
            #    min_dif = np.min(diff_hujan)
            #    max_dif = np.max(diff_hujan)
            #    avg_dif = np.mean(diff_hujan)
            #    print('stat', min_dif, avg_dif, max_dif, single_file)
        #print(np.sum(hujan))
        #print('lat', ds['latitude'].values)
        #print('lon', ds['longitude'].values)

        ds_atmos = xr.open_dataset(single_bt+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'atmosphere', 'stepType': 'avg'})
        cloud_clover = ds_atmos['tcc'].values
        arr_cloud_clover.append(cloud_clover)

        ds_isobaric = xr.open_dataset(single_bt+'/'+single_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
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
        #print(wspd_850)
        #exit()
    
    #finalize hujan
    arr_hujans = np.array(arr_hujans)
    hujan_daily = np.zeros((len_day_predict, len(arr_hujans[0]), len(arr_hujans[0,0])))
    for iter_daily in range(len_day_predict):
        hujan_daily[iter_daily, :, :] = np.sum(arr_hujans[4*iter_daily:4*(iter_daily+1)], axis=0)
    

    #finalize cloud
    arr_cloud_clover = np.array(arr_cloud_clover)
    arr_cloud_clover[np.isnan(arr_cloud_clover)] = 0
    cloud_clover_daily = np.zeros((len_day_predict, len(arr_cloud_clover[0]), len(arr_cloud_clover[0,0])))
    for iter_daily in range(len_day_predict):
        cloud_clover_daily[iter_daily, :, :] = np.nanmean(arr_hujans[24*iter_daily:24*(iter_daily+1)], axis=0)
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

    #print('cc', cloud_clover_daily)
    #print('T', T700_daily)
    #print('wspd', np.shape(wspd500_daily))

    hujan_dailys.append(hujan_daily)
    cloud_clover_dailys.append(cloud_clover_daily)
    T1000_dailys.append(T1000_daily)
    T850_dailys.append(T850_daily)
    T700_dailys.append(T700_daily)
    T500_dailys.append(T500_daily)

    rh1000_dailys.append(rh1000_daily)
    rh850_dailys.append(rh850_daily)
    rh700_dailys.append(rh700_daily)
    rh500_dailys.append(rh500_daily)

    wspd1000_dailys.append(wspd1000_daily)
    wspd850_dailys.append(rh850_daily)
    wspd700_dailys.append(rh700_daily)
    wspd500_dailys.append(rh500_daily)

    #lon_low = lon[0]
    #lon_high = lon[-1]
    #lat_low = lat[0]
    #lat_high = lat[-1]

    lon_low_idx = search_index(lon_mat_target, lon_low)
    lon_high_idx = search_index(lon_mat_target, lon_high)
    lat_low_idx = search_index(lat_mat_target, lat_low)
    lat_high_idx = search_index(lat_mat_target, lat_high)

    X_used = X[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
    Y_used = Y[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]


    gsmap_daily = []
    for iter_gsmap in range(len_day_predict):
        single_bt_dt_plus = single_bt_dt + timedelta(days=iter_gsmap)
        single_gsmap = data_gsmap+'/'+datetime.strftime(single_bt_dt_plus, 'gsmap_gauge.%Y%m%d.0.1d.daily.00Z-23Z.dat.gz')

        gz = gzip.GzipFile(single_gsmap, 'rb')
        gsmap_data = np.frombuffer(gz.read(),dtype=np.float32)
        gsmap_data = gsmap_data.reshape((1200,3600))
        gsmap_data = gsmap_data[::-1]
        gsmap_data = gsmap_data[lat_low_idx: lat_high_idx+1, lon_low_idx: lon_high_idx+1]
        gsmap_data = np.copy(gsmap_data)
        gsmap_data[gsmap_data == -99] = np.nan

        print('cek regrid', np.shape(gsmap_data), np.shape(X_used), np.shape(Y_used))
        gsmap_data_regrid = regrid_data(gsmap_data, X_used, Y_used, lon2d, lat2d)
        gsmap_daily.append(gsmap_data_regrid)
        #print(np.shape(gsmap_data_regrid))
    gsmap_daily = np.array(gsmap_daily)
    gsmap_dailys.append(gsmap_daily)

    #print(hujan_daily)
    #print(np.shape(hujan_daily))
    #print(np.sum(hujan_daily[2]))
        #cnt += 1
    #break

gsmap_dailys = np.array(gsmap_dailys)
hujan_dailys = np.array(hujan_dailys)
cloud_clover_dailys = np.array(cloud_clover_dailys)
T1000_dailys = np.array(T1000_dailys)
T850_dailys = np.array(T850_dailys)
T700_dailys = np.array(T700_dailys)
T500_dailys = np.array(T500_dailys)

rh1000_dailys = np.array(rh1000_dailys)
rh850_dailys = np.array(rh850_dailys)
rh700_dailys = np.array(rh700_dailys)
rh500_dailys = np.array(rh500_dailys)

wspd1000_dailys = np.array(wspd1000_dailys)
wspd850_dailys = np.array(wspd850_dailys)
wspd700_dailys = np.array(wspd700_dailys)
wspd500_dailys = np.array(wspd500_dailys)

#create brain
map_emit = {}
map_emit['historic'] = len_day_predict
map_emit['obs_list'] = ['hujan_model', 'cc_model', 'T1000', 'T850', 'T700', 'T500',
                        'rh1000', 'rh850', 'rh700', 'rh500',
                        'wspd1000', 'wspd850', 'wspd700', 'wspd500',]
map_emit['transition_label'] = [['yes', 'no'], ['yes', 'no']]
map_emit['transition_value'] = np.zeros((2,2))
map_emit['transition_value'][:,:] = 0.5

#calculate trans probability
yy_mat_sum = 0
nn_mat_sum = 0
ny_mat_sum = 0
yn_mat_sum = 0
trans_prob = np.zeros((2,2))
for iter_gt in range(len(gsmap_dailys)):
    single_set = gsmap_dailys[iter_gt]
    for iter_set in range(1, len(single_set)):
        prev_hujan = single_set[iter_set-1]
        after_hujan = single_set[iter_set]

        prev_hujan_bin = np.zeros_like(prev_hujan)
        after_hujan_bin = np.zeros_like(after_hujan)

        prev_hujan_bin[prev_hujan > threshold_hujan] = 1
        after_hujan_bin[after_hujan > threshold_hujan] = 1

        prev_hujan_bin = prev_hujan_bin.ravel()
        after_hujan_bin = after_hujan_bin.ravel()
        print(np.shape(prev_hujan), np.shape(after_hujan))

        combine_hujan = np.array([prev_hujan_bin, after_hujan_bin])
        sum_mat = np.sum(combine_hujan, axis=0)
        dif_mat = prev_hujan_bin - after_hujan_bin
        
        yy_mat = np.zeros_like(sum_mat)
        yy_mat[sum_mat == 2] = 1

        nn_mat = np.zeros_like(sum_mat)
        nn_mat[sum_mat == 0] = 1

        ny_mat = np.zeros_like(sum_mat)
        ny_mat[dif_mat == -1] = 1

        yn_mat = np.zeros_like(sum_mat)
        yn_mat[dif_mat == 1] = 1

        yy_mat_sum += np.sum(yy_mat)
        nn_mat_sum += np.sum(nn_mat)
        ny_mat_sum += np.sum(ny_mat)
        yn_mat_sum += np.sum(yn_mat)

trans_prob[0,0] = yy_mat_sum / (yy_mat_sum + yn_mat_sum)
trans_prob[0,1] = yn_mat_sum / (yy_mat_sum + yn_mat_sum)
        
trans_prob[1,0] = ny_mat_sum / (ny_mat_sum + nn_mat_sum)
trans_prob[1,1] = nn_mat_sum / (ny_mat_sum + nn_mat_sum)

print(trans_prob)

map_emit['transition_value'] = trans_prob

hujan_dailys_limits = np.array([np.percentile(hujan_dailys, 5), np.percentile(hujan_dailys, 50), np.percentile(hujan_dailys, 95)])
cloud_clover_dailys_limits = np.array([np.percentile(cloud_clover_dailys, 5), np.percentile(cloud_clover_dailys, 50), np.percentile(cloud_clover_dailys, 95)])
T1000_dailys_limits = np.array([np.percentile(T1000_dailys, 5), np.percentile(T1000_dailys, 50), np.percentile(T1000_dailys, 95)])
T850_dailys_limits = np.array([np.percentile(T850_dailys, 5), np.percentile(T850_dailys, 50), np.percentile(T850_dailys, 95)])
T700_dailys_limits = np.array([np.percentile(T700_dailys, 5), np.percentile(T700_dailys, 50), np.percentile(T700_dailys, 95)])
T500_dailys_limits = np.array([np.percentile(T500_dailys, 5), np.percentile(T500_dailys, 50), np.percentile(T500_dailys, 95)])

rh1000_dailys_limits = np.array([np.percentile(rh1000_dailys, 5), np.percentile(rh1000_dailys, 50), np.percentile(rh1000_dailys, 95)])
rh850_dailys_limits = np.array([np.percentile(rh850_dailys, 5), np.percentile(rh850_dailys, 50), np.percentile(rh850_dailys, 95)])
rh700_dailys_limits = np.array([np.percentile(rh700_dailys, 5), np.percentile(rh700_dailys, 50), np.percentile(rh700_dailys, 95)])
rh500_dailys_limits = np.array([np.percentile(rh500_dailys, 5), np.percentile(rh500_dailys, 50), np.percentile(rh500_dailys, 95)])

wspd1000_dailys_limits = np.array([np.percentile(wspd1000_dailys, 5), np.percentile(wspd1000_dailys, 50), np.percentile(wspd1000_dailys, 95)])
wspd850_dailys_limits = np.array([np.percentile(wspd850_dailys, 5), np.percentile(wspd850_dailys, 50), np.percentile(wspd850_dailys, 95)])
wspd700_dailys_limits = np.array([np.percentile(wspd700_dailys, 5), np.percentile(wspd700_dailys, 50), np.percentile(wspd700_dailys, 95)])
wspd500_dailys_limits = np.array([np.percentile(wspd500_dailys, 5), np.percentile(wspd500_dailys, 50), np.percentile(wspd500_dailys, 95)])

obs_percentile = np.zeros((14, 3))
obs_percentile[0,:] = hujan_dailys_limits
obs_percentile[1,:] = cloud_clover_dailys_limits
obs_percentile[2,:] = T1000_dailys_limits
obs_percentile[3,:] = T850_dailys_limits
obs_percentile[4,:] = T700_dailys_limits
obs_percentile[5,:] = T500_dailys_limits
obs_percentile[6,:] = rh1000_dailys_limits
obs_percentile[7,:] = rh850_dailys_limits
obs_percentile[8,:] = rh700_dailys_limits
obs_percentile[9,:] = rh500_dailys_limits
obs_percentile[10,:] = wspd1000_dailys_limits
obs_percentile[11,:] = wspd850_dailys_limits
obs_percentile[12,:] = wspd700_dailys_limits
obs_percentile[13,:] = wspd500_dailys_limits

map_emit['obs_percentile'] = obs_percentile

mat_freq_yes = np.zeros((14, 4))
mat_freq_no = np.zeros((14, 4))
for iter_gt in range(len(gsmap_dailys)):
    single_set = gsmap_dailys[iter_gt]
    single_hujan_model = hujan_dailys[iter_gt]
    single_cloud_clover_model = cloud_clover_dailys[iter_gt]

    single_T1000_model = T1000_dailys[iter_gt]
    single_T850_model = T850_dailys[iter_gt]
    single_T700_model = T700_dailys[iter_gt]
    single_T500_model = T500_dailys[iter_gt]

    single_rh1000_model = rh1000_dailys[iter_gt]
    single_rh850_model = rh850_dailys[iter_gt]
    single_rh700_model = rh700_dailys[iter_gt]
    single_rh500_model = rh500_dailys[iter_gt]

    single_wspd1000_model = wspd1000_dailys[iter_gt]
    single_wspd850_model = wspd850_dailys[iter_gt]
    single_wspd700_model = wspd700_dailys[iter_gt]
    single_wspd500_model = wspd500_dailys[iter_gt]
    for iter_set in range(len(single_set)):
        #hujan_period_gt_prev = single_set[iter_set-1].ravel()
        hujan_period_gt = single_set[iter_set].ravel()
        hujan_period_model = single_hujan_model[iter_set].ravel()
        cloud_clover_period_model = single_cloud_clover_model[iter_set].ravel()

        T1000_period_model = single_T1000_model[iter_set].ravel()
        T850_period_model = single_T850_model[iter_set].ravel()
        T700_period_model = single_T700_model[iter_set].ravel()
        T500_period_model = single_T500_model[iter_set].ravel()

        rh1000_period_model = single_rh1000_model[iter_set].ravel()
        rh850_period_model = single_rh850_model[iter_set].ravel()
        rh700_period_model = single_rh700_model[iter_set].ravel()
        rh500_period_model = single_rh500_model[iter_set].ravel()

        wspd1000_period_model = single_wspd1000_model[iter_set].ravel()
        wspd850_period_model = single_wspd850_model[iter_set].ravel()
        wspd700_period_model = single_wspd700_model[iter_set].ravel()
        wspd500_period_model = single_wspd500_model[iter_set].ravel()

        #determine quadrant
        #hujan_period_model_qindex = np.zeros_like(hujan_period_model)
        for q in range(len(hujan_period_model)):
            #hujan_period_model_qindex[q] = determine_quadrant(hujan_period_model[q], hujan_dailys_limits)
            hujan_model_q = determine_quadrant(hujan_period_model[q], hujan_dailys_limits)
            cloud_clover_model_q = determine_quadrant(cloud_clover_period_model[q], cloud_clover_dailys_limits)

            T1000_model_q = determine_quadrant(T1000_period_model[q], T1000_dailys_limits)
            T850_model_q = determine_quadrant(T850_period_model[q], T850_dailys_limits)
            T700_model_q = determine_quadrant(T700_period_model[q], T700_dailys_limits)
            T500_model_q = determine_quadrant(T500_period_model[q], T500_dailys_limits)

            rh1000_model_q = determine_quadrant(rh1000_period_model[q], rh1000_dailys_limits)
            rh850_model_q = determine_quadrant(rh850_period_model[q], rh850_dailys_limits)
            rh700_model_q = determine_quadrant(rh700_period_model[q], rh700_dailys_limits)
            rh500_model_q = determine_quadrant(rh500_period_model[q], rh500_dailys_limits)

            wspd1000_model_q = determine_quadrant(wspd1000_period_model[q], wspd1000_dailys_limits)
            wspd850_model_q = determine_quadrant(wspd850_period_model[q], wspd850_dailys_limits)
            wspd700_model_q = determine_quadrant(wspd700_period_model[q], wspd700_dailys_limits)
            wspd500_model_q = determine_quadrant(wspd500_period_model[q], wspd500_dailys_limits)
            
            if hujan_period_gt[q] > threshold_hujan:
                mat_freq_yes[0, hujan_model_q] += 1
                mat_freq_yes[1, cloud_clover_model_q] += 1
                mat_freq_yes[2, T1000_model_q] += 1
                mat_freq_yes[3, T850_model_q] += 1
                mat_freq_yes[4, T700_model_q] += 1
                mat_freq_yes[5, T500_model_q] += 1
                mat_freq_yes[6, rh1000_model_q] += 1
                mat_freq_yes[7, rh850_model_q] += 1
                mat_freq_yes[8, rh700_model_q] += 1
                mat_freq_yes[9, rh500_model_q] += 1
                mat_freq_yes[10, wspd1000_model_q] += 1
                mat_freq_yes[11, wspd850_model_q] += 1
                mat_freq_yes[12, wspd700_model_q] += 1
                mat_freq_yes[13, wspd500_model_q] += 1
            else:
                mat_freq_no[0, hujan_model_q] += 1
                mat_freq_no[1, cloud_clover_model_q] += 1
                mat_freq_no[2, T1000_model_q] += 1
                mat_freq_no[3, T850_model_q] += 1
                mat_freq_no[4, T700_model_q] += 1
                mat_freq_no[5, T500_model_q] += 1
                mat_freq_no[6, rh1000_model_q] += 1
                mat_freq_no[7, rh850_model_q] += 1
                mat_freq_no[8, rh700_model_q] += 1
                mat_freq_no[9, rh500_model_q] += 1
                mat_freq_no[10, wspd1000_model_q] += 1
                mat_freq_no[11, wspd850_model_q] += 1
                mat_freq_no[12, wspd700_model_q] += 1
                mat_freq_no[13, wspd500_model_q] += 1

        #print(hujan_period_model_qindex, hujan_dailys_limits)
        print(mat_freq_yes, mat_freq_no)

mat_freq_yes = mat_freq_yes.T / np.sum(mat_freq_yes, axis=1)
mat_freq_yes = mat_freq_yes.T

mat_freq_no = mat_freq_no.T / np.sum(mat_freq_no, axis=1)
mat_freq_no = mat_freq_no.T

print(mat_freq_yes)

map_emit['emit_label'] = {}
map_emit['emit_label']['yes'] = mat_freq_yes
map_emit['emit_label']['no'] = mat_freq_no

with open('brain_full.pkl', 'wb') as fp:
    pickle.dump(map_emit, fp, protocol=pickle.HIGHEST_PROTOCOL)