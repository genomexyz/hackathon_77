import numpy as np
import pandas as pd
import pickle

#setting
file_dset = 'Data Fklim BMKG 2023.csv'
len_historic = 2
num_param = 5
len_predict = 3
wmo2kota = {}
wmo2kota[96017] = 'Kota Banda Aceh'
wmo2kota[96753] = 'Kota Bogor'
wmo2kota[96687] = 'Kota Banjarmasin'
wmo2kota[97184] = 'Kota Makassar'
wmo2kota[97692] = 'Kota Jayapura'

df = pd.read_csv(file_dset)

df['RAINFALL 24H MM'] = df['RAINFALL 24H MM'].replace(8888, 0)
df['RAINFALL 24H MM'] = df['RAINFALL 24H MM'].replace(np.nan, 0)

df['SUNSHINE 24H H'] = df['SUNSHINE 24H H'].replace(9999, 0)
df['SUNSHINE 24H H'] = df['SUNSHINE 24H H'].replace(np.nan, 0)

#print(df['WMO ID'].unique())

#param
#temperature avg, rh avg, windspeed avg, sun total
map_emit = {}
map_emit['historic'] = len_historic
map_emit['transition_label'] = [['yes', 'no'], ['yes', 'no']]
map_emit['transition_value'] = {}
map_emit['obs_list'] = ['temperature_avg', 'rh_avg', 'wspd_avg', 'sunshine_24h']
map_emit['emit_label'] = {}
#map_emit['emit_label']['yes'] = {}
#map_emit['emit_label']['no'] = {}

all_keys = list(dict.keys(wmo2kota))
for iter_keys in range(len(all_keys)):
    single_keys = all_keys[iter_keys]
    single_wmo_data = df[df['WMO ID'] == single_keys]

    print('cek len', len(single_wmo_data))

    #get trans probability
    vector_rain = np.array(single_wmo_data['RAINFALL 24H MM'])
    vector_rain[vector_rain > 0] = 1
    vector_rain[vector_rain <= 0] = 0
    
    matrix_freq = np.zeros((2,2))
    matrix_freq2 = np.zeros((2,2))
    matrix_freq3 = np.zeros((2,2))

    for iter_v in range(3, len(vector_rain)):
        #print('cek seq', vector_rain[iter_v-1], vector_rain[iter_v])
        if vector_rain[iter_v-3] == 0 and vector_rain[iter_v-2] == 0:
            matrix_freq[1,1] += 1
        elif vector_rain[iter_v-3] == 0 and vector_rain[iter_v-2] == 1:
            matrix_freq[0,1] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v-2] == 0:
            matrix_freq[1,0] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v-2] == 1:
            matrix_freq[0,0] += 1
        
        if vector_rain[iter_v-3] == 0 and vector_rain[iter_v-1] == 0:
            matrix_freq2[1,1] += 1
        elif vector_rain[iter_v-3] == 0 and vector_rain[iter_v-1] == 1:
            matrix_freq2[0,1] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v-1] == 0:
            matrix_freq2[1,0] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v-1] == 1:
            matrix_freq2[0,0] += 1
        
        if vector_rain[iter_v-3] == 0 and vector_rain[iter_v] == 0:
            matrix_freq3[1,1] += 1
        elif vector_rain[iter_v-3] == 0 and vector_rain[iter_v] == 1:
            matrix_freq3[0,1] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v] == 0:
            matrix_freq3[1,0] += 1
        elif vector_rain[iter_v-3] == 1 and vector_rain[iter_v] == 1:
            matrix_freq3[0,0] += 1
    
    trans_prob = np.zeros((2,2))
    trans_prob2 = np.zeros((2,2))
    trans_prob3 = np.zeros((2,2))

    for iter_mat in range(len(matrix_freq)):
        trans_prob[iter_mat, 0] = matrix_freq[iter_mat, 0] / (matrix_freq[iter_mat, 0] + matrix_freq[iter_mat, 1])
        trans_prob[iter_mat, 1] = matrix_freq[iter_mat, 1] / (matrix_freq[iter_mat, 0] + matrix_freq[iter_mat, 1])

        trans_prob2[iter_mat, 0] = matrix_freq2[iter_mat, 0] / (matrix_freq2[iter_mat, 0] + matrix_freq2[iter_mat, 1])
        trans_prob2[iter_mat, 1] = matrix_freq2[iter_mat, 1] / (matrix_freq2[iter_mat, 0] + matrix_freq2[iter_mat, 1])

        trans_prob3[iter_mat, 0] = matrix_freq3[iter_mat, 0] / (matrix_freq3[iter_mat, 0] + matrix_freq3[iter_mat, 1])
        trans_prob3[iter_mat, 1] = matrix_freq3[iter_mat, 1] / (matrix_freq3[iter_mat, 0] + matrix_freq3[iter_mat, 1])
    
    map_emit['transition_value'][wmo2kota[single_keys]] = {}
    map_emit['transition_value'][wmo2kota[single_keys]]['+1'] = trans_prob
    map_emit['transition_value'][wmo2kota[single_keys]]['+2'] = trans_prob2
    map_emit['transition_value'][wmo2kota[single_keys]]['+3'] = trans_prob3



    #get emit probability
    #mean = single_wmo_data['TEMPERATURE AVG C'].mean()
    median_temp = single_wmo_data['TEMPERATURE AVG C'].median()
    median_rh = single_wmo_data['REL HUMIDITY AVG PC'].median()
    median_wspd = single_wmo_data['WIND SPEED 24H MAX MS'].median()
    median_sun = single_wmo_data['SUNSHINE 24H H'].median()

    map_emit['obs_median'] = [median_temp, median_rh, median_wspd, median_sun]

    #order temp, rh, wspd, sun
    mat_freq_emit_yes = np.zeros((len_historic*num_param,2))
    mat_freq_emit_no = np.zeros((len_historic*num_param,2))

    mat_freq_emit_yes2 = np.zeros((len_historic*num_param,2))
    mat_freq_emit_no2 = np.zeros((len_historic*num_param,2))

    mat_freq_emit_yes3 = np.zeros((len_historic*num_param,2))
    mat_freq_emit_no3 = np.zeros((len_historic*num_param,2))

    single_arr_ori = np.array(single_wmo_data[['TEMPERATURE AVG C', 'REL HUMIDITY AVG PC', 'WIND SPEED 24H MAX MS', 'SUNSHINE 24H H']])
    for iter_v in range(len(vector_rain)-len_historic-3):
        #print('cek shape', np.shape(np.reshape(vector_rain[iter_v:iter_v+len_historic+1], (-1, 1))))
        #single_arr = np.array(single_wmo_data[['TEMPERATURE AVG C', 'REL HUMIDITY AVG PC', 
        #                              'WIND SPEED 24H MAX MS', 'SUNSHINE 24H H']].loc[iter_v:iter_v+len_historic])
        
        single_arr_used = single_arr_ori[iter_v:iter_v+len_historic+3]
        #print('cek single arr', single_arr)
        if len(single_arr_used) < len_historic:
            continue

        #print('check shape', np.shape(single_arr_used), np.shape(vector_rain[iter_v:iter_v+len_historic+1]))
        single_arr_used = np.column_stack((single_arr_used, vector_rain[iter_v:iter_v+len_historic+3]))

        single_label = single_arr_used[-3, -1]
        single_label2 = single_arr_used[-2, -1]
        single_label3 = single_arr_used[-1, -1]

        #print('cek len', len(single_arr), np.shape(mat_freq_emit_yes), np.shape(mat_freq_emit_no))
        if single_label > 0:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_yes[iter_single_arr * len_historic + 4, 1] += 1
        else:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_no[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_no[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_no[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_no[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_no[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_no[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_no[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_no[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_no[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_no[iter_single_arr * len_historic + 4, 1] += 1
        

        if single_label2 > 0:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_yes2[iter_single_arr * len_historic + 4, 1] += 1
        else:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_no2[iter_single_arr * len_historic + 4, 1] += 1
        

        if single_label3 > 0:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_yes3[iter_single_arr * len_historic + 4, 1] += 1
        else:
            for iter_single_arr in range(len_historic):
                #print('cek iter', iter_single_arr)
                if single_arr_used[iter_single_arr, 0] > median_temp:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 0, 0] += 1
                else:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 0, 1] += 1
                
                if single_arr_used[iter_single_arr, 1] > median_rh:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 1, 0] += 1
                else:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 1, 1] += 1
                
                if single_arr_used[iter_single_arr, 2] > median_wspd:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 2, 0] += 1
                else:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 2, 1] += 1
                
                if single_arr_used[iter_single_arr, 3] > median_sun:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 3, 0] += 1
                else:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 3, 1] += 1
                
                if single_arr_used[iter_single_arr, 4] > 0:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 4, 0] += 1
                else:
                    mat_freq_emit_no3[iter_single_arr * len_historic + 4, 1] += 1

    mat_emit_yes = mat_freq_emit_yes.T / np.sum(mat_freq_emit_yes, axis=1)
    mat_emit_yes = mat_emit_yes.T

    mat_emit_no = mat_freq_emit_no.T / np.sum(mat_freq_emit_no, axis=1)
    mat_emit_no = mat_emit_no.T

    mat_emit_yes2 = mat_freq_emit_yes2.T / np.sum(mat_freq_emit_yes2, axis=1)
    mat_emit_yes2 = mat_emit_yes2.T

    mat_emit_no2 = mat_freq_emit_no2.T / np.sum(mat_freq_emit_no2, axis=1)
    mat_emit_no2 = mat_emit_no2.T

    mat_emit_yes3 = mat_freq_emit_yes3.T / np.sum(mat_freq_emit_yes3, axis=1)
    mat_emit_yes3 = mat_emit_yes3.T

    print('cek component', mat_freq_emit_no3.T, np.shape(mat_freq_emit_no3.T))
    mat_emit_no3 = mat_freq_emit_no3.T / np.sum(mat_freq_emit_no3, axis=1)
    mat_emit_no3 = mat_emit_no3.T

    #print('cek yes', mat_freq_emit_yes)
    #print('cek no', mat_freq_emit_no)

    map_emit['emit_label'][wmo2kota[single_keys]] = {}
    map_emit['emit_label'][wmo2kota[single_keys]]['+1'] = {}
    map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'] = mat_emit_yes
    map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'] = mat_emit_no

    map_emit['emit_label'][wmo2kota[single_keys]]['+2'] = {}
    map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'] = mat_emit_yes2
    map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'] = mat_emit_no2

    map_emit['emit_label'][wmo2kota[single_keys]]['+3'] = {}
    map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'] = mat_emit_yes3
    map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'] = mat_emit_no3



#with open('brain.json', 'w') as fp:
#    json.dump(map_emit, fp)

with open('new_brain%s.pkl'%(len_historic), 'wb') as fp:
    pickle.dump(map_emit, fp, protocol=pickle.HIGHEST_PROTOCOL)