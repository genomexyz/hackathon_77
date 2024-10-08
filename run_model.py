import numpy as np
import pandas as pd
import pickle

#setting
file_dset = 'Data Fklim BMKG 2023.csv'
len_historic = 5
len_prediction = 3
wmo2kota = {}
wmo2kota[96017] = 'Kota Banda Aceh'
wmo2kota[96753] = 'Kota Bogor'
wmo2kota[96687] = 'Kota Banjarmasin'
wmo2kota[97184] = 'Kota Makassar'
wmo2kota[97692] = 'Kota Jayapura'

def predict_with_brain(brain, param, len_memory):
    median_temp = brain['obs_median'][0]
    median_rh = brain['obs_median'][1]
    median_wspd = brain['obs_median'][2]
    median_sun = brain['obs_median'][3]

    if param[-1, -1] > 0:
        trans_prob_yes = brain['transition_value'][wmo2kota[single_keys]]['+1'][0,0]
        trans_prob_no = brain['transition_value'][wmo2kota[single_keys]]['+1'][0,1]

        trans_prob_yes2 = brain['transition_value'][wmo2kota[single_keys]]['+2'][0,0]
        trans_prob_no2 = brain['transition_value'][wmo2kota[single_keys]]['+2'][0,1]

        trans_prob_yes3 = brain['transition_value'][wmo2kota[single_keys]]['+3'][0,0]
        trans_prob_no3 = brain['transition_value'][wmo2kota[single_keys]]['+3'][0,1]
    else:
        #print('cek map', map_emit['transition_value'])
        trans_prob_yes = brain['transition_value'][wmo2kota[single_keys]]['+1'][1,0]
        trans_prob_no = brain['transition_value'][wmo2kota[single_keys]]['+1'][1,1]

        trans_prob_yes2 = map_emit['transition_value'][wmo2kota[single_keys]]['+2'][1,0]
        trans_prob_no2 = brain['transition_value'][wmo2kota[single_keys]]['+2'][1,1]

        trans_prob_yes3 = brain['transition_value'][wmo2kota[single_keys]]['+3'][1,0]
        trans_prob_no3 = brain['transition_value'][wmo2kota[single_keys]]['+3'][1,1]
    
    #convert obs to binary
    v_temp_bin = np.zeros((len_memory))
    v_rh_bin = np.zeros((len_memory))
    v_wspd_bin = np.zeros((len_memory))
    v_sun_bin = np.zeros((len_memory))
    v_rain_bin = np.zeros((len_memory))

    v_temp_bin[single_arr_param[:,0] <= median_temp] = 1
    v_rh_bin[single_arr_param[:,1] <= median_rh] = 1
    v_wspd_bin[single_arr_param[:,2] <= median_wspd] = 1
    v_sun_bin[single_arr_param[:,3] <= median_sun] = 1
    v_rain_bin[single_arr_param[:,4] <= 0] = 1

    v_temp_bin = v_temp_bin.astype(int)
    v_rh_bin = v_temp_bin.astype(int)
    v_wspd_bin = v_temp_bin.astype(int)
    v_sun_bin = v_temp_bin.astype(int)
    v_rain_bin = v_temp_bin.astype(int)

    prob_yes = trans_prob_yes
    prob_no = trans_prob_no

    prob_yes2 = trans_prob_yes2
    prob_no2 = trans_prob_no2

    prob_yes3 = trans_prob_yes3
    prob_no3 = trans_prob_no3

    for iter_history in range(len_historic):
        prob_yes = prob_yes * map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['yes'][4,v_rain_bin[iter_history]]
                
        prob_no = prob_no *   map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+1']['no'][4,v_rain_bin[iter_history]]
            
        prob_yes2 = prob_yes2 * map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['yes'][4,v_rain_bin[iter_history]]
                
        prob_no2 = prob_no2 *   map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+2']['no'][4,v_rain_bin[iter_history]]
            
        prob_yes3 = prob_yes3 * map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['yes'][4,v_rain_bin[iter_history]]
                
        prob_no3 = prob_no3 *   map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'][0,v_temp_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'][1,v_rh_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'][2,v_wspd_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'][3,v_sun_bin[iter_history]] * \
                              map_emit['emit_label'][wmo2kota[single_keys]]['+3']['no'][4,v_rain_bin[iter_history]]
    
    if prob_yes == prob_no:
        prob_yes = 0.5
        prob_no = 0.5
    
    if prob_yes2 == prob_no2:
        prob_yes2 = 0.5
        prob_no2 = 0.5
    
    if prob_yes3 == prob_no3:
        prob_yes3 = 0.5
        prob_no3 = 0.5

    real_prob_yes = prob_yes / (prob_yes + prob_no)
    real_prob_yes2 = prob_yes2 / (prob_yes2 + prob_no2)
    real_prob_yes3 = prob_yes3 / (prob_yes3 + prob_no3)

    real_prob_no = prob_no / (prob_yes + prob_no)
    real_prob_no2 = prob_no2 / (prob_yes2 + prob_no2)
    real_prob_no3 = prob_no3 / (prob_yes3 + prob_no3)

    if real_prob_yes > real_prob_no:
        prediction = 1
    elif real_prob_yes < real_prob_no:
        prediction = 0
    elif real_prob_yes == real_prob_no:
        chance = np.random.uniform(0,1)
        if chance >= 0.5:
            prediction = 1
        else:
            prediction = 0
        
    if real_prob_yes2 > real_prob_no2:
        prediction2 = 1
    elif real_prob_yes2 < real_prob_no2:
        prediction2 = 0
    elif real_prob_yes2 == real_prob_no2:
        chance = np.random.uniform(0,1)
        if chance >= 0.5:
            prediction2 = 1
        else:
            prediction2 = 0
        
    if real_prob_yes3 > real_prob_no3:
        prediction3 = 1
    elif real_prob_yes3 < real_prob_no3:
        prediction3 = 0
    elif real_prob_yes3 == real_prob_no3:
        chance = np.random.uniform(0,1)
        if chance >= 0.5:
            prediction3 = 1
        else:
            prediction3 = 0
    
    return np.array([prediction, prediction2, prediction3])
    

#load model
with open('new_brain%s.pkl'%(len_historic), 'rb') as fp:
    map_emit = pickle.load(fp)

with open('new_brain%s.pkl'%(3), 'rb') as fp:
    map_emit3 = pickle.load(fp)

with open('new_brain%s.pkl'%(2), 'rb') as fp:
    map_emit2 = pickle.load(fp)

#print(map_emit)

#read data
df = pd.read_csv(file_dset)

#transform data
df['RAINFALL 24H MM'] = df['RAINFALL 24H MM'].replace(8888, 0)
df['RAINFALL 24H MM'] = df['RAINFALL 24H MM'].replace(np.nan, 0)
df['SUNSHINE 24H H'] = df['SUNSHINE 24H H'].replace(9999, 0)
df['SUNSHINE 24H H'] = df['SUNSHINE 24H H'].replace(np.nan, 0)

all_keys = list(dict.keys(wmo2kota))
for iter_keys in range(len(all_keys)):
    single_keys = all_keys[iter_keys]
    single_wmo_data = df[df['WMO ID'] == single_keys]

    vector_rain = np.array(single_wmo_data['RAINFALL 24H MM'])
    vector_rain[vector_rain > 0] = 1
    vector_rain[vector_rain <= 0] = 0

    single_arr_ori = np.array(single_wmo_data[['TEMPERATURE AVG C', 'REL HUMIDITY AVG PC', 'WIND SPEED 24H MAX MS', 'SUNSHINE 24H H']])

    mat_acc = []
    mat_acc2 = []
    mat_acc3 = []
    for iter_v in range(len(vector_rain)-len_historic-3):
        single_arr_used = single_arr_ori[iter_v:iter_v+len_historic+3]
        single_arr_used = np.column_stack((single_arr_used, vector_rain[iter_v:iter_v+len_historic+3]))

        single_arr_param = single_arr_used[:-3, :]
        single_arr_param2 = single_arr_used[-5:-3, :]
        single_arr_param3 = single_arr_used[-6:-3, :]

        single_label = single_arr_used[-3, -1]
        single_label2 = single_arr_used[-2, -1]
        single_label3 = single_arr_used[-1, -1]

        predictions = predict_with_brain(map_emit, single_arr_param, 5)
        predictions_brain2 = predict_with_brain(map_emit, single_arr_param2, 2)
        predictions_brain3 = predict_with_brain(map_emit, single_arr_param3, 3)


        
        #check accuracy
        prediction = predictions[0]
        prediction3 = predictions[1]
        prediction2 = predictions[2]

        mat_acc.append(abs(prediction-single_label))
        mat_acc2.append(abs(prediction2-single_label2))
        mat_acc3.append(abs(prediction3-single_label3))

        #print('cek prob', prob_yes, prob_no)

        #print(single_arr_used)
    
    mat_acc = np.array(mat_acc)
    mat_acc2 = np.array(mat_acc2)
    mat_acc3 = np.array(mat_acc3)

    acc = 1 - (np.sum(mat_acc) / len(mat_acc))
    acc2 = 1 - (np.sum(mat_acc2) / len(mat_acc2))
    acc3 = 1 - (np.sum(mat_acc3) / len(mat_acc3))

    print('akurasi', acc, acc2, acc3)

    #break

