from flask import Flask, render_template, send_from_directory
from datetime import datetime, timedelta, date
import xml.etree.ElementTree as ET
import requests
import numpy as np
import pandas as pd

#setting
web_cuaca = 'https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml'
weather_conditions = {
    "0": "Cerah / Clear Skies",
    "1": "Cerah Berawan / Partly Cloudy",
    "2": "Cerah Berawan / Partly Cloudy",
    "3": "Berawan / Mostly Cloudy",
    "4": "Berawan Tebal / Overcast",
    "5": "Udara Kabur / Haze",
    "10": "Asap / Smoke",
    "45": "Kabut / Fog",
    "60": "Hujan Ringan / Light Rain",
    "61": "Hujan Sedang / Rain",
    "63": "Hujan Lebat / Heavy Rain",
    "80": "Hujan Lokal / Isolated Shower",
    "95": "Hujan Petir / Severe Thunderstorm",
    "97": "Hujan Petir / Severe Thunderstorm"
}
kota = [
    "Kota Banda Aceh",
    "Kota Denpasar",
    "Kota Serang",
    "Kota Bengkulu",
    "Kota Yogyakarta",
    "Kota Adm. Jakarta Pusat",
    "Gorontalo",
    "Kota Jambi",
    "Kota Bandung",
    "Kota Semarang",
    "Kota Surabaya",
    "Kota Pontianak",
    "Kota Banjarmasin",
    "Kota Palangkaraya",
    "Kota Samarinda",
    "Kota Tarakan",
    "Kota Pangkal Pinang",
    "Kota Tanjung Pinang",
    "Kota Bandar Lampung",
    "Kota Ambon",
    "Kota Ternate",
    "Kota Mataram",
    "Kota Kupang",
    "Kota Jayapura",
    "Kota Pekanbaru",
    "Mamuju",
    "Kota Makassar",
    "Kota Kendari",
    "Kota Manado",
    "Kota Padang",
    "Kota Palembang",
    "Kota Medan"
]

dt_flag = datetime(1999, 1, 1, 0, 0)

app = Flask(__name__)

def xml_to_dict(element):
    # Convert an XML element and its children into a dictionary
    def elem_to_dict(elem):
        d = {}
        if elem.text and elem.text.strip():
            d['text'] = elem.text.strip()
        
        for key, value in elem.attrib.items():
            d[f'@{key}'] = value
        
        for child in elem:
            child_dict = elem_to_dict(child)
            tag = child.tag
            
            if tag in d:
                if isinstance(d[tag], list):
                    d[tag].append(child_dict)
                else:
                    d[tag] = [d[tag], child_dict]
            else:
                d[tag] = child_dict
        
        return d

    return {element.tag: elem_to_dict(element)}

def fetch_xml_from_web(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request failed
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the XML data: {e}")
        return None

def parse_xml_to_dict_from_web(url):
    xml_data = fetch_xml_from_web(url)
    root = ET.fromstring(xml_data)
    return xml_to_dict(root)

def get_rain_web():
    xml_dict = parse_xml_to_dict_from_web(web_cuaca)
    issued_datetime_raw = xml_dict['data']['forecast']['issue']
    area_fct_raw = xml_dict['data']['forecast']['area']

    res_area_fct_arr = np.zeros((len(kota), 4))

    for area_iter in range(len(area_fct_raw)):
        single_area = area_fct_raw[area_iter]
        #get nama
        single_name_area = single_area['@description']
        
        #wx
        #get wx, list id wx:
        #hu
        #weather
        #t
        #wd
        #ws
        #tmin
        #tmax
        #humin
        #humax
        single_param_list = single_area['parameter']
        single_wx_list = []
        for iter_param in range(len(single_param_list)):
            single_param = single_param_list[iter_param]
            single_param_id = single_param['@id']
            if single_param_id != 'weather':
                continue
            single_wx_list = single_param['timerange']
            break
        
        single_wx_dt = []
        single_icon_wx = []
        for iter_wx in range(len(single_wx_list)):
            try:
                single_dt = datetime.strptime(single_wx_list[iter_wx]['@datetime'], '%Y%m%d%H00')
            except ValueError:
                print('value error:', single_wx_list[iter_wx]['@datetime'])
                single_dt = dt_flag
            single_icon_code = single_wx_list[iter_wx]['value']['text']
            try:
                single_icon = weather_conditions[single_icon_code]
                if 'Hujan' in single_icon:
                    single_hujan_state = 1
                else:
                    single_hujan_state = 0
            except KeyError:
                single_hujan_state = 0
            single_wx_dt.append(single_dt)
            single_icon_wx.append(single_hujan_state)
        
        single_wx_dt_day_list = []
        single_wx_dt_day_proper_list = []
        for iter_dt in range(len(single_wx_dt)):
            single_day_str = datetime.strftime(single_wx_dt[iter_dt], '%Y%m%d')
            single_day_proper = single_wx_dt[iter_dt]
            single_day_proper = datetime(single_day_proper.year, single_day_proper.month, single_day_proper.day, 0, 0)
            if single_day_str in single_wx_dt_day_list:
                continue
            single_wx_dt_day_list.append(single_day_str)
            single_wx_dt_day_proper_list.append(single_day_proper)

        single_wx_day_arr = np.zeros(len(single_wx_dt_day_list))
        for iter_dt in range(len(single_wx_dt)):
            single_day_str = datetime.strftime(single_wx_dt[iter_dt], '%Y%m%d')
            idx_day = single_wx_dt_day_list.index(single_day_str)
            if single_icon_wx[iter_dt] > 0:
                single_wx_day_arr[idx_day] = single_icon_wx[iter_dt]
        
        try:
            idx_fct = kota.index(single_name_area)
            res_area_fct_arr[idx_fct, :] = single_wx_day_arr
        except KeyError:
            continue
        except ValueError:
            continue
    
    print(res_area_fct_arr)
    return res_area_fct_arr, single_wx_dt_day_proper_list
    #print(issued_datetime_raw)
    #print(area_fct_raw)
    #print(xml_dict)

def get_rain_predictions():
    # Dummy rain prediction data
    today = date.today()
    return [
        {"date": today, "prediction": "Rainy"},
        {"date": today + timedelta(days=1), "prediction": "Cloudy"},
        {"date": today + timedelta(days=2), "prediction": "Sunny"},
    ]

def get_rain_prediction_real(arr, dts):
    #area_hujans = []
    res_dicts = []
    for iter_arr in range(1, len(arr[0])):
        single_v = arr[:, iter_arr]
        idxs_kota = np.argwhere(single_v > 0)
        single_hari_hujan_list = []
        for iter_idx in range(len(idxs_kota)):
            single_hari_hujan_list.append(kota[idxs_kota[iter_idx][0]])
        single_hari_hujan = ','.join(single_hari_hujan_list)
        single_dict = {}
        single_dict['date'] = dts[iter_arr]
        single_dict['prediction'] = single_hari_hujan
        res_dicts.append(single_dict)
        #area_hujans.append(single_hari_hujan)
    #print(area_hujans)
    return res_dicts

def get_rain_prediction_real2():
    df = pd.read_csv('static/stat.csv')
    nama_area = list(df['nama'])
    lat_area = np.array(df['lat'])
    lon_area = np.array(df['lon'])
    status1 = list(df['status1'])
    status2 = list(df['status2'])
    status3 = list(df['status3'])

    day1_dtstr = list(df['dt1'])[0]
    day2_dtstr = list(df['dt2'])[1]
    day3_dtstr = list(df['dt3'])[2]

    day1 = []
    day2 = []
    day3 = []
    for iter_nama in range(len(nama_area)):
        single_area = nama_area[iter_nama]
        single_lat = lat_area[iter_nama]
        single_lon = lon_area[iter_nama]

        single_status1 = status1[iter_nama]
        single_status2 = status2[iter_nama]
        single_status3 = status3[iter_nama]

        if single_status1 == 'hujan':
            day1.append(single_area)
        if single_status2 == 'hujan':
            day2.append(single_area)
        if single_status3 == 'hujan':
            day3.append(single_area)
    
    res_dicts = []
    single_dict = {}
    single_dict['date'] = day1_dtstr
    single_dict['prediction'] = ','.join(day1)
    res_dicts.append(single_dict)

    single_dict = {}
    single_dict['date'] = day2_dtstr
    single_dict['prediction'] = ','.join(day2)
    res_dicts.append(single_dict)

    single_dict = {}
    single_dict['date'] = day3_dtstr
    single_dict['prediction'] = ','.join(day3)
    res_dicts.append(single_dict)
    return res_dicts

@app.route('/')
def index():
    #predictions = get_rain_predictions()
    #fct_web, dt_fct = get_rain_web()
    #predictions = get_rain_prediction_real(fct_web, dt_fct)
    predictions = get_rain_prediction_real2()
    print('cek pred', predictions)
    return render_template('index.html', predictions=predictions)

@app.route('/display')
def display():
    #predictions = get_rain_predictions()
    return render_template('display_map.html')

@app.route('/overlay/<path:filename>')
def serve_image(filename):
    image_directory = 'overlay'
    return send_from_directory(image_directory, filename)

if __name__ == '__main__':
    app.run(debug=True)