from flask import Flask, render_template, send_from_directory
from datetime import datetime, timedelta, date
import xml.etree.ElementTree as ET
import requests
import numpy as np
import pandas as pd

#setting
web_cuaca = 'https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml'

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

xml_dict = parse_xml_to_dict_from_web(web_cuaca)

area_fct_raw = xml_dict['data']['forecast']['area']
df_dict = {}
namas = []
lats = []
lons = []
for area_iter in range(len(area_fct_raw)):
    single_area = area_fct_raw[area_iter]
    single_lat = float(single_area['@latitude'])
    single_lon = float(single_area['@longitude'])
    single_nama = single_area['@description']

    namas.append(single_nama)
    lats.append(single_lat)
    lons.append(single_lon)
    print(single_lat, single_lon)

df_dict['nama'] = namas
df_dict['lat'] = lats
df_dict['lon'] = lons

df = pd.DataFrame(df_dict)
df.to_csv('lokasi.csv')