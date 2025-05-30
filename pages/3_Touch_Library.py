'''
Copyright (C) 2025 Oliver Rice - All Rights Reserved

Permission is hereby granted to any individual to use and modify this software solely for personal, non-commercial purposes.

You May Not:

 - Distribute, sublicense, or share the software or modified versions in any form.

 - Use the software or any part of it for commercial purposes.

 - Use the software as part of a service, product, or offering to others.

This software is provided "as is", without warranty of any kind, express or implied. In no event shall the authors be liable for any claim, damages, or other liability.

If you would like to license or publish this software commerically, please contact oliverricesolar@gmail.com
'''

import streamlit as st
import numpy as np
import pandas as pd
import re

from strike_model import find_ideal_times
from strike_model_band import find_ideal_times_band
from rwp_model import find_ideal_times_rwp
from methods import find_method_things, print_composition
from data_visualisations import calculate_stats, obtain_striking_markdown, plot_errors_time, plot_bar_charts, plot_histograms, plot_boxes, plot_blue_line


from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import os

cmap = plt.cm.gnuplot2
cmap = [
    '#1f77b4',  
    '#ff7f0e', 
    '#2ca02c',  
    '#d62728', 
    '#9467bd', 
    '#8c564b', 
    '#e377c2',  
    '#7f7f7f', 
    '#bcbd22',  
    '#17becf'  
]

def dealwith_upload():
    for uploaded_file in uploaded_files:
        if uploaded_file.name[-4:] != '.csv':
            st.error('%s Not in the correct format' % uploaded_file.name)
            uploaded_files.pop(uploaded_files.index(uploaded_file))
        else:
            isfine = True
            uploaded_file.name = uploaded_file.name.replace(" ", "_")
            uploaded_file.name = uploaded_file.name.replace("'", "")
            with open('./tmp/%s' % uploaded_file.name, 'wb') as f: 
                f.write(uploaded_file.getvalue())        
            try:
                raw_data = pd.read_csv('./tmp/%s' % uploaded_file.name)
                #Convert this into rows? Nah. Do it in a bit, if at all
                #st.session_state.selected_data = raw_data
                #Present as an option on the side.
                if "Bell No" not in raw_data.columns or "Actual Time" not in raw_data.columns:
                    isfine = False
                #st.write(raw_data.columns)
                strike_data = ["Unknown Tower", int(len(raw_data)/np.max(raw_data["Bell No"])), uploaded_file.name[:-4]]
            except:
                st.error('Cannot interpret %s as readable data' % uploaded_file.name)
                uploaded_files.pop(uploaded_files.index(uploaded_file))
                isfine = False
            
            if isfine:
                if strike_data not in st.session_state.cached_data:
                    st.session_state.cached_data.append(strike_data)
                    st.session_state.cached_strikes.append([])
                    st.session_state.cached_certs.append([])
                    st.session_state.cached_rawdata.append(raw_data)
                    #st.write(strike_data)
        os.system('rm -r ./tmp/%s' % uploaded_file.name)
           
    if len(uploaded_files) > 0:
        st.session_state.uploader_key += 1
        st.session_state.current_touch = -1

        st.rerun()
    return
    
def find_existing_names():
    #Finds a list of existing collection names
    print(os.listdir('./saved_touches/'))
    return os.listdir('./saved_touches/')

def add_new_folder(name):
    print('Making new folder...')
    os.system('mkdir "saved_touches/%s"' % name)
    st.session_state.collection_status = 0
    st.session_state.current_collection_name = name
    return

if not os.path.exists('./tmp/'):
    os.system('mkdir tmp')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir frequency_data')
if not os.path.exists('./striking_data/'):
    os.system('mkdir striking_data')
if not os.path.exists('./saved_touches/'):
    os.system('mkdir saved_touches')

st.set_page_config(page_title="Touch Library", page_icon="📚")
st.markdown("## Touch Library")

st.write(
    """
    On this page you can create, organise and view your collections of touches. 
    """
)

if 'cached_strikes' not in st.session_state:
    st.session_state.cached_strikes = []  
if 'cached_certs' not in st.session_state:
    st.session_state.cached_certs = []  
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = []  
if 'cached_rawdata' not in st.session_state:
    st.session_state.cached_rawdata = [[] for val in range(len(st.session_state.cached_data))]   #This will contain a dataframe, if one exists.
if 'current_touch' not in st.session_state:
    st.session_state.current_touch = 0
if 'collection_status' not in st.session_state:
    st.session_state.collection_status = -1   #-1 for nothing, 0 for opening an existing one and 1 for creating a new one
if 'current_collection_name' not in st.session_state:
    st.session_state.current_collection_name = None   #-1 for nothing, 0 for opening an existing one and 1 for creating a new one
if 'new_collection_name' not in st.session_state:
    st.session_state.new_collection_name = None   #-1 for nothing, 0 for opening an existing one and 1 for creating a new one
#Remove the large things from memory -- need a condition on this maybe?
# st.session_state.trimmed_signal = None
# st.session_state.audio_signal = None
# st.session_state.raw_file = None
Paras = None
Data = None

touch_titles = []
raw_titles = []

#FInd the touches currently in the cache
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = '' + st.session_state.cached_data[i][2] + ''# + ': ' + str(st.session_state.cached_data[i][1]) + ' changes'
    touch_titles.append(title)
    raw_titles.append(st.session_state.cached_data[i][2])

if len(touch_titles) > 0:
    selection = st.pills("Currently loaded touches:", touch_titles, default = touch_titles[st.session_state.current_touch])
    uploaded_files = st.file_uploader(
        "Or upload more from your device:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
else:
    st.write('No touches are currently loaded. Load them from the library or upload a .csv from your device:')
    uploaded_files = st.file_uploader(
        "Upload data from device:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")

dealwith_upload()
       
st.session_state.current_touch = touch_titles.index(selection)

selected_title = selection
    
existing_names = find_existing_names()
#Open or create a method collection
st.write('Open an existing or create a new touch collection:')
            
cols = st.columns(2)
with cols[0]:
    if st.button('Open an existing collection'):
        st.session_state.collection_status = 0

with cols[1]:
    if st.button('Create a new collection'):
        st.session_state.collection_status = 1            

print(st.session_state.collection_status) 
new_collection_name = None    
#Create a new one
if st.session_state.collection_status == 1:
    st.write("Choose a name (a single word) for the new collection. If you'd like it not to be found by anyone else, choose something unguessable.")
    new_collection_name = st.text_input('New collection name (no spaces or special characters)')
    if len(new_collection_name) > 0:
        st.session_state.new_collection_name = new_collection_name
        new_collection_name = None
        st.rerun()
    os.path.splitext(st.session_state.new_collection_name)
    new_collection_name = re.sub(r'[^\w\-]', '_', st.session_state.new_collection_name)
    print(existing_names, st.session_state.new_collection_name)
    if new_collection_name not in existing_names and len(st.session_state.new_collection_name) > 0:
        st.write('This name is valid and unused. Good good.')
        st.button('Create new collection', on_click = add_new_folder(st.session_state.new_collection_name))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
