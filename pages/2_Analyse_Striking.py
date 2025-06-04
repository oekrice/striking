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

import random
import string
from datetime import datetime

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
    for uploaded_file in st.session_state.uploaded_files:
        bytes_data = uploaded_file.read()
        if uploaded_file.name[-4:] != '.csv':
            st.error('%s Not in the correct format' % uploaded_file.name)
            st.session_state.uploaded_files.pop(st.session_state.uploaded_files.index(uploaded_file))
        else:
            isfine = True
            uploaded_file.name, _ = os.path.splitext(uploaded_file.name)
            uploaded_file.name = re.sub(r'[^\w\-]', '_', uploaded_file.name)
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
                st.session_state.uploaded_files.pop(st.session_state.uploaded_files.index(uploaded_file))
                isfine = False
            
            if isfine:
                if strike_data not in st.session_state.cached_data:
                    st.session_state.cached_data.append(strike_data)
                    st.session_state.cached_strikes.append([])
                    st.session_state.cached_certs.append([])
                    st.session_state.cached_rawdata.append(raw_data)

                    st.session_state.cached_touch_id.append(''.join(random.choices(string.ascii_letters + string.digits, k=10)))
                    st.session_state.cached_read_id.append(uploaded_file.name)
                    st.session_state.cached_nchanges.append('')
                    st.session_state.cached_methods.append('')
                    st.session_state.cached_tower.append('')
                    st.session_state.cached_datetime.append(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    #st.write(strike_data)
        os.system('rm -r ./tmp/%s' % uploaded_file.name)
           
    if len(st.session_state.uploaded_files) > 0:
        st.session_state.uploader_key += 1
        st.session_state.current_touch = -1
        st.rerun()
    return
    
if not os.path.exists('./tmp/'):
    os.system('mkdir tmp')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir frequency_data')
if not os.path.exists('./striking_data/'):
    os.system('mkdir striking_data')
if not os.path.exists('./saved_touches/'):
    os.system('mkdir saved_touches')

def determine_collection_from_url(existing_names):
    if 'collection' in st.query_params.keys():
        collection_name = st.query_params["collection"]
        if collection_name in existing_names:
            return collection_name
        else:
            return None
    else:
        return None

def find_existing_names():
    #Finds a list of existing collection names
    return os.listdir('./saved_touches/')


st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("## Analyse Striking")

st.write(
    """
    This page is for analysing the striking from the strike times either generated with the [Analyse Recording](https://brenda.oekrice.com/Analyse_Recording) page, from the [Touch Library](https://brenda.oekrice.com/Touch_Library) or from an uploaded .csv file. \\
    The 'ideal' times will be calculated and various statistics shown. If you have any suggestions for anything else you'd like to see, please let me know.
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
    st.session_state.current_touch = -1   #Positive if do want to use existing frequencies. Negative if not.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "rhythm_variation_time" not in st.session_state:
    st.session_state.rhythm_variation_time = 4
if "handstroke_gap_variation_time" not in st.session_state:
    st.session_state.handstroke_gap_variation_time = 6
if 'current_touch' not in st.session_state:
    st.session_state.current_touch = 0
if 'cached_touch_id' not in st.session_state:
    st.session_state.cached_touch_id = []
if 'cached_read_id' not in st.session_state:
    st.session_state.cached_read_id = []
if 'cached_nchanges' not in st.session_state:
    st.session_state.cached_nchanges = []
if 'cached_methods' not in st.session_state:
    st.session_state.cached_methods = []
if 'cached_tower' not in st.session_state:
    st.session_state.cached_tower = []
if 'cached_datetime' not in st.session_state:
    st.session_state.cached_datetime = []
if 'collection_status' not in st.session_state:
    st.session_state.collection_status = -1   #-1 for nothing, 0 for opening an existing one and 1 for creating a new one

def add_collection_to_cache(ntouches, saved_index_list):
    if ntouches!= 0 and saved_index_list[0][0] != ' ':
        #Put everything from this collection into the cache (if it's not already there)
        for touch_info in saved_index_list:
            #Check if it's already in there
            if touch_info[0] not in st.session_state.cached_touch_id:
                #Load in csv
                raw_data = pd.read_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,touch_info[0]))

                st.session_state.cached_data.append([touch_info[4], int(len(raw_data)/np.max(raw_data["Bell No"])), touch_info[1]])
                st.session_state.cached_strikes.append([])
                st.session_state.cached_certs.append([])
                st.session_state.cached_rawdata.append(raw_data)

                st.session_state.cached_touch_id.append(touch_info[0])
                st.session_state.cached_read_id.append(touch_info[1])
                st.session_state.cached_nchanges.append(touch_info[2])
                st.session_state.cached_methods.append(touch_info[3])
                st.session_state.cached_tower.append(touch_info[4])
                st.session_state.cached_datetime.append(touch_info[5])
            else:
                cache_index = st.session_state.cached_touch_id.index(touch_info[0])
                raw_data = pd.read_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,touch_info[0]))

                st.session_state.cached_data[cache_index] = [touch_info[4], int(len(raw_data)/np.max(raw_data["Bell No"])), touch_info[1]]
                st.session_state.cached_strikes[cache_index] = []
                st.session_state.cached_certs[cache_index] = []
                st.session_state.cached_rawdata[cache_index] = raw_data

                st.session_state.cached_touch_id[cache_index] = touch_info[0]
                st.session_state.cached_read_id[cache_index] = touch_info[1]
                st.session_state.cached_nchanges[cache_index] = touch_info[2]
                st.session_state.cached_methods[cache_index] = touch_info[3]
                st.session_state.cached_tower[cache_index] = touch_info[4]
                st.session_state.cached_datetime[cache_index] = touch_info[5]

st.session_state.existing_names = find_existing_names()
st.session_state.url_collection = determine_collection_from_url(st.session_state.existing_names)

if st.session_state.url_collection is not None:
    st.session_state.collection_status = 0
    st.session_state.current_collection_name = st.session_state.url_collection

    if os.path.getsize("./saved_touches/%s/index.csv" % st.session_state.current_collection_name) > 0:
        st.session_state.saved_index_list = np.loadtxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, delimiter = ';', dtype = str)
    else:
        st.session_state.saved_index_list = np.array([[' ',' ',' ',' ',' ',' ']], dtype = 'str')
    if len(np.shape(st.session_state.saved_index_list)) == 1:
        st.session_state.saved_index_list = np.array([st.session_state.saved_index_list])

    st.session_state.ntouches = len(st.session_state.saved_index_list)

    add_collection_to_cache(st.session_state.ntouches, st.session_state.saved_index_list)

#Remove the large things from memory -- need a condition on this maybe?
# st.session_state.trimmed_signal = None
# st.session_state.audio_signal = None
# st.session_state.raw_file = None
st.session_state.Paras = None
st.session_state.Data = None

st.session_state.touch_titles = []
st.session_state.raw_titles = []
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    st.session_state.title = st.session_state.cached_read_id[i]
    st.session_state.touch_titles.append(st.session_state.title)
    st.session_state.raw_titles.append(st.session_state.cached_data[i][2])

if len(st.session_state.touch_titles) > 0:
    st.session_state.selection = st.pills("Choose a touch to analyse:", st.session_state.touch_titles, default = st.session_state.touch_titles[-1])

    with st.expander("Upload more touches from device"):
        st.session_state.uploaded_files = st.file_uploader(
            "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
else:
    st.session_state.uploaded_files = st.file_uploader(
        "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
    
dealwith_upload()

if len(st.session_state.touch_titles) == 0:
    st.write('No data currently loaded: either upload a .csv file with striking data or generate some using the Analyse Recording page')
    st.stop()


if st.session_state.selection is None:
    st.stop()

st.session_state.current_touch = st.session_state.touch_titles.index(st.session_state.selection)
st.session_state.selected_title = st.session_state.selection

if len(st.session_state.touch_titles) > 0 and st.session_state.current_touch < 0:
      st.session_state.current_touch = len(st.session_state.touch_titles) - 1  

if st.session_state.current_touch < 0:
    st.write('**Select a touch from the options above, or upload a new one**')
else:
    st.write('Analysing ringing from "%s"' % st.session_state.touch_titles[st.session_state.current_touch])

if len(st.session_state.touch_titles) == 0:
    st.session_state.current_touch = -1

#Add a bit to easily add a touch to this collection

if st.session_state.collection_status == 0:
    if os.path.getsize("./saved_touches/%s/index.csv" % st.session_state.current_collection_name) > 0:
        saved_index_list = np.loadtxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, delimiter = ';', dtype = str)
    else:
        saved_index_list = np.array([[' ',' ',' ',' ',' ',' ']], dtype = 'str')
    if len(np.shape(saved_index_list)) == 1:
        saved_index_list = np.array([saved_index_list])

    if st.session_state.cached_touch_id[st.session_state.current_touch] not in saved_index_list[:,0]:
        #Load in csv
        if st.button("Add this touch to collection **%s**" % st.session_state.current_collection_name):
            current_ids = [listitem[0] for listitem in saved_index_list]
            if saved_index_list[0][0] != ' ':
                new_index_list = saved_index_list.tolist()
            else:
                new_index_list = []
            new_list_entry = [st.session_state.cached_touch_id[st.session_state.current_touch], st.session_state.cached_read_id[st.session_state.current_touch], st.session_state.cached_nchanges[st.session_state.current_touch], st.session_state.cached_methods[st.session_state.current_touch],st.session_state.cached_tower[st.session_state.current_touch], st.session_state.cached_datetime[st.session_state.current_touch]]
            new_index_list.append(new_list_entry)

            @st.cache_data(ttl=300)
            def convert_for_download(df):
                return df.to_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,st.session_state.cached_touch_id[st.session_state.current_touch]))
            convert_for_download(st.session_state.cached_rawdata[st.session_state.current_touch])
            np.savetxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, np.array(new_index_list, dtype = str), fmt = '%s', delimiter = ';')
            st.rerun()
    else:
        st.write('This touch is saved in collection **%s**' % st.session_state.current_collection_name)

if st.session_state.current_touch >= 0:
    #Write in to a local bit to actually do the analysis
    st.session_state.strikes = st.session_state.cached_strikes[st.session_state.current_touch]
    st.session_state.certs = st.session_state.cached_certs[st.session_state.current_touch]
    
    st.session_state.available_models = []
    #If data is uploaded, treat it slightly differently to otherwise. Can just output various things immediately without calculation
    if len(st.session_state.strikes) == 0:
        #This is from a .csv
        st.session_state.raw_data = st.session_state.cached_rawdata[st.session_state.current_touch]
        st.session_state.cols = st.session_state.raw_data.columns.tolist()

        st.session_state.existing_models = [val for val in st.session_state.cols if val not in ["Bell No", "Confidence", "Actual Time"]]
        st.session_state.existing_models = [val for val in st.session_state.existing_models if val[:7] != "Unnamed"]
        st.session_state.existing_models = [val for val in st.session_state.existing_models if val[:17] != "Corrected Bells"]

        st.session_state.raw_actuals = st.session_state.raw_data["Actual Time"]
        
        st.session_state.nbells = np.max(st.session_state.raw_data["Bell No"])
        
        if "Confidence" not in  st.session_state.raw_data.columns.tolist():
            #st.write(len(raw_data['Actual Time'])//nbells)
            st.session_state.raw_data['Confidence'] = np.ones(len(st.session_state.raw_data["Actual Time"]))

    else:        
        #I think it would be easiest to just plonk this into a dataframe and treat it like an imported one, given I've already written code for that
        st.session_state.allbells = []
        st.session_state.allcerts_save = []
        st.session_state.allstrikes = []
        st.session_state.yvalues = np.arange(len(st.session_state.cached_strikes[st.session_state.current_touch][:,0])) + 1
        st.session_state.orders = []
        for row in range(len(st.session_state.cached_strikes[st.session_state.current_touch][0])):
            st.session_state.order = np.array([val for _, val in sorted(zip(st.session_state.cached_strikes[st.session_state.current_touch][:,row], st.session_state.yvalues), reverse = False)])
            st.session_state.certs = np.array([val for _, val in sorted(zip(st.session_state.cached_strikes[st.session_state.current_touch][:,row], st.session_state.cached_certs[st.session_state.current_touch][:,row]), reverse = False)])
            st.session_state.allstrikes = st.session_state.allstrikes + sorted((st.session_state.cached_strikes[st.session_state.current_touch][:,row]).tolist())
            st.session_state.allcerts_save = st.session_state.allcerts_save + st.session_state.certs.tolist()
            st.session_state.allbells = st.session_state.allbells + st.session_state.order.tolist()
            st.session_state.orders.append(st.session_state.order)

        st.session_state.allstrikes = 1000*np.array(st.session_state.allstrikes)*0.01
        st.session_state.allbells = np.array(st.session_state.allbells)
        st.session_state.allcerts_save = np.array(st.session_state.allcerts_save)
        st.session_state.orders = np.array(st.session_state.orders)

        st.session_state.raw_data = pd.DataFrame({'Bell No': st.session_state.allbells, 'Actual Time': st.session_state.allstrikes, 'Confidence': st.session_state.allcerts_save})

        st.session_state.raw_actuals = st.session_state.raw_data["Actual Time"]
        st.session_state.nbells = np.max(st.session_state.raw_data["Bell No"])

        st.session_state.existing_models = []
        
        #st.write(len(raw_data)//nbells)
    
    class striking_data():
        #Empty class with the raw striking data and errors etc.
        def __init__(self):
            return

    st.method_message = st.empty()
    st.method_message.write("Figuring out methods and composition...")

    st.session_state.methods, st.session_state.hunt_types, st.session_state.calls, st.session_state.start_row, st.session_state.end_row, st.session_state.allrows_correct, st.session_state.quality = find_method_things(st.session_state.raw_data["Bell No"])

    if len(st.session_state.methods) > 0:
        st.session_state.call_string, st.session_state.comp_html = print_composition(st.session_state.methods, st.session_state.hunt_types, st.session_state.calls, st.session_state.allrows_correct)
        st.session_state.method_flag = True
    else:
        st.session_state.method_flag = False

    st.session_state.composition_flag = False
    if len(st.session_state.methods) > 0:
        st.session_state.nchanges = len(st.session_state.allrows_correct) - 1
        st.session_state.end_row = int(np.ceil((st.session_state.start_row + len(st.session_state.allrows_correct))/2)*2)
        if st.session_state.quality > 0.7:
            st.session_state.composition_flag = True
            if len(st.session_state.methods) == 1:   #Single method
                st.session_state.method_title = st.session_state.methods[0][0]
                if st.session_state.method_title.rsplit(' ')[0] == "Stedman" or st.session_state.method_title.rsplit(' ')[0] == "Erin":
                    st.session_state.method_title = st.session_state.method_title.rsplit(' ')[0] + " " + st.session_state.method_title.rsplit(' ')[1]
                    st.session_state.lead_length = 12
                else:
                    st.session_state.lead_length = 4*int(st.session_state.hunt_types[0][1] + 1)

            else:   #Spliced methods
                st.session_state.stages = [name.rsplit(' ',1)[-1] for  name in np.array(st.session_state.methods)[:,0]]
                st.session_state.stagetypes = [name.rsplit(' ')[-2] + ' ' + name.rsplit(' ')[-1] for  name in np.array(st.session_state.methods)[:,0]]
                if len(set(st.session_state.stagetypes)) == 1:
                    st.session_state.method_title = "Spliced " + st.session_state.stagetypes[0]
                elif len(set(st.session_state.stages)) == 1:
                    st.session_state.method_title = "Spliced " + st.session_state.stages[0]
                else:
                    st.session_state.method_title = "Spliced"
                st.session_state.lead_length = 2*int(st.session_state.hunt_types[0][1] + 1)
            if st.session_state.quality > 0.85:
                st.method_message.write("**Method(s) detected: " + str(st.session_state.nchanges) + " " + st.session_state.method_title + "**")
            else:
                st.method_message.write("**Method(s) detected: " + str(st.session_state.nchanges) + " " + st.session_state.method_title + " (sort of)**")

            st.session_state.cached_methods[st.session_state.current_touch] = st.session_state.method_title
            st.session_state.cached_nchanges[st.session_state.current_touch] = st.session_state.nchanges

            if st.session_state.nchanges/int(len(st.session_state.raw_actuals)/st.session_state.nbells) < 0.25:
                st.session_state.start_row = 0; st.session_state.end_row = int(len(st.session_state.raw_actuals)/st.session_state.nbells)

        else:
            st.method_message.write("**Probably a method but not entirely sure what...**")
            st.session_state.cached_methods[st.session_state.current_touch] = "Unknown Method"
            st.session_state.cached_nchanges[st.session_state.current_touch] = int(len(st.session_state.raw_actuals)/st.session_state.nbells)
            st.session_state.method_flag = False
            st.session_state.lead_length = 24
            st.session_state.start_row = 0; st.session_state.end_row = int(len(st.session_state.raw_actuals)/st.session_state.nbells)
    else:
        st.method_message.write("**No method detected**")
        st.session_state.cached_methods[st.session_state.current_touch] = "Rounds and/or Calls"
        st.session_state.cached_nchanges[st.session_state.current_touch] = int(len(st.session_state.raw_actuals)/st.session_state.nbells)
        st.session_state.start_row = 0; st.session_state.end_row = int(len(st.session_state.raw_actuals)/st.session_state.nbells)
        st.session_state.lead_length = 24

    if "Team Model" not in  st.session_state.raw_data.columns.tolist():
        st.session_state.count_test = st.session_state.rhythm_variation_time*st.session_state.nbells; st.session_state.gap_test = st.session_state.handstroke_gap_variation_time
        st.session_state.ideal_times = find_ideal_times_band(st.session_state.raw_data['Actual Time'], st.session_state.nbells, ncount = st.session_state.count_test, ngaps = st.session_state.gap_test, key = st.session_state.current_touch)
        st.session_state.raw_data['Team Model'] = st.session_state.ideal_times
        st.session_state.existing_models.append('Team Model')
    
    st.session_state.ideal_times = find_ideal_times_rwp(st.session_state.raw_data['Actual Time'], st.session_state.nbells, key = st.session_state.current_touch)
    if "RWP Model" not in  st.session_state.raw_data.columns.tolist():
        #st.write(len(raw_data['Actual Time'])//nbells)
        st.session_state.ideal_times = find_ideal_times_rwp(st.session_state.raw_data['Actual Time'], st.session_state.nbells, key = st.session_state.current_touch)
        st.session_state.raw_data["RWP Model"] = st.session_state.ideal_times
        st.session_state.existing_models.append("RWP Model")

    if "Corrected Bells" not in  st.session_state.raw_data.columns.tolist():
        #Add the correct order the bells should be in, based on the method analysis
        st.session_state.corrected_order = st.session_state.raw_data["Bell No"].values.copy()  #The original order
        st.session_state.start_index = st.session_state.start_row*st.session_state.nbells
        st.session_state.corrected_order[st.session_state.start_index:st.session_state.start_index + np.size(st.session_state.allrows_correct)] = np.ravel(st.session_state.allrows_correct)
        st.session_state.raw_data["Corrected Bells"] = st.session_state.corrected_order
        
    if "Metronomic Model" not in  st.session_state.raw_data.columns.tolist():
        @st.cache_data(ttl=300)
        def find_metronomic(raw_data):
            nrows = int(len(raw_data['Actual Time'])//st.session_state.nbells)
            all_metros = []
            for row in range(nrows):
                actual = np.array(st.session_state.raw_actuals[row*st.session_state.nbells:(row+1)*st.session_state.nbells])
                start = np.min(actual)
                end = np.max(actual)
                metronomic_target = np.linspace(start, end, st.session_state.nbells)
                all_metros = all_metros + metronomic_target.tolist()

            return all_metros

        all_metros = find_metronomic(st.session_state.raw_data)
        if len(all_metros) == len(st.session_state.raw_data["Actual Time"]):  #Bodge for a bug.
            st.session_state.raw_data['Metronomic Model'] = all_metros
            st.session_state.existing_models.append('Metronomic Model')
    
    if "Individual Model" not in  st.session_state.raw_data.columns.tolist():
        st.session_state.count_test = st.session_state.rhythm_variation_time*st.session_state.nbells; st.session_state.gap_test = st.session_state.handstroke_gap_variation_time
        st.session_state.ideal_times = find_ideal_times(st.session_state.raw_data['Actual Time'], st.session_state.nbells, ncount = st.session_state.count_test, ngaps = st.session_state.gap_test, key = st.session_state.current_touch)
        st.session_state.raw_data['Individual Model'] = st.session_state.ideal_times
        st.session_state.existing_models.append('Individual Model')

    if len(st.session_state.existing_models) > 0:
                
        with st.expander("Change Statistical Options"):

            st.session_state.selection = st.selectbox("Select striking model:", options = st.session_state.existing_models, index = st.session_state.existing_models.index("Team Model"))   #Can set default for this later?
            #st.write(raw_data["Actual Time"][0:100:12])
            st.session_state.raw_target = np.array(st.session_state.raw_data[st.session_state.selection])
            st.session_state.raw_bells = np.array(st.session_state.raw_data["Bell No"])
            st.session_state.correct_bells = np.array(st.session_state.raw_data["Corrected Bells"])
            #Plot blue line
            st.session_state.nstrikes = len(st.session_state.raw_actuals)
            st.session_state.nrows = int(st.session_state.nstrikes//st.session_state.nbells)

            if st.session_state.selection == "Individual Model":
                st.session_state.rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 6, format = "%d changes", step = 2, key = 100 + st.session_state.current_touch)
                st.session_state.rhythm_variation_time = st.session_state.rhythm_variation_time
                st.session_state.ideal_times = find_ideal_times(st.session_state.raw_data['Actual Time'],st.session_state. nbells, ncount = st.session_state.rhythm_variation_time*st.session_state.nbells, ngaps = st.session_state.handstroke_gap_variation_time)
                if "Individual Model" not in  st.session_state.raw_data.columns.tolist():
                    st.session_state.existing_models.append('Individual Model')

                st.session_state.raw_data['Individual Model'] = st.session_state.ideal_times

            if st.session_state.selection == "Team Model":

                rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 6, format = "%d changes", step = 2, key = 200 + st.session_state.current_touch)
                st.session_state.rhythm_variation_time = rhythm_variation_time
                ideal_times = find_ideal_times_band(st.session_state.raw_data['Actual Time'], st.session_state.nbells, ncount = st.session_state.rhythm_variation_time*st.session_state.nbells, ngaps = st.session_state.handstroke_gap_variation_time)
                if "Team Model" not in  st.session_state.raw_data.columns.tolist():
                    st.session_state.existing_models.append('Team Model')
                
                st.session_state.raw_data['Team Model'] = ideal_times
                
            st.session_state.remove_mistakes = st.checkbox("Remove presumed method mistakes from the stats?", value = True)
            st.session_state.remove_confidence = st.checkbox("Remove not-confident strike times from the stats?", value = True)
            if st.session_state.method_flag:
                st.session_state.use_method_info = st.checkbox("Use presumed composition to identify correct times?", value = True)
            else:
                st.session_state.use_method_info = False
            st.session_state.min_include_change, st.session_state.max_include_change = st.slider("For the stats, include changes in range:", min_value = 0, max_value = st.session_state.nrows, value=(st.session_state.start_row+1, st.session_state.end_row), format = "%d", step = 2, key = 300 + st.session_state.current_touch)

        Strike_Data = striking_data()  #Initialise empty thing
        Strike_Data.method_flag = st.session_state.method_flag
        Strike_Data.use_method_info = st.session_state.use_method_info
        Strike_Data.remove_confidence = st.session_state.remove_confidence
        Strike_Data.remove_mistakes= st.session_state.remove_mistakes
        Strike_Data.nbells = st.session_state.nbells
        Strike_Data.raw_data = st.session_state.raw_data
        Strike_Data.raw_actuals = st.session_state.raw_actuals
        Strike_Data.raw_target = st.session_state.raw_target
        Strike_Data.raw_bells = st.session_state.raw_bells
        Strike_Data.correct_bells = st.session_state.correct_bells
        Strike_Data.min_include_change = st.session_state.min_include_change
        Strike_Data.max_include_change = st.session_state.max_include_change

        Strike_Data = calculate_stats(Strike_Data)

        st.message = st.empty()
        st.message_2 = st.empty()
        st.message.write("Calculating stats and things...")
        #Blue Line
        st.blueline = st.empty()
        
        st.session_state.titles = ['All blows', 'Handstrokes', 'Backstrokes']

        st.session_state.overall_quality = 1.0 - np.mean(Strike_Data.alldiags[2,2,:])/Strike_Data.cadence

        k = 17.5; x0 = 0.727
        shifted_quality = 1.0/(1.0 + np.exp(-k*(st.session_state.overall_quality - x0)))
        st.message.write("Standard deviation from ideal for this touch: %.1fms" % np.mean(Strike_Data.alldiags[2,2,:]))
        st.message_2.write("Overall striking quality: **%.2f%%**" % (100*shifted_quality))

        if st.session_state.composition_flag:
            with st.expander("View Composition"):
                st.html(st.session_state.comp_html)

        with st.expander('View Plaintext Striking Report'):
            st.empty()
            obtain_striking_markdown(Strike_Data.alldiags, Strike_Data.time_errors, Strike_Data.lead_times, Strike_Data.cadence, Strike_Data.remove_mistakes)

         
        with st.expander("View Grid/Blue Line"):
            st.empty()
            st.session_state.min_plot_change, st.session_state.max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = st.session_state.nrows, value=(st.session_state.start_row, min(300, st.session_state.end_row)), format = "%d", step = 2, key = 400 + st.session_state.current_touch)
            st.session_state.view_numbers = st.checkbox("View Bell Numbers", value = False)
            st.session_state.options = ["Bell %d" % bell for bell in range(1,st.session_state.nbells+1)]
            st.session_state.highlight_bells = st.pills("Highlight Bells", st.session_state.options, selection_mode="multi")
            plot_blue_line(st.session_state.raw_target, st.session_state.raw_actuals, Strike_Data.raw_bells, st.session_state.nbells, st.session_state.lead_length, st.session_state.min_plot_change, st.session_state.max_plot_change, st.session_state.highlight_bells, view_numbers = st.session_state.view_numbers)

        #Bar Chart
        with st.expander("View Error Bar Charts"):
            st.empty()
            plot_bar_charts(Strike_Data.alldiags, Strike_Data.nbells, st.session_state.titles)
                        
        with st.expander("View Bell Errors Through Time"):
            st.empty()
            st.session_state.min_plot_change, st.session_state.max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = st.session_state.nrows, value=(0, st.session_state.nrows), format = "%d", step = 2, key = 500 + st.session_state.current_touch)
            
            st.session_state.absvalues = st.radio("Use absolute values?", ["Absolute Error", "Relative Error"])
            st.session_state.options = ["Average"] + ["Bell %d" % bell for bell in range(1,st.session_state.nbells+1)]
            st.session_state.highlight_bells = st.pills("Plot Bells", st.session_state.options, default = ["Average"], selection_mode="multi", key = 600 + st.session_state.current_touch)
            
            st.session_state.smooth = st.checkbox("Smooth data?", value = False)
                    
            strokes = ["Both Strokes", "Handstrokes", "Backstrokes"]
            if len(st.session_state.highlight_bells) == 1:
                st.session_state.strokes_plot = st.pills("Select Strokes", strokes, default = "Both Strokes", selection_mode="multi", key = 700 + st.session_state.current_touch)
            elif len(st.session_state.highlight_bells) > 1:
                st.session_state.strokes_plot = st.pills("Select Strokes", strokes, default = "Both Strokes", selection_mode="single", key = 800 + st.session_state.current_touch)
                st.session_state.strokes_plot = [st.session_state.strokes_plot]
            else:
                st.session_state.strokes_plot = None
                
            if len(st.session_state.highlight_bells) > 0 and st.session_state.strokes_plot is not None:
                    plot_errors_time(Strike_Data.time_errors, st.session_state.min_plot_change, st.session_state.max_plot_change, st.session_state.absvalues, st.session_state.highlight_bells, st.session_state.strokes_plot, st.session_state.smooth)
        
        with st.expander("View Histograms"):
            st.empty()
            st.session_state.x_range = st.slider("Histogram x range:", min_value = 50, max_value = 250, value= 160, format = "%dms")
            st.session_state.nbins_default = min(100, max(int(len(Strike_Data.errors)/2.5),10))
            st.session_state.nbins = st.slider("Number of histogram bins", min_value = 10, max_value = 100, value= st.session_state.nbins_default, format = "%d", step = 1)
            plot_histograms(Strike_Data.errors, st.session_state.x_range, st.session_state.nbins, st.session_state.nbells, Strike_Data.raw_bells, Strike_Data.correct_bells, Strike_Data.min_include_change, Strike_Data.max_include_change, Strike_Data.use_method_info, Strike_Data.remove_mistakes, Strike_Data.cadence, Strike_Data.raw_actuals, Strike_Data.raw_target, st.session_state.titles)

        with st.expander("View Box Plots"):
            st.empty()
            plot_boxes(Strike_Data.time_errors, st.session_state.nbells, st.session_state.titles)

        @st.cache_data(ttl=300)
        def convert_for_download(df):
            return df.to_csv().encode("utf-8")

        csv = convert_for_download(st.session_state.raw_data)
        if st.session_state.current_touch > -1 and len(st.session_state.raw_titles) > 0:
            st.download_button("Download analysis to device as .csv", csv, file_name = st.session_state.raw_titles[st.session_state.current_touch] + '.csv', mime="text/csv")
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
