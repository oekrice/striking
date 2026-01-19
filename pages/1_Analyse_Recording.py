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
import os
import gc
import io
import random
import string
from datetime import datetime
import re

from listen_classes import audio_data, parameters
from listen_main_functions import establish_initial_rhythm, do_reinforcement, find_final_strikes, save_strikes, filter_final_strikes
from rwp_model import find_ideal_times_rwp
from listen_other_functions import find_colour
from methods import find_method_things

st.set_page_config(page_title="Analyse Recording", page_icon="🎤")

st.markdown("""
    <style>
    [data-testid="stDeployButton"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("## Analyse a Recording")

st.write("This page is to find strike times from a recording of bellringing")
with st.expander('How to use this'):
    st.markdown(
        """
        1. Select the tower and bells being rung -- take care to uncheck bells if only ringing the back 6 of an 8 etc..
        2. Upload the recording of ringing. Ringing must start **within 1 minute** of the start of the (trimmed) recording, and **must begin with at least three whole pulls of approximate rounds**.   
        3. Choose whether to use existing frequency profiles (if they exist) or learn new ones.
        4. If the latter, you'll be given the option to do this. This can be quite slow, especially for more than 8 bells.
        5. Once decent frequencies are found, the code will attempt to strike times throughout. 
        6. These strike times can then be analysed on the other page, or downloaded as a .csv for use later.   \\
        $~$ \\
        If the analysis fails, this might just be doomed to failure, but you could try:
        * Trimming the recording such that it starts with good rounds (useful if it takes a while to settle, there is loud talking or a change in volume near the start).
        * Generating new frequencies from this specific recording. Things can sound different from different recording locations so this can sometimes work.
        * If learning the frequencies isn't working, try using a clearer recording to learn them and then use those profiles with the less-good recording.
        """
    )

if not os.path.exists('./tmp/'):
    os.system('mkdir tmp')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir frequency_data')
if not os.path.exists('./striking_data/'):
    os.system('mkdir striking_data')
if not os.path.exists('./saved_touches/'):
    os.system('mkdir saved_touches')

#Establish persistent variables

st.session_state.testing_mode = False
#Establish persistent variables
if st.session_state.testing_mode:
    input_matrix = np.loadtxt("test_cases.txt", delimiter = ';', dtype = str)    
    init_test = 52
    single_test = True

    if "test_counter" not in st.session_state:
        st.session_state.test_counter = init_test
    if "single_test" not in st.session_state:
        st.session_state.single_test = single_test

if "counter" not in st.session_state:
    st.session_state.counter = 0
if 'tower_selected' not in st.session_state:
    st.session_state.tower_selected = False
if 'tower_name' not in st.session_state:
    st.session_state.tower_name = None
if 'nominals_confirmed' not in st.session_state:
    st.session_state.nominals_confirmed = False
if 'bell_nominals' not in st.session_state:
    st.session_state.bell_nominals = False
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'reinforce_status' not in st.session_state:
    st.session_state.reinforce_status = 0
if 'found_new_freqs' not in st.session_state:
    st.session_state.found_new_freqs = False
if 'use_existing_freqs' not in st.session_state:
    st.session_state.use_existing_freqs = -1   #Positive if do want to use existing frequencies. Negative if not.
if 'already_saved' not in st.session_state:
    st.session_state.already_saved = False  
if 'good_frequencies_selected' not in st.session_state:
    st.session_state.good_frequencies_selected = False   #Decent frequencies are selected (either preloaded or calculated now)
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = 0   
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = 0
if 'trim_flag' not in st.session_state:
    st.session_state.trim_flag = False   
if 'checked' not in st.session_state:
    st.session_state.checked = []  
if 'raw_file' not in st.session_state:
    st.session_state.raw_file = None  
if 'collection_status' not in st.session_state:
    st.session_state.collection_status = -1   #-1 for nothing, 0 for opening an existing one and 1 for creating a new one
#Frequency data to be saved throughout
if 'reinforce_test_frequencies' not in st.session_state:
    st.session_state.reinforce_test_frequencies = None   
if 'reinforce_frequency_profile' not in st.session_state:
    st.session_state.reinforce_frequency_profile = None  
if 'reinforce_frequency_data' not in st.session_state:
    st.session_state.reinforce_frequency_data = None  
 
#Final frequency data for use
if 'final_freqs' not in st.session_state:
    st.session_state.final_freqs = None   
if 'final_freqprobs' not in st.session_state:
    st.session_state.final_freqprobs = None   
    
if 'allstrikes' not in st.session_state:
    st.session_state.allstrikes = None  
if 'allcerts' not in st.session_state:
    st.session_state.allcerts = None  
   
if 'cached_strikes' not in st.session_state:
    st.session_state.cached_strikes = [] 
if 'cached_certs' not in st.session_state:
    st.session_state.cached_certs = [] 
if 'cached_data' not in st.session_state:
    st.session_state.cached_data = []   
if 'cached_rawdata' not in st.session_state:
    st.session_state.cached_rawdata = []  
if 'incache' not in st.session_state:
    st.session_state.incache = False   
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
if 'cached_score' not in st.session_state:
    st.session_state.cached_score = []
if 'cached_ms' not in st.session_state:
    st.session_state.cached_ms = []



#Audio things to be cached -- need to be careful here
if 'audio_signal' not in st.session_state:
    st.session_state.audio_signal = None
if 'trimmed_signal' not in st.session_state:
    st.session_state.trimmed_signal = None
if 'fs' not in st.session_state:
    st.session_state.fs = None
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
    
if st.session_state.testing_mode:
    st.session_state.analysis_status = 0
    st.session_state.reinforce_status = 0
    if st.session_state.test_counter >= len(input_matrix):
        print('___________________________________________')
        print('Tests complete with no serious errors')
        st.session_state.test_counter = 0
        st.stop()
    if single_test:
        if init_test != st.session_state.test_counter:
            print('___________________________________________')
            print('Tests complete with no serious errors')
            st.stop()
    print('___________________________________________')
    print('TESTING CASE', st.session_state.test_counter)
    print('Fname:', input_matrix[st.session_state.test_counter][2])

gc.collect()

st.session_state.Audio = None
st.session_state.Data = None
st.session_state.Paras = None

def reset_nominals():
    st.session_state.nominals_confirmed = False
    st.session_state.bell_nominals = False
    st.session_state.reinforce_frequency_data = None
    st.session_state.reinforce_status = 0
    st.session_state.good_frequencies_selected = False
    st.session_state.checked = []
    st.session_state.use_existing_freqs = -1   #Positive if do want to use existing frequencies. Negative if not.
    st.session_state.analysis_status = 0 #If new file is uploaded, reset status
    #st.rerun()
    #st.session_state.isfile = False
    #st.session_state.reinforce = 0

def reset_on_upload():
    st.session_state.reinforce_frequency_data = None
    st.session_state.reinforce_status = 0
    st.session_state.analysis_status = 0 #If new file is uploaded, reset status
    st.session_state.trim_flag = False
    st.session_state.allstrikes = None  
    st.session_state.allcerts = None  
    st.session_state.audio_signal = None
    st.session_state.trimmed_signal = None
    st.session_state.raw_file = None
    
def reset_file():
    st.session_state.audio_signal = None
    st.session_state.trimmed_signal = None
    st.session_state.raw_file = None

st.session_state.counter += 1

st.session_state.progress_counter = 0   #How far through the thing is

@st.cache_data(ttl=300)               
def read_bell_data():
    nominal_import = pd.read_csv('./bell_data/nominal_data.csv')
    return nominal_import

def determine_collection_from_url(existing_names):
    if 'collection' in st.query_params.keys():
        collection_name = st.query_params["collection"]
        collection_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), collection_name)   
        if collection_name in existing_names:
            return collection_name
        else:
            return None
    else:
        return None
    
def find_existing_names():
    #Finds a list of existing collection names
    names_raw = os.listdir('./saved_touches/')
    names_lower = []
    for name in names_raw:
        lower_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), name)   
        names_lower.append(lower_name)
    return names_lower

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
                st.session_state.cached_score.append(touch_info[6])
                st.session_state.cached_ms.append(touch_info[7])
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
                st.session_state.cached_score[cache_index] = touch_info[6]
                st.session_state.cached_ms[cache_index] = touch_info[7]
                
existing_names = find_existing_names()
url_collection = determine_collection_from_url(existing_names)

def determine_url_params():
    if 'current_collection_name' in st.session_state:
        if st.session_state.current_collection_name is not None:
            st.query_params.from_dict({"collection": [st.session_state.current_collection_name]})
    return
determine_url_params()

if url_collection is not None:
    st.session_state.collection_status = 0
    st.session_state.current_collection_name = url_collection

    if os.path.getsize("./saved_touches/%s/index.csv" % st.session_state.current_collection_name) > 0:
        saved_index_list = np.loadtxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, delimiter = ';', dtype = str)
    else:
        saved_index_list = np.array([[' ',' ',' ',' ',' ',' ',' ',' ']], dtype = 'str')
    if len(np.shape(saved_index_list)) == 1:
        saved_index_list = np.array([saved_index_list])
    if len(saved_index_list[0]) < 8: #Add extra spaces
        saved_index_list = saved_index_list.tolist()
        nshort = 8 - len(saved_index_list[0])
        for li, item in enumerate(saved_index_list):
            for i in range(nshort):
                item.append(' ')
            saved_index_list[li] = item
    saved_index_list = np.array(saved_index_list)

    ntouches = len(saved_index_list)

    add_collection_to_cache(ntouches, saved_index_list)

st.session_state.nominal_data = read_bell_data()
st.session_state.freq_filename = None

st.session_state.tower_names = st.session_state.nominal_data["Tower Name"].tolist()

st.session_state.default_index = st.session_state.tower_names.index("Brancepeth, S Brandon")

if st.session_state.testing_mode:
    reset_nominals()
    reset_on_upload()
    st.session_state.tower_name = input_matrix[st.session_state.test_counter][0]
    st.session_state.default_index = st.session_state.tower_names.index(st.session_state.tower_name)

st.session_state.existing_selected = False
if st.session_state.tower_name is not None:
    if st.session_state.tower_name != "Unknown Tower":
        st.session_state.existing_selected = True
        st.session_state.index = st.session_state.tower_names.index(st.session_state.tower_name)
        #st.session_state.nominals_confirmed = True

if st.session_state.testing_mode:
    st.session_state.tower_name = st.selectbox('Select a tower...', st.session_state.tower_names, index = st.session_state.default_index, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)

else:
    if not st.session_state.existing_selected:
        st.session_state.tower_name = st.selectbox('Select a tower...', st.session_state.tower_names, index = None, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)
    else:
        st.session_state.tower_name = st.selectbox('Select a tower...', st.session_state.tower_names, index = st.session_state.index, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)

if st.session_state.tower_name:
    st.session_state.tower_selected = True 
else:
    st.session_state.tower_selected = False  
    st.session_state.nominals_confirmed = False  
    st.write("If you can't find the required tower in the list, try to find a similarly-tuned ring (look for the nominal frequencies) on Dove's guide and that may work -- but please don't save the frequency profiles.")
if st.session_state.tower_selected:
    selected_index = st.session_state.tower_names.index(st.session_state.tower_name)
    st.session_state.tower_id = st.session_state.nominal_data["Tower ID"][selected_index]

    st.session_state.bell_options = list(st.session_state.nominal_data.columns)[3:]
    #Determine the number of valid bells
    st.session_state.bell_names = []; st.session_state.bell_nominals = []
    
    st.session_state.nbells_max = 0
    for option in st.session_state.bell_options:
        if float(st.session_state.nominal_data.loc[selected_index][option]) > 0:
            st.session_state.bell_names.append(option)
            if option.isnumeric():
                st.session_state.nbells_max = max(st.session_state.nbells_max, int(option))
    
    st.write("Ring of ", str(st.session_state.nbells_max), "with", str(len(st.session_state.bell_names)), "bells to choose from:")
    
    st.session_state.nrows = len(st.session_state.bell_names)//11 + 1
    st.session_state.per_row = int((len(st.session_state.bell_names)-1e-6)//st.session_state.nrows) + 1 
    st.session_state.mincheck = len(st.session_state.bell_names)-1; st.session_state.maxcheck = 0

    st.session_state.nbells_save = 0; st.session_state.max_bell = 0

    if st.session_state.testing_mode:
        st.session_state.nominals_confirmed = False

        st.session_state.nbells_select = int(input_matrix[st.session_state.test_counter][1])
        for row in range(st.session_state.nrows):
            #Need more than one row sometimes (if more than 8?)
            st.session_state.start = row*st.session_state.per_row
            st.session_state.end = min(len(st.session_state.bell_names), (row+1)*st.session_state.per_row)
            st.session_state.cols = st.columns(st.session_state.end-st.session_state.start)
            #Display checkboxes
            for i in range(st.session_state.start, st.session_state.end):
                                
                if st.session_state.bell_names[i].isnumeric():
                    st.session_state.mincheck = min(i, st.session_state.mincheck)
                    st.session_state.maxcheck = max(i, st.session_state.maxcheck)
                    st.session_state.bell_nominals.append(float(st.session_state.nominal_data.loc[selected_index, st.session_state.bell_names[i]]))

        st.session_state.mincheck = st.session_state.maxcheck-st.session_state.nbells_select + 1
        st.session_state.max_bell = st.session_state.maxcheck + 1
        st.session_state.bell_nominals = st.session_state.bell_nominals[-st.session_state.nbells_select:]
    else:
        st.session_state.allchecked = []
        for row in range(st.session_state.nrows):
            #Need more than one row sometimes (if more than 8?)
            st.session_state.start = row*st.session_state.per_row
            st.session_state.end = min(len(st.session_state.bell_names), (row+1)*st.session_state.per_row)
            cols = st.columns(st.session_state.end-st.session_state.start)
            #Display checkboxes
            for i in range(st.session_state.start, st.session_state.end):
                with cols[i%st.session_state.per_row]:
                    if len(st.session_state.checked) > 0:
                        if st.session_state.bell_names[i] in st.session_state.checked:
                            checked = st.checkbox(st.session_state.bell_names[i], value = True, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                            if st.session_state.bell_names[i].isnumeric():
                                st.session_state.max_bell = max(st.session_state.max_bell, int(st.session_state.bell_names[i]))    
                        else:
                            checked = st.checkbox(st.session_state.bell_names[i], value = False, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                    else:
                        if st.session_state.bell_names[i].isnumeric():
                            checked = st.checkbox(st.session_state.bell_names[i], value = True, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                            if st.session_state.bell_names[i].isnumeric():
                                st.session_state.max_bell = max(st.session_state.max_bell, int(st.session_state.bell_names[i]))    
                        else:
                            checked = st.checkbox(st.session_state.bell_names[i], value = False, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                    if checked:
                        st.session_state.allchecked.append(st.session_state.bell_names[i])
                        st.session_state.mincheck = min(i, st.session_state.mincheck)
                        st.session_state.maxcheck = max(i, st.session_state.maxcheck)
                        st.session_state.bell_nominals.append(float(st.session_state.nominal_data.loc[selected_index, st.session_state.bell_names[i]]))
        st.session_state.checked = st.session_state.allchecked

    st.session_state.nbells_save = len(st.session_state.bell_nominals)
    st.session_state.bell_nominals = sorted(st.session_state.bell_nominals, reverse = True)
      
    st.write(str(len(st.session_state.bell_nominals)), 'bells selected, with (editable) nominal frequencies in Hz')
        
    #Make nice dataframe for this
    
    if st.session_state.nominals_confirmed:
        #These have been confirmed so don't repopulate unless they've been reset otherwise
        freq_df = pd.DataFrame(data = np.array(st.session_state.bell_nominals)[np.newaxis,:], columns = ["Bell %d" % (ri + 1) for ri in range(len(st.session_state.bell_nominals))])
    else:
        #These haven't been confirmed -- fill dataframe will raw data
        freq_df = pd.DataFrame(data = np.array(st.session_state.bell_nominals)[np.newaxis,:], columns = ["Bell %d" % (ri + 1) for ri in range(len(st.session_state.bell_nominals))])
        
    edited_nominals = np.array(st.data_editor(freq_df, hide_index = True, on_change = reset_nominals))[0].tolist()
    
    if len(st.session_state.bell_nominals) < 4:
        st.write('Please select four or more bells')
        st.stop()

    if st.session_state.testing_mode:
        st.session_state.nominals_confirmed = True
        st.session_state.bell_nominals = edited_nominals
    else:
        if st.button("Confirm Tower and Frequencies", disabled = st.session_state.nominals_confirmed):
            st.session_state.nominals_confirmed = True
            st.session_state.bell_nominals = edited_nominals
            st.rerun()
    
def process_audio_files(raw_file, doprints):
    #Function that needs caching to avoid the need to keep uploading and converting things
    audio_data(raw_file, doprints)
        
    return st.session_state.audio_signal, st.session_state.fs, st.session_state.audio_filename
    
if st.session_state.tower_selected and st.session_state.nominals_confirmed:
    #This should come up EVERY time after the first confirmation. Actually no.
    
    #Establish filename for the frequencies.
    #Needs to contain tower, first and last bells, and a counter. Can work on formats in a bit.
    st.session_state.freq_root = '%05d_%02d_%02d_a' % (st.session_state.tower_id, st.session_state.nbells_save, st.session_state.max_bell)
    
    #rst.write(freq_root)

    #Check for files with this handle
    existing_files = 0; allquals = []; allcs = []; max_existing = -1
    existing_freq_files = os.listdir('./frequency_data/')
    for file in existing_freq_files:
        if file[:len(st.session_state.freq_root)] == st.session_state.freq_root:
            if file[-11:] == "quality.npy":
                max_existing = max(max_existing, int(file[14:17]))   #This has been changed for the new file format
                quals = np.load('./frequency_data/' + file)
                allquals.append(quals[2])
                c = find_colour(quals[2])
                allcs.append(c)
                    
            existing_files += 1
    frequency_counter = existing_files//3    #ID of THIS frequency set
    freq_filename = st.session_state.freq_root + '_%03d' % (max_existing + 1)         
        
    def stop_analysis():
        if st.session_state.final_freqs is not None:
            st.session_state.analysis_status = 2    #Need a done condition here
        else:
            st.session_state.analysis_status = 0
        return
    
    best_freq_quality = 0.
    if len(allquals) > 0:
        best_freq_quality = np.max(allquals)

    if frequency_counter == 1:
        st.write('Found %d existing frequency profile which matches the selected bells:' % frequency_counter)
        #st.write('Choose existing profile or make a new one (can change your mind later):.')
        allstrings = ["Make new profile", ":%s[Profile 1: %.1f%% match]" % (allcs[0], 100*allquals[0])]
        if best_freq_quality > 0.97:
            options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings, on_change = stop_analysis, index = 1)
        else:
            options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings, on_change = stop_analysis, index = 0)

    elif frequency_counter > 1:
        st.write('Found %d existing frequency profiles which match the selected bells...' % frequency_counter)
        #st.write('Choose existing profile or make a new one (can change your mind later):')
        allstrings = ["Make new profile"]
        if best_freq_quality > 0.98:
            maxind = np.where(allquals == np.max(allquals))[0][0]
        else:
            maxind = -1
        for qi, qual in enumerate(allquals):
            allstrings.append(":%s[Profile %d: %.1f%% match]" % (allcs[qi], qi + 1, 100*qual))
        options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings, on_change = stop_analysis, index = int(maxind + 1))
    
    else:
        st.write('No existing frequency profiles which match these bells -- will need to create one.')
        st.write('**All frequencies learnt prior to 0.9.8 update will have to be redone for maths reasons -- apologies**')
        options = "Make new profile"
        
    if options == "Make new profile" or st.session_state.testing_mode:
        st.session_state.use_existing_freqs = -1
    else:
        st.session_state.use_existing_freqs  = allstrings.index(options) - 1
        st.session_state.existing_filename = st.session_state.freq_root + '_%03d' % (st.session_state.use_existing_freqs)  
    #Nominal frequencies detected. Proceed to upload audio...
    #st.write("Upload ringing audio:")
    st.session_state.input_option = 0
    #st.write(st.session_state.audio_signal is not None)
    if st.session_state.testing_mode:
        test_fname = input_matrix[st.session_state.test_counter][2]
        with open (test_fname, "rb") as f:
            file_bytes = f.read()
            input_file = io.BytesIO(file_bytes)
            input_file.name = test_fname
    else:
        #Decide between uploading a file and recording directly
        st.session_state.input_option = st.pills(label = 'Upload a recording or record something directly:', options = ["Upload", "Record"], default = "Upload", on_change = reset_file)
        if st.session_state.input_option != "Record":
            input_file = st.file_uploader("Upload recording of ringing for analysis", on_change = reset_on_upload, key = st.session_state.uploader_key)
            if input_file is not None:
                st.session_state.raw_file = input_file
        else:
            input_file = st.audio_input("Record ringing", on_change = reset_on_upload, key = st.session_state.uploader_key)
            if input_file is not None:
                #Name for the uploaded file... Tower plus random number?
                st.session_state.tower_short = st.session_state.tower_name.rsplit(' ')[0][:-1]
                input_file.name = ('%s_record_%04d.wav' % (st.session_state.tower_short, int(random.random()*10000)))
                if input_file is not None:
                    st.session_state.raw_file = input_file

    if input_file is not None:
        process_audio_files(input_file, doprints = True)  
        del input_file
        st.session_state.uploader_key += 1
        st.session_state.recorder_key += 1
        if not st.session_state.testing_mode:
            st.rerun()

    #st.write(raw_file is not None, st.session_state.audio_signal is not None)
    if (st.session_state.raw_file is not None) and (st.session_state.audio_signal is not None):
        st.audio(st.session_state.raw_file)
    if st.session_state.input_option == "Record" and st.session_state.raw_file is not None:
        st.download_button("Download this recording to device", st.session_state.raw_file, file_name = st.session_state.raw_file.name)

    if st.session_state.input_option == "Upload":
        if st.session_state.trimmed_signal is not None:
            st.write('File "%s" read in successfully.' % st.session_state.audio_filename)
            st.write('Trimmed recording length: %d seconds.' % (len(st.session_state.trimmed_signal)/st.session_state.fs))
        elif st.session_state.audio_signal is not None:
            #Put some prints to indicate a file has been uploaded
            st.write('File "%s" read in successfully.' % st.session_state.audio_filename)
            st.write('Imported recording length: %d seconds.' % (len(st.session_state.audio_signal)/st.session_state.fs))
    else:
        if st.session_state.trimmed_signal is not None:
            #st.write('File "%s" recorded successfully.' % st.session_state.audio_filename)
            st.write('Trimmed recording length: %d seconds.' % (len(st.session_state.trimmed_signal)/st.session_state.fs))
        elif st.session_state.audio_signal is not None:
            #Put some prints to indicate a file has been uploaded
            #st.write('File "%s" recorded successfully.' % st.session_state.audio_filename)
            st.write('Imported recording length: %d seconds.' % (len(st.session_state.audio_signal)/st.session_state.fs))
        
    if ['uploaded_file'] in st.session_state:
        del st.session_state['uploaded_file']

if st.session_state.nominals_confirmed and st.session_state.tower_selected and (st.session_state.audio_signal is not None or st.session_state.trimmed_signal is not None):
    
    #st.write(st.session_state.audio_signal is not None, st.session_state.trimmed_signal is not None)
    def change_reinforce():
        if st.session_state.reinforce_status != 1:
            st.session_state.reinforce_status = 1
        elif st.session_state.reinforce_frequency_data is not None:
            st.session_state.reinforce_status = 0
            if st.session_state.reinforce_frequency_data[2] > 0.85:
                st.session_state.reinforce_status = 2
            else:
                st.session_state.reinforce_status = 0
        else:
            st.session_state.reinforce_status = 0
        return
        
    st.write("Recording parameters:")
    st.session_state.tmax = len(st.session_state.audio_signal)/st.session_state.fs

    st.session_state.overall_tmin, st.session_state.overall_tmax = st.slider("Trim recording for use overall (remove silence before ringing if possible):", min_value = 0.0, max_value = 0.0, value=(0.0, st.session_state.tmax),step = 1. ,format = "%ds", disabled = False)
        
    if st.session_state.use_existing_freqs < 0:
        st.session_state.reinforce_tmax = st.slider("Length of recording used for frequency analysis -- longer is slower but more accurate (but don't include bad ringing!):", min_value = 45.0, max_value = min(120.0, st.session_state.tmax), step = 1., value=(60.0), format = "%ds")
        st.session_state.nreinforces = int(st.slider("Max number of frequency analysis steps -- may not need this many and will stop automatically if so:", min_value = 2, max_value = 15, value = 10, step = 1))
    else:
        st.session_state.reinforce_tmax = 90.0
        st.session_state.nreinforces = 5

    if st.session_state.use_existing_freqs < 0:
        if len(st.session_state.bell_nominals) > 8:
            st.session_state.npicks_mode = st.checkbox(label = "Run a more intense frequency analysis to improve accuracy (may be very slow but recommended on 10/12)", value = True)
        else:
            st.session_state.npicks_mode = st.checkbox(label = "Run a more intense frequency analysis to improve accuracy (may be very slow but recommended on 10/12)", value = False)
    else:
        st.session_state.npicks_mode = False

    st.session_state.Paras = parameters(np.array(st.session_state.bell_nominals), st.session_state.overall_tmin, st.session_state.overall_tmax, st.session_state.reinforce_tmax, st.session_state.nreinforces, st.session_state.npicks_mode)
    st.session_state.Paras.fname = str(st.session_state.tower_id)

    if st.session_state.use_existing_freqs < 0:
        
        if st.session_state.testing_mode:
            change_reinforce()
        else:
            if st.session_state.reinforce_status == 0 or st.session_state.reinforce_status == 2:
                st.button("Find new frequency profiles and strike times", on_click = change_reinforce)
            else: 
                st.button("Stop finding frequencies", on_click = change_reinforce)
                     
        if st.session_state.reinforce_status > 0:
            #Create placeholders for frequency detection text
            st.main_log = st.empty()
            st.quality_log = st.empty()
            st.current_log = st.empty()
    
        if st.session_state.reinforce_status == 1:
            st.session_state.trim_flag = True
            #st.session_state.audio_signal = None   #Remove original untrimmed signal as it's a waste of space
            #Begin frequency reinforcement -- stop when the flag stops being on!
            #Zero for not at all, 1 for doing it and 2 for done. Need something to fill this space if never doing reinforcement
            st.main_log.write('**Detecting initial rhythm**')
            
            if st.session_state.reinforce_frequency_data is not None and st.session_state.reinforce_frequency_data[2] > 0.005:
                toprint = st.session_state.reinforce_frequency_data[2]
                c = find_colour(toprint)
                st.quality_log.write('Best yet frequency match: :%s[%.1f%%]' % (c, 100*toprint))
            else:
                st.quality_log.write('Best yet frequency match: :%s[%.1f%%]' % ('red', 0.0))
        
            st.current_log.write('Detecting ringing...')
        
            st.session_state.Data = establish_initial_rhythm(st.session_state.Paras)
            
            st.current_log.write('Established initial rhythm using ' + str(len(st.session_state.Data.strikes[0])) + ' changes')
                    
            st.session_state.Data = do_reinforcement(st.session_state.Paras, st.session_state.Data)
                    
            if st.session_state.reinforce_frequency_data is not None:
                if st.session_state.reinforce_frequency_data[2] > 0.85:
                    st.session_state.reinforce_status = 2
                else:
                    st.main_log.write("**Frequency profiles not good enough to use... Apologies**")
                    st.session_state.reinforce_status = 0
            else:
                st.session_state.reinforce_status = 0
                        
            del st.session_state.Data   #Can probably delete some other stuff as well...
            if not st.session_state.testing_mode:
                st.rerun()
                        
        if st.session_state.reinforce_status == 2:   #At least some frequency reinforcement has happened, print out some things to this end
                             
            st.main_log.write('**Frequency analysis completed seemingly succesfully**')
            #Determine colours:
            colour_thresholds = [0.95,0.98]; colours = ['red', 'orange', 'green']
            toprint = st.session_state.reinforce_frequency_data[2]
            c = colours[0]
            if toprint > colour_thresholds[0]:
                c = colours[1]
            if toprint > colour_thresholds[1]:
                c = colours[2]
    
            st.quality_log.write('Best frequency analysis quality = :%s[%.1f%%]' % (c, 100*toprint))
            
            if toprint < 0.95:
                st.current_log.write('This might not be good enough to provide anything useful. Suggest using a nicer touch or doing more intense analysis. But may as well try if not...')
            elif toprint < 0.975:
                st.current_log.write('Not perfect but it\'ll probably do. If accuracy is important I\'d suggest finding a nicer recording to learn the frequencies, or run a more intense analysis.')
            else:
                st.current_log.write('That should be fine to detect everything reasonably well.')

            if st.session_state.analysis_status == 0 and toprint > 0.95:
                st.session_state.analysis_status = 1 #Automatically find strike times
            st.divider()
                    
#st.write(st.session_state.reinforce_status, st.session_state.use_existing_freqs)
#Proceed to actual calculation
if (st.session_state.reinforce_status == 2 and st.session_state.use_existing_freqs < 0) or st.session_state.use_existing_freqs >= 0:
    st.session_state.good_frequencies_selected = True
else:
    st.session_state.good_frequencies_selected = False
    
if st.session_state.good_frequencies_selected and st.session_state.trimmed_signal is not None and st.session_state.nominals_confirmed or (st.session_state.analysis_status == 2):
    if st.session_state.allcerts is None and st.session_state.analysis_status == 2:
        st.session_state.analysis_status = 0
    #st.write(st.session_state.analysis_status)

    if st.session_state.good_frequencies_selected and st.session_state.trimmed_signal is not None:
        if st.session_state.use_existing_freqs < 0 and st.session_state.analysis_status != 2:
            st.empty().write('New frequency profile calculated. Ready to find strike times.')
        elif st.session_state.analysis_status != 2:
            st.empty().write('Existing frequency profile loaded. Ready to find strike times.')
        else:
            st.empty().write("Strike times already found but you're welcome to try again if it's gone wrong.")
        def change_analysis():
            if st.session_state.analysis_status == 1:
                if st.session_state.final_freqs is not None:
                    st.session_state.analysis_status = 2    #Need a done condition here
                else:
                    st.session_state.analysis_status = 0
            else:
                st.session_state.analysis_status = 1
            return

        if st.session_state.testing_mode:
            st.session_state.analysis_status = 1
        else:
            if st.session_state.analysis_status != 1:
                st.empty().button("Find strike times", on_click = change_analysis)
            else:
                st.empty().button("Stop", on_click = change_analysis)
            
    st.analysis_log = st.empty()
    st.analysis_sublog = st.empty()
    st.analysis_sublog2 = st.empty()

    if st.session_state.analysis_status == 1:
        st.session_state.incache = False
        st.analysis_log.write('**Finding strike times**')
        st.analysis_sublog.progress(0, text = 'Finding initial rhythm')

        #Need to establish initial rhythm again in case this has changed. Shouldn't take too long.
        establish_initial_rhythm(st.session_state.Paras, final = True)
       
        #Load in final frequencies as session variables
        if st.session_state.use_existing_freqs < 0:
            st.session_state.final_freqs = st.session_state.reinforce_test_frequencies
            st.session_state.final_freqprobs = st.session_state.reinforce_frequency_profile
        else:
            st.session_state.final_freqs = np.load('./frequency_data/' + st.session_state.existing_filename + '_freqs.npy')
            st.session_state.final_freqprobs = np.load('./frequency_data/' + st.session_state.existing_filename + '_freqprobs.npy')
            
        find_final_strikes(st.session_state.Paras)
        filter_final_strikes(st.session_state.Paras)

        if len(st.session_state.allstrikes) == 0:
            st.session_state.analysis_status = 0
        else:
            st.session_state.analysis_status = 2
        #Update freuqency data to reflect the quality of the whole thing
        bellconfs_individual = np.mean(np.array(st.session_state.allcerts)[1:,:], axis = 0)
        st.session_state.reinforce_frequency_data = np.array([st.session_state.Paras.dt, st.session_state.Paras.fcut_length, np.mean(np.array(st.session_state.allcerts)[1:]), np.min(np.array(st.session_state.allcerts)[1:])])
        st.session_state.reinforce_frequency_data = np.concatenate((st.session_state.reinforce_frequency_data, bellconfs_individual))

        if not st.session_state.testing_mode:
            st.rerun()

    if st.session_state.analysis_status == 2:
        st.analysis_log.write('**Recording Analysed**')
        st.analysis_sublog.progress(100, text = 'Analysis complete')
        c = find_colour(np.mean(st.session_state.allcerts))

        st.analysis_sublog2.write('**%d rows found with average confidence :%s[%.1f%%]**' % (len(st.session_state.allstrikes[0]), c,  100*np.mean(st.session_state.allcerts)))

        allbells = []
        allstrikes = []
        yvalues = np.arange(len(st.session_state.allstrikes[:,0])) + 1
        orders = []
        for row in range(len(st.session_state.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((st.session_state.allstrikes[:,row]).tolist())
            allbells = allbells + order.tolist()
            orders.append(order)
        allstrikes = 1000*np.array(allstrikes)*0.01
        allbells = np.array(allbells)
        orders = np.array(orders)

        methods, hunt_types, calls, start_row, end_row, allrows_correct, quality = find_method_things(allbells)

        #Judge if these frequencies are fine
        goodenough = True
        if np.mean(st.session_state.allcerts) < 0.95:
            goodenough = False
        if len(st.session_state.allstrikes[0]) < 60.0:
            goodenough = False
        if (quality > 0.95 and len(allrows_correct) > 60):
            goodenough = True

        #If it's good, give an option to save out so it can be used next time
        if st.session_state.use_existing_freqs == -1 and not st.session_state.already_saved and goodenough and freq_filename is not None and st.session_state.reinforce_frequency_data is not None:
            #Check for an automatic save. If these are better than the existing AND either 98% + or ge. than 60 changes of a method with good match
            save_automatically = False
            if np.mean(st.session_state.allcerts) > best_freq_quality:
                if np.mean(st.session_state.allcerts) > 0.98:
                    save_automatically = True
                elif np.mean(st.session_state.allcerts) > 0.95 and quality > 0.95 and len(allrows_correct) > 60 and len(methods) > 0:
                    save_automatically = True

            st.save_option = st.empty()
            st.save_button = st.empty()

            st.save_option.write("Save these bell frequency profiles for future use? Only do so if you're confident this analysis is accurate and there don't already exist better profiles.")
    
            #Redefine to be based on an entire touch given that's what matters
            st.session_state.reinforce_frequency_data[2] = np.mean(st.session_state.allcerts)
            if save_automatically:
                np.save('%s%s_freqs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_test_frequencies)
                np.save('%s%s_freqprobs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_profile)
                np.save('%s%s_freq_quality.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_data)
                st.session_state.already_saved = True
                if not st.session_state.testing_mode:
                    st.rerun()

            if st.save_button.button("Save bell frequency profiles", disabled = st.session_state.already_saved):
                np.save('%s%s_freqs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_test_frequencies)
                np.save('%s%s_freqprobs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_profile)
                np.save('%s%s_freq_quality.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_data)
                st.session_state.already_saved = True
                if not st.session_state.testing_mode:
                    st.rerun()
                
        if st.session_state.tower_name is None:
            st.session_state.tower_name = "Unknown Tower"

        if len(methods) > 0:
            if quality > 0.7:
                nchanges = len(allrows_correct) - 1

                if len(methods) == 1:   #Single method
                    method_title = methods[0][0]
                    if method_title.rsplit(' ')[0] == "Stedman" or method_title.rsplit(' ')[0] == "Erin":
                        method_title = method_title.rsplit(' ')[0] + " " + method_title.rsplit(' ')[1]

                else:   #Spliced methods
                    stages = [name.rsplit(' ',1)[-1] for  name in np.array(methods)[:,0]]
                    stagetypes = [name.rsplit(' ')[-2] + ' ' + name.rsplit(' ')[-1] for  name in np.array(methods)[:,0]]
                    if len(set(stagetypes)) == 1:
                        method_title = "Spliced " + stagetypes[0]
                    elif len(set(stages)) == 1:
                        method_title = "Spliced " + stages[0]
                    else:
                        method_title = "Spliced"
                    lead_length = 2*int(hunt_types[0][1] + 1)
                st.write("**Method(s) detected: " + str(nchanges) + " " + method_title + ", %.1f %% match**" % (100*quality))
            else:
                st.write("**Probably a method but not entirely sure what...**")
                
        else:
            st.write("**No method detected**")
            start_row = 0; end_row = len(allrows_correct)
        #Give options to save to the cache (so this works on the analysis page) or to download as a csv
        cache_names = [val[2] for val in st.session_state.cached_data]
        striking_df, orders = save_strikes(st.session_state.Paras)

        if not st.session_state.incache:
            if st.session_state.audio_filename in cache_names:   #This does the check. Seems reasonable
                update_index = cache_names.index(st.session_state.audio_filename)
                if st.session_state.handstroke_first:
                    #st.write('HANDSTROKE FIRST')
                    st.session_state.cached_strikes[update_index] = st.session_state.allstrikes
                    st.session_state.cached_certs[update_index] = st.session_state.allcerts
                    st.session_state.cached_data[update_index] = [st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename]
                    st.session_state.cached_rawdata[update_index] = striking_df

                    st.session_state.touch_length = len(st.session_state.allstrikes[0])
                else:
                    #st.write('BACKSTROKE FIRST')
                    st.session_state.cached_strikes[update_index] = st.session_state.allstrikes[:,1:]
                    st.session_state.cached_certs[update_index] = st.session_state.allcerts[:,1:]
                    st.session_state.cached_data[update_index] = [st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename]
                    st.session_state.cached_rawdata[update_index] = striking_df

                    st.session_state.touch_length = len(st.session_state.allstrikes[0])  - 1
                st.session_state.incache = True
                if not st.session_state.testing_mode:
                    st.rerun()
            else:
                #Introduce name check
                #if st.button("Save this striking data to the cache for analysis"):
                #st.write(st.session_state.handstroke_first)
                if st.session_state.handstroke_first:
                    #st.write('HANDSTROKE FIRST')
                    st.session_state.cached_strikes.append(st.session_state.allstrikes)
                    st.session_state.cached_certs.append(st.session_state.allcerts)
                    st.session_state.cached_data.append([st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename])
                    st.session_state.cached_rawdata.append(striking_df)
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])

                    st.session_state.cached_touch_id.append(''.join(random.choices(string.ascii_letters + string.digits, k=10)))
                    st.session_state.cached_read_id.append(st.session_state.audio_filename)
                    st.session_state.cached_nchanges.append('')
                    st.session_state.cached_methods.append('')
                    st.session_state.cached_tower.append(st.session_state.tower_name)
                    st.session_state.cached_datetime.append(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    st.session_state.cached_score.append('')
                    st.session_state.cached_ms.append('')
                else:
                    #st.write('BACKSTROKE FIRST')
                    st.session_state.cached_strikes.append(st.session_state.allstrikes[:,1:])
                    st.session_state.cached_certs.append(st.session_state.allcerts[:,1:])
                    st.session_state.cached_data.append([st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename])
                    st.session_state.cached_rawdata.append(striking_df)
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])  - 1

                    st.session_state.cached_touch_id.append(''.join(random.choices(string.ascii_letters + string.digits, k=10)))
                    st.session_state.cached_read_id.append(st.session_state.audio_filename)
                    st.session_state.cached_nchanges.append('')
                    st.session_state.cached_methods.append('')
                    st.session_state.cached_tower.append(st.session_state.tower_name)
                    st.session_state.cached_datetime.append(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    st.session_state.cached_score.append('')
                    st.session_state.cached_ms.append('')

                st.session_state.current_touch = -1
                st.session_state.incache = True
                if not st.session_state.testing_mode:
                    st.rerun()
        else:
            st.page_link("pages/2_View_Touches.py", label = ":blue[View analysis for this touch]")


        if len(st.session_state.cached_strikes) == 1:
            st.write('%d set of striking data currently in the cache - view using the tab on the left or the link above' % len(st.session_state.cached_strikes))
        elif len(st.session_state.cached_strikes) > 1:
            st.write('%d sets of striking data currently in the cache - view using the tab on the left or the link above' % len(st.session_state.cached_strikes))
        
        #Give options to save to the cache (so this works on the analysis page) or to download as a csv
        #Create strike data in the right format (like the Strikeometer)
        #striking_df.attrs = {"Tower Name": st.session_state.tower_name, "Touch Length": st.session_state.touch_length, "File Name": st.session_state.audio_filename}
        @st.cache_data(ttl=300)
        def convert_for_download(df):
            return df.to_csv().encode("utf-8")

        csv = convert_for_download(striking_df)
        st.download_button("Download raw timing data to device as .csv", csv, file_name = st.session_state.audio_filename + '.csv', mime="text/csv")

        st.write("**To analyse another touch, change the tower or upload a new file**")

        if st.session_state.testing_mode:
            print('Rows detected:', len(st.session_state.allstrikes[0]))
            if len(methods) > 0 and quality > 0.8:
                print(str(nchanges) + " " + method_title + ", %.1f %% match" % (100*quality))
            else:
                print('No method detected')
            print('Confidence: ', "%.1f %% " % (100*np.mean(st.session_state.allcerts)))
            allstrikes_save = []

            for bell in range(len(st.session_state.allstrikes)):
                diff1s = st.session_state.allstrikes[0][2::2] - st.session_state.allstrikes[0][0:-2:2]
                diff2s = st.session_state.allstrikes[0][3::2] - st.session_state.allstrikes[0][1:-2:2]
            for row in range(len(st.session_state.allstrikes[0])):
                allstrikes_save = allstrikes_save + sorted((st.session_state.allstrikes[:,row]).tolist())
            allstrikes_save = np.array(allstrikes_save)
            ideal_times = find_ideal_times_rwp(allstrikes_save, st.session_state.Paras.nbells, key = 0)
            errors = allstrikes_save - ideal_times
            standard_deviation = np.sqrt(np.sum(errors**2)/len(errors))

            print('Approx SD', 1000*standard_deviation*st.session_state.Paras.dt)
            st.session_state.test_counter += 1
            st.rerun()
        else:
            st.stop()
