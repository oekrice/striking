# Copyright 2018-2022 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import inspect
import textwrap
import time
import numpy as np
from utils import show_code
import pandas as pd
from scipy.io import wavfile
import os

from listen_classes import audio_data, parameters
from listen_main_functions import establish_initial_rhythm, do_reinforcement

def plotting_demo():
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()   #Used as a space for the next though it's not there yet
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")


st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("# Analyse Striking")
#st.sidebar.header("Analyse Striking")
st.write(
    """This page is designed to test analysing the striking from uploaded (or perhaps live...) audio"""
    
)

st.markdown(
    """
    1. Select a tower from drop down from Dove. This populates frequency data
    2. Select the bells which are being rung with checkboxes
    3. (Optionally) tweak the parameters
    4. Upload the audio
    5. Trim start and end times
    6. Display if existing frequency data exists on the device and give an option to use if so
    7. This should all happen automatically. Then put a big button up
    8. If determining frequencies, but a progress bar with percentage quality
    9. Have an interrupt button for when this is good enough
    10. Display 'frequencies done' and give the option to start
    """
)
if "counter" not in st.session_state:
    st.session_state.counter = 0
if 'nominals' not in st.session_state:
    st.session_state.nominals = False
if 'tower' not in st.session_state:
    st.session_state.tower = False
if 'isfile' not in st.session_state:
    st.session_state.isfile = False
if 'reinforce' not in st.session_state:
    st.session_state.reinforce = 0
if 'tower_name' not in st.session_state:
    st.session_state.tower_name = None
if 'raw_file' not in st.session_state:
    st.session_state.raw_file = None 
if 'found_new_freqs' not in st.session_state:
    st.session_state.found_new_freqs = False
    
def reset_nominals():
    st.session_state.nominals = False
    st.session_state.isfile = False
    st.session_state.reinforce = 0

st.write('count', st.session_state.counter, st.session_state.reinforce, st.session_state.found_new_freqs)
st.session_state.counter += 1

progress_counter = 0   #How far through the thing is

@st.cache_data               
def read_bell_data():
    nominal_import = pd.read_csv('./bell_data/nominal_data.csv')
    return nominal_import

nominal_data = read_bell_data()

tower_names = nominal_data["Tower Name"].tolist()

default_index = tower_names.index("Brancepeth, S Brandon")

if not st.session_state.tower_name:
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = None, key=None, placeholder="Choose a tower", label_visibility="visible")
else:
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = tower_names.index(st.session_state.tower_name), key=None, placeholder="Choose a tower", label_visibility="visible")

if st.session_state.tower_name:
    st.session_state.tower = True 

if st.session_state.tower:
    selected_index = tower_names.index(st.session_state.tower_name)
    st.session_state.tower_id = nominal_data["Tower ID"][selected_index]

    bell_options = list(nominal_data.columns)[3:]
    #Determine the number of valid bells
    bell_names = []; bell_nominals = []
    
    nbells_max = 0
    for option in bell_options:
        if float(nominal_data.loc[selected_index][option]) > 0:
            bell_names.append(option)
            if option.isnumeric():
                nbells_max = max(nbells_max, int(option))
    
    st.write("Ring of ", str(nbells_max), "with", str(len(bell_names)), "bells to choose from:")
    
    nrows = len(bell_names)//11 + 1
    per_row = 1 + len(bell_names)//nrows 
    
    for row in range(nrows):
        #Need more than one row sometimes (if more than 8?)
        start = row*per_row
        end = min(len(bell_names), (row+1)*per_row)
        cols = st.columns(end-start)
        mincheck = len(bell_names)-1; maxcheck = 0
        #Display checkboxes
        for i in range(start, end):
            with cols[i%per_row]:
                if bell_names[i].isnumeric():
                    checked = st.checkbox(bell_names[i], value = True)#, on_change = reset_nominals)
                else:
                    checked = st.checkbox(bell_names[i], value = False)#, on_change = reset_nominals)
                if checked:
                    mincheck = min(i, mincheck)
                    maxcheck = max(i, maxcheck)
                    bell_nominals.append(float(nominal_data.loc[selected_index, bell_names[i]]))
    
    bell_nominals = sorted(bell_nominals, reverse = True)
    st.write(str(len(bell_nominals)), 'bells selected, with (editable) nominal frequencies in Hz')

    
    with st.form("Frequency Data"):

        #Make nice dataframe for this
        freq_df = pd.DataFrame(data = np.array(bell_nominals)[np.newaxis,:], columns = ["Bell %d" % (ri + 1) for ri in range(len(bell_nominals))])
            
        freq_df_new = st.data_editor(freq_df, hide_index = True)
                    
        bell_nominals = np.array(freq_df.loc[0])
                    
        if st.form_submit_button("Confirm Nominal Frequencies"):
            st.session_state.nominals = True
            st.write("Nominal frequencies confirmed.")
            
@st.cache_data               
def process_audio_files(raw_file):
    #Function that needs caching to avoid the need to keep uploading and converting things
        
    Audio = audio_data(raw_file)
    
    if Audio.signal is not None:
        
        st.write('Imported audio length: %d seconds.' % (len(Audio.signal)/Audio.fs))
    return Audio
    
if st.session_state.nominals:
    
    #Establish filename for the frequencies.
    #Needs to contain tower, first and last bells, and a counter. Can work on formats in a bit.
    freq_root = '%05d_%02d_%02d' % (st.session_state.tower_id, mincheck, maxcheck)
    
    #Determine colours:
    colour_thresholds = [0.95,0.98]; colours = ['red', 'orange', 'green']

    #Check for files with this handle
    existing_files = 0; allquals = []; allcs = []; max_existing = -1
    existing_freq_files = os.listdir('./frequency_data/')
    for file in existing_freq_files:
        if file[:len(freq_root)] == freq_root:
            if file[-11:] == "quality.npy":
                max_existing = max(max_existing, int(file[12:15]))
                quals = np.load('./frequency_data/' + file)
                allquals.append(quals[2])
                c = colours[0]
                if quals[2] > colour_thresholds[0]:
                    c = colours[1]
                if quals[2] > colour_thresholds[1]:
                    c = colours[2]
                allcs.append(c)
                    
            existing_files += 1
    frequency_counter = existing_files//3    #ID of THIS frequency set
    freq_filename = freq_root + '_%03d' % (max_existing + 1)         
    
    if frequency_counter == 1:
        st.write('Found %d existing frequency profile which match the selected bells:' % frequency_counter)
        #st.write('Choose existing profile or make a new one (can change your mind later):.')
        options = st.radio("Choose existing profile or make a new one (can change your mind later):", ["Make new profile", ":%s[Profile 1: %.1f%% match]" % (allcs[0], 100*allquals[0])])
        
    elif frequency_counter > 1:
        st.write('Found %d existing frequency profiles which match the selected bells...' % frequency_counter)
        #st.write('Choose existing profile or make a new one (can change your mind later):')
        allstrings = ["Make new profile"]
        for qi, qual in enumerate(allquals):
            allstrings.append(":%s[Profile %d: %.1f%% match]" % (allcs[qi], qi + 1, 100*qual))
        options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings)

    else:
        st.write('No existing frequency profiles which match these bells --  need to create one using the uploaded audio.')

    #Nominal frequencies detected. Proceed to upload audio...
    #st.write("Upload ringing audio:")
    
    raw_file = st.file_uploader("Upload ringing audio")
    
    if raw_file is not None:
        st.session_state.raw_file = raw_file
            
    if st.session_state.raw_file is not None:
        st.session_state.isfile = True
    else:
        st.session_state.isfile = False
        
    if st.session_state.isfile:
        Audio = process_audio_files(st.session_state.raw_file)   
       
    
if st.session_state.isfile:

    def change_reinforce():
        if st.session_state.reinforce != 1:
            st.session_state.reinforce = 1
        elif st.session_state.found_new_freqs:
            st.session_state.reinforce = 2
        else:
            st.session_state.reinforce = 0
        return

    with st.form("Set up parameters"):
        
        st.write("Audio parameters:")
        tmax = len(Audio.signal)/Audio.fs
        
        overall_tmin, overall_tmax = st.slider("Trim audio for use overall:", min_value = 0.0, max_value = 0.0, value=(0.0, tmax), format = "%ds")
        
        rounds_tmax = st.slider("Max. length of reliable rounds (be conservative):", min_value = 0.0, max_value = min(60.0, tmax), value=(30.0), format = "%ds")
        
        reinforce_tmax = st.slider("Max. time for reinforcement (longer is slower but more accurate):", min_value = 0.0, max_value = min(90.0, tmax), value=(60.0), format = "%ds")
    
        nreinforces = int(st.slider("Max number of frequency reinforcements:", min_value = 2, max_value = 10, value = 5, step = 1))
        
        if st.session_state.reinforce == 0 or st.session_state.reinforce == 2:
            st.form_submit_button("Find new frequency profiles", on_click = change_reinforce)
        else: 
            st.form_submit_button("Stop finding frequencies", on_click = change_reinforce)
                
if st.session_state.reinforce > 0:
    #Begin frequency reinforcement -- stop when the flag stops being on!
    #Zero for not at all, 1 for doing it and 2 for done
    if st.session_state.reinforce == 1:
        Paras = parameters(Audio, bell_nominals, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, nreinforces)
        Paras.fname = str(st.session_state.tower_id)
    
    st.main_log = st.empty()
    st.main_log.write('**Detecting initial rhythm**')
    
    st.quality_log = st.empty()
    st.quality_log.write('Frequency confidence: %.1f %%' % 0.0)

    st.current_log = st.empty()
    if st.session_state.reinforce == 1:
        st.current_log.write('Detecting ringing...')

        Data = establish_initial_rhythm(Audio, Paras)
        
        st.current_log.write('Initial rhythm established with ' + str(len(Data.strikes)) + ' tenor strikes.')
        
        Data = do_reinforcement(Paras, Data, Audio)

        if Data.freq_data[2] > 0.95:
            
            st.write('Good enough', Data.freq_data[2])
            st.session_state.reinforce = 2
        else:
            st.write('Not Good enough', Data.freq_data[2])
            st.session_state.reinforce = 0

        time.sleep(5.0)
        
        st.main_log.write('**Bell frequency profiles established**')
        
        st.rerun()
        
    if st.session_state.reinforce == 2:   #At least some frequency reinforcement has happened
        
        st.current_log.write('Things have finished')

    

        #fs, data = wavfile.read(raw_file.getvalue())        
    #See if it is a .wav or not with a try and except
    
#plotting_demo()
#show_code(plotting_demo)
















