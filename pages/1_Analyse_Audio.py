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

'''
Some thoughts on logic:
For the first time in the session, don't display anything other than the tower box. 
BUT after this, always have the frequency box up
Then after the FIRST TIME commiting frequencies the rest appears with the audio etc. -- and stays there.
DO need to reset parameter sliders etc. after new audio. That's OK.
Finding frequencies -- need a couple of states. 0 for not done frequencies, 1 for doing and 2 for done
Separately, have a check for if the existing loaded data is fine. Could be done frequencies or loaded in ones.
OK.
'''
#Establish persistent variables

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
if 'raw_file' not in st.session_state:
    st.session_state.raw_file = None 
if 'found_new_freqs' not in st.session_state:
    st.session_state.found_new_freqs = False
if 'use_existing_freqs' not in st.session_state:
    st.session_state.use_existing_freqs = -1   #Positive if do want to use existing frequencies. Negative if not.
if 'already_saved' not in st.session_state:
    st.session_state.already_saved = False   #Positive if do want to use existing frequencies. Negative if not.
    
#Frequency data to be saved throughout
if 'reinforce_test_frequencies' not in st.session_state:
    st.session_state.reinforce_test_frequencies = None   #Positive if do want to use existing frequencies. Negative if not.
if 'reinforce_frequency_profile' not in st.session_state:
    st.session_state.reinforce_frequency_profile = None   #Positive if do want to use existing frequencies. Negative if not.
if 'reinforce_frequency_data' not in st.session_state:
    st.session_state.reinforce_frequency_data = None   #Positive if do want to use existing frequencies. Negative if not.
  
def reset_nominals():
    st.session_state.nominals_confirmed = False
    st.session_state.bell_nominals = False
    st.session_state.reinforce_frequency_data = None
    st.session_state.reinforce_status = 0
    st.write('Resetting nominals')
    #st.session_state.isfile = False
    #st.session_state.reinforce = 0

st.write('Init status', st.session_state.counter, st.session_state.reinforce_status, st.session_state.found_new_freqs)
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
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = None, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)
else:
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = None, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)

if st.session_state.tower_name:
    st.session_state.tower_selected = True 
else:
     st.session_state.tower_selected = False    

if st.session_state.tower_selected:
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
    per_row = len(bell_names)//nrows 
    mincheck = len(bell_names)-1; maxcheck = 0

    nbells_save = 0; max_bell = 0

    for row in range(nrows):
        #Need more than one row sometimes (if more than 8?)
        start = row*per_row
        end = min(len(bell_names), (row+1)*per_row)
        cols = st.columns(end-start)
        #Display checkboxes
        for i in range(start, end):
            with cols[i%per_row]:
                if bell_names[i].isnumeric():
                    checked = st.checkbox(bell_names[i], value = True, on_change = reset_nominals)
                    max_bell = max(max_bell, int(bell_names[i]))    
                else:
                    checked = st.checkbox(bell_names[i], value = False, on_change = reset_nominals)
                if checked:
                    mincheck = min(i, mincheck)
                    maxcheck = max(i, maxcheck)
                    bell_nominals.append(float(nominal_data.loc[selected_index, bell_names[i]]))
        
    nbells_save = len(bell_nominals)
    bell_nominals = sorted(bell_nominals, reverse = True)
      
    st.write(str(len(bell_nominals)), 'bells selected, with (editable) nominal frequencies in Hz')
        
    #Make nice dataframe for this
    
    if st.session_state.nominals_confirmed:
        #These have been confirmed so don't repopulate unless they've been reset otherwise
        freq_df = pd.DataFrame(data = np.array(st.session_state.bell_nominals)[np.newaxis,:], columns = ["Bell %d" % (ri + 1) for ri in range(len(bell_nominals))])
    else:
        #These haven't been confirmed -- fill dataframe will raw data
        freq_df = pd.DataFrame(data = np.array(bell_nominals)[np.newaxis,:], columns = ["Bell %d" % (ri + 1) for ri in range(len(bell_nominals))])
        
    edited_nominals = np.array(st.data_editor(freq_df, hide_index = True, on_change = reset_nominals))[0].tolist()
    
    if st.button("Confirm Tower and Frequencies"):
        st.session_state.nominals_confirmed = True
        st.session_state.bell_nominals = edited_nominals
            
    #if st.session_state.nominals_confirmed:
    #    st.write("Nominal frequencies confirmed (remove this later)")

@st.cache_data               
def process_audio_files(raw_file):
    #Function that needs caching to avoid the need to keep uploading and converting things
        
    Audio = audio_data(raw_file)
    
    if Audio.signal is not None:
        
        st.write('Imported audio length: %d seconds.' % (len(Audio.signal)/Audio.fs))
    return Audio
    
if st.session_state.tower_selected and st.session_state.nominals_confirmed:
    
    #This should come up EVERY time after the first confirmation
    
    #Establish filename for the frequencies.
    #Needs to contain tower, first and last bells, and a counter. Can work on formats in a bit.
    freq_root = '%05d_%02d_%02d' % (st.session_state.tower_id, nbells_save, max_bell)
    
    #rst.write(freq_root)
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
        st.write('Found %d existing frequency profile which matches the selected bells:' % frequency_counter)
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
        st.write('No existing frequency profiles which match these bells -- will need to create one.')

    #Nominal frequencies detected. Proceed to upload audio...
    #st.write("Upload ringing audio:")
    def reset_on_upload():
        st.session_state.reinforce_frequency_data = None
        st.session_state.reinforce_status = 0

    raw_file = st.file_uploader("Upload ringing audio for analysis", on_change = reset_on_upload)
    
    if raw_file is not None:
        st.session_state.raw_file = raw_file
            
    if st.session_state.raw_file is not None:
        st.session_state.file_uploaded = True
    else:
        st.session_state.file_uploaded = False
        
    if st.session_state.file_uploaded:
        Audio = process_audio_files(st.session_state.raw_file)   
       
if st.session_state.file_uploaded and st.session_state.nominals_confirmed:

    def change_reinforce():
        if st.session_state.reinforce_status != 1:
            st.session_state.reinforce_status = 1
        elif st.session_state.reinforce_frequency_data is not None:
            st.session_state.reinforce_status = 0
            if st.session_state.reinforce_frequency_data[2] > 0.95:
                st.session_state.reinforce_status = 2
            else:
                st.session_state.reinforce_status = 0
        else:
            st.session_state.reinforce_status = 0
        return
        
    st.write("Audio parameters:")
    tmax = len(Audio.signal)/Audio.fs
    
    overall_tmin, overall_tmax = st.slider("Trim audio for use overall:", min_value = 0.0, max_value = 0.0, value=(0.0, tmax), format = "%ds")
    
    rounds_tmax = st.slider("Max. length of reliable rounds (be conservative):", min_value = 20.0, max_value = min(60.0, tmax), value=(30.0), format = "%ds")
    
    reinforce_tmax = st.slider("Max. time for frequency analysis (longer is slower but more accurate):", min_value = 60.0, max_value = min(120.0, tmax), value=(90.0), format = "%ds")

    nreinforces = int(st.slider("Max number of frequency analysis steps:", min_value = 2, max_value = 10, value = 5, step = 1))
    
    Paras = parameters(Audio, np.array(st.session_state.bell_nominals), overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, nreinforces)
    Paras.fname = str(st.session_state.tower_id)

    if st.session_state.reinforce_status == 0 or st.session_state.reinforce_status == 2:
        st.button("Find new frequency profiles", on_click = change_reinforce)
    else: 
        st.button("Stop finding frequencies", on_click = change_reinforce)
                    
    if st.session_state.reinforce_status == 1:
        #Begin frequency reinforcement -- stop when the flag stops being on!
        #Zero for not at all, 1 for doing it and 2 for done. Need something to fill this space if never doing reinforcement
        st.main_log = st.empty()
        st.main_log.write('**Detecting initial rhythm**')
        
        st.quality_log = st.empty()

        if st.session_state.reinforce_frequency_data is not None:
            colour_thresholds = [0.95,0.98]; colours = ['red', 'orange', 'green']
            toprint = st.session_state.reinforce_frequency_data[2]
            c = colours[0]
            if toprint > colour_thresholds[0]:
                c = colours[1]
            if toprint > colour_thresholds[1]:
                c = colours[2]
    
            st.quality_log.write('Best yet frequency match = :%s[%.1f%%]' % (c, 100*toprint))
        else:
            st.quality_log.write('Best yet frequency match = :%s[%.1f%%]' % ('red', 0.0))
    
        st.current_log = st.empty()
        
        st.current_log.write('Detecting ringing...')
    
        st.save_option = st.empty()
        st.save_button = st.empty()

        Data = establish_initial_rhythm(Audio, Paras)
                
        st.current_log.write('Established initial rhythm using ' + str(len(Data.strikes[0])) + ' changes')
                
        Data = do_reinforcement(Paras, Data, Audio)
    
        st.write(st.session_state.reinforce_frequency_data)
        
        if st.session_state.reinforce_frequency_data is not None:
            if st.session_state.reinforce_frequency_data[2] > 0.90:
                st.session_state.reinforce_status = 2
            else:
                st.session_state.reinforce_status = 0
        else:
            st.session_state.reinforce_status = 0
                    
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
            st.current_log.write('This might not be good enough to provide anything useful. But may as well try...')
        elif toprint < 0.975:
            st.current_log.write('Not perfect but it\'ll probably do.')
        else:
            st.current_log.write('That should be fine to detect everything reasonably well.')
  
        #If it's good, give an option to save out so it can be used next time
        if st.session_state.reinforce_frequency_data[2] > 0.95 :
            st.save_option.write('Save these frequency profiles for future use? They will be available to all users.')
    
            if st.save_button.button("Save frequency profiles", disabled = st.session_state.already_saved):
                np.save('%s%s_freqs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_test_frequencies)
                np.save('%s%s_freqprobs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_profile)
                np.save('%s%s_freq_quality.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_data)
                st.session_state.already_saved = True
                st.rerun()





















