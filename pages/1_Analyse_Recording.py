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
import sys

from listen_classes import audio_data, parameters
from listen_main_functions import establish_initial_rhythm, do_reinforcement, find_final_strikes, save_strikes
from listen_other_functions import find_colour
from methods import find_method_things

st.set_page_config(page_title="Analyse Recording", page_icon="🎤")
st.markdown("## Analyse Recording")
#st.sidebar.header("Analyse Striking")
st.markdown(
    """
    This page is to find strike times from uploaded (or perhaps live at some point...) audio.
    1. Select the tower and bells being rung.
    2. Upload the audio file for the recording. Ringing must start within 1 minute of the start of the (trimmed) recording, and **must begin with at least three whole pulls of approximate rounds**.   
    3. Choose whether to use existing frequency profiles (if they exist) or learn new ones.
    4. If the latter, you'll be given the option to do this. This can be quite slow, especially for more than 8 bells.
    5. Once decent frequencies are found, the code will attempt to strike times throughout. 
    6. These strike times can then be analysed on the other page, or downloaded as a .csv for use later.   
    """
)

if not os.path.exists('./tmp/'):
    os.system('mkdir ./tmp/')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir ./frequency_data/')
if not os.path.exists('./striking_data/'):
    os.system('mkdir ./striking_data/')

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
if 'trim_flag' not in st.session_state:
    st.session_state.trim_flag = False   
if 'checked' not in st.session_state:
    st.session_state.checked = []  
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

#Audio things to be cached -- need to be careful here
if 'audio_signal' not in st.session_state:
    st.session_state.audio_signal = None
if 'trimmed_signal' not in st.session_state:
    st.session_state.trimmed_signal = None
if 'fs' not in st.session_state:
    st.session_state.fs = None
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
    
gc.collect()

Audio = None
Data = None
Paras = None

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

st.session_state.counter += 1

progress_counter = 0   #How far through the thing is

@st.cache_data               
def read_bell_data():
    nominal_import = pd.read_csv('./bell_data/nominal_data.csv')
    return nominal_import

nominal_data = read_bell_data()
freq_filename = None

tower_names = nominal_data["Tower Name"].tolist()

default_index = tower_names.index("Brancepeth, S Brandon")

existing_selected = False
if st.session_state.tower_name is not None:
    if st.session_state.tower_name != "Unknown Tower":
        existing_selected = True
        index = tower_names.index(st.session_state.tower_name)
        #st.session_state.nominals_confirmed = True

if not existing_selected:
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = None, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)
else:
    st.session_state.tower_name = st.selectbox('Select tower...', tower_names, index = index, key = None, placeholder="Choose a tower", label_visibility="visible", on_change = reset_nominals)

if st.session_state.tower_name:
    st.session_state.tower_selected = True 
else:
    st.session_state.tower_selected = False  
    st.session_state.nominals_confirmed = False  
    st.write("If you can't find the required tower in the list, try to find a similarly-tuned ring (look for the nominal frequencies) on Dove's guide and that may work -- but please don't save the frequency profiles.")
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
    per_row = int((len(bell_names)-1e-6)//nrows) + 1 
    mincheck = len(bell_names)-1; maxcheck = 0

    nbells_save = 0; max_bell = 0

    allchecked = []
    for row in range(nrows):
        #Need more than one row sometimes (if more than 8?)
        start = row*per_row
        end = min(len(bell_names), (row+1)*per_row)
        cols = st.columns(end-start)
        #Display checkboxes
        for i in range(start, end):
            with cols[i%per_row]:
                if len(st.session_state.checked) > 0:
                    if bell_names[i] in st.session_state.checked:
                        checked = st.checkbox(bell_names[i], value = True, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                        max_bell = max(max_bell, int(bell_names[i]))    
                    else:
                        checked = st.checkbox(bell_names[i], value = False, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                else:
                    if bell_names[i].isnumeric():
                        checked = st.checkbox(bell_names[i], value = True, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                        max_bell = max(max_bell, int(bell_names[i]))    
                    else:
                        checked = st.checkbox(bell_names[i], value = False, on_change = reset_nominals, key = i + st.session_state.tower_id*1000)
                if checked:
                    allchecked.append(bell_names[i])
                    mincheck = min(i, mincheck)
                    maxcheck = max(i, maxcheck)
                    bell_nominals.append(float(nominal_data.loc[selected_index, bell_names[i]]))
    st.session_state.checked = allchecked

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
    
    if st.button("Confirm Tower and Frequencies", disabled = st.session_state.nominals_confirmed):
        st.session_state.nominals_confirmed = True
        st.session_state.bell_nominals = edited_nominals
        st.rerun()
    #if st.session_state.nominals_confirmed:
    #    st.write("Nominal frequencies confirmed (remove this later)")
    
#@st.cache_data(ttl=60)
def process_audio_files(raw_file, doprints):
    #Function that needs caching to avoid the need to keep uploading and converting things
    audio_data(raw_file, doprints)
        
    return st.session_state.audio_signal, st.session_state.fs, st.session_state.audio_filename
    
if st.session_state.tower_selected and st.session_state.nominals_confirmed:
    #This should come up EVERY time after the first confirmation. Actually no.
    
    #Establish filename for the frequencies.
    #Needs to contain tower, first and last bells, and a counter. Can work on formats in a bit.
    freq_root = '%05d_%02d_%02d' % (st.session_state.tower_id, nbells_save, max_bell)
    
    #rst.write(freq_root)

    #Check for files with this handle
    existing_files = 0; allquals = []; allcs = []; max_existing = -1
    existing_freq_files = os.listdir('./frequency_data/')
    for file in existing_freq_files:
        if file[:len(freq_root)] == freq_root:
            if file[-11:] == "quality.npy":
                max_existing = max(max_existing, int(file[12:15]))
                quals = np.load('./frequency_data/' + file)
                allquals.append(quals[2])
                c = find_colour(quals[2])
                allcs.append(c)
                    
            existing_files += 1
    frequency_counter = existing_files//3    #ID of THIS frequency set
    freq_filename = freq_root + '_%03d' % (max_existing + 1)         
        
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
        options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings, on_change = stop_analysis, index = 1)
        
    elif frequency_counter > 1:
        st.write('Found %d existing frequency profiles which match the selected bells...' % frequency_counter)
        #st.write('Choose existing profile or make a new one (can change your mind later):')
        allstrings = ["Make new profile"]
        maxind = np.where(allquals == np.max(allquals))[0][0]
        for qi, qual in enumerate(allquals):
            allstrings.append(":%s[Profile %d: %.1f%% match]" % (allcs[qi], qi + 1, 100*qual))
        options = st.radio("Choose existing profile or make a new one (can change your mind later):", allstrings, on_change = stop_analysis, index = int(maxind + 1))
    
    else:
        st.write('No existing frequency profiles which match these bells -- will need to create one.')
        options = "Make new profile"
        
    if options == "Make new profile":
        st.session_state.use_existing_freqs = -1
    else:
        st.session_state.use_existing_freqs  = allstrings.index(options) - 1
        existing_filename = freq_root + '_%03d' % (st.session_state.use_existing_freqs)  
    #Nominal frequencies detected. Proceed to upload audio...
    #st.write("Upload ringing audio:")
    def reset_on_upload():
        st.session_state.reinforce_frequency_data = None
        st.session_state.reinforce_status = 0
        st.session_state.analysis_status = 0 #If new file is uploaded, reset status
        st.session_state.trim_flag = False
        st.session_state.allstrikes = None  
        st.session_state.allcerts = None  

    #st.write(st.session_state.audio_signal is not None)

    raw_file = st.file_uploader("Upload recording of ringing for analysis, or record directly (works on Android sometimes)", on_change = reset_on_upload, key = st.session_state.uploader_key)
           
    #st.write(raw_file is not None, st.session_state.audio_signal is not None)
    
    if raw_file is not None:
        process_audio_files(raw_file, doprints = True)  
        del raw_file
        st.rerun()
        
    #st.write(st.session_state.trimmed_signal is not None)
    #st.write(raw_file is not None, st.session_state.audio_signal is not None)

    if st.session_state.trimmed_signal is not None:
        st.write('Audio file "%s" read in successfully.' % st.session_state.audio_filename)
        st.write('Trimmed audio length: %d seconds.' % (len(st.session_state.trimmed_signal)/st.session_state.fs))
    elif st.session_state.audio_signal is not None:
        #Put some prints to indicate a file has been uploaded
        st.write('Audio file "%s" read in successfully.' % st.session_state.audio_filename)
        st.write('Imported audio length: %d seconds.' % (len(st.session_state.audio_signal)/st.session_state.fs))

    if ['uploaded_file'] in st.session_state:
        del st.session_state['uploaded_file']


#st.write('Audio in', st.session_state.audio_signal is not None)

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
        
    st.write("Audio parameters:")
    tmax = len(st.session_state.audio_signal)/st.session_state.fs

    overall_tmin, overall_tmax = st.slider("Trim audio for use overall (remove silence before ringing if possible):", min_value = 0.0, max_value = 0.0, value=(0.0, tmax),step = 1. ,format = "%ds", disabled = False)
        
    if st.session_state.use_existing_freqs < 0:
        reinforce_tmax = st.slider("Length of recording used for frequency analysis -- longer is slower but more accurate (don't include bad ringing):", min_value = 45.0, max_value = min(120.0, tmax), step = 1., value=(60.0), format = "%ds")
        nreinforces = int(st.slider("Max number of frequency analysis steps -- may not need this many and will stop automatically if so:", min_value = 2, max_value = 15, value = 10, step = 1))
    else:
        reinforce_tmax = 90.0
        nreinforces = 5

    Paras = parameters(np.array(st.session_state.bell_nominals), overall_tmin, overall_tmax, reinforce_tmax, nreinforces)
    Paras.fname = str(st.session_state.tower_id)

    if st.session_state.use_existing_freqs < 0:
        
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
        
            Data = establish_initial_rhythm(Paras)
            
            st.current_log.write('Established initial rhythm using ' + str(len(Data.strikes[0])) + ' changes')
                    
            Data = do_reinforcement(Paras, Data)
                    
            if st.session_state.reinforce_frequency_data is not None:
                if st.session_state.reinforce_frequency_data[2] > 0.85:
                    st.session_state.reinforce_status = 2
                else:
                    st.main_log.write("**Frequency profiles not good enough to use... Apologies**")
                    st.session_state.reinforce_status = 0
            else:
                st.session_state.reinforce_status = 0
                        
            del Data   #Can probably delete some other stuff as well...
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

            if st.session_state.analysis_status == 0:
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
        if st.session_state.use_existing_freqs < 0:
            st.empty().write('New frequency profile calculated. Ready to find strike times.')
        else:
            st.empty().write('Existing frequency profile loaded. Ready to find strike times.')
            
        def change_analysis():
            if st.session_state.analysis_status == 1:
                if st.session_state.final_freqs is not None:
                    st.session_state.analysis_status = 2    #Need a done condition here
                else:
                    st.session_state.analysis_status = 0
            else:
                st.session_state.analysis_status = 1
            return
    
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
        establish_initial_rhythm(Paras, final = True)
       
        #Load in final frequencies as session variables
        if st.session_state.use_existing_freqs < 0:
            st.session_state.final_freqs = st.session_state.reinforce_test_frequencies
            st.session_state.final_freqprobs = st.session_state.reinforce_frequency_profile
        else:
            st.session_state.final_freqs = np.load('./frequency_data/' + existing_filename + '_freqs.npy')
            st.session_state.final_freqprobs = np.load('./frequency_data/' + existing_filename + '_freqprobs.npy')
            
        find_final_strikes(Paras)

        if len(st.session_state.allstrikes) == 0:
            st.session_state.analysis_status = 0
        else:
            st.session_state.analysis_status = 2
        #Update freuqency data to reflect the quality of the whole thing
        bellconfs_individual = np.mean(np.array(st.session_state.allcerts)[1:,:], axis = 0)
        st.session_state.reinforce_frequency_data = np.array([Paras.dt, Paras.fcut_length, np.mean(np.array(st.session_state.allcerts)[1:]), np.min(np.array(st.session_state.allcerts)[1:])])
        st.session_state.reinforce_frequency_data = np.concatenate((st.session_state.reinforce_frequency_data, bellconfs_individual))

        st.rerun()

    if st.session_state.analysis_status == 2:
        st.analysis_log.write('**Audio Analysed**')
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

            st.save_option.write("Save these bell frequency profiles for future use? Only do so if you're confident this analysis is accurate")
    
            #Redefine to be based on an entire touch given that's what matters
            st.session_state.reinforce_frequency_data[2] = np.mean(st.session_state.allcerts)
            if save_automatically:
                np.save('%s%s_freqs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_test_frequencies)
                np.save('%s%s_freqprobs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_profile)
                np.save('%s%s_freq_quality.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_data)
                st.session_state.already_saved = True
                st.rerun()

            if st.save_button.button("Save bell frequency profiles", disabled = st.session_state.already_saved):
                np.save('%s%s_freqs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_test_frequencies)
                np.save('%s%s_freqprobs.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_profile)
                np.save('%s%s_freq_quality.npy' % ('./frequency_data/', freq_filename), st.session_state.reinforce_frequency_data)
                st.session_state.already_saved = True
                st.rerun()
                
        if st.session_state.tower_name is None:
            st.session_state.tower_name = "Unknown Tower"

        if len(methods) > 0:
            if quality > 0.8:
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

        if not st.session_state.incache:
            if st.session_state.audio_filename in cache_names:
                update_index = cache_names.index(st.session_state.audio_filename)
                if st.session_state.handstroke_first:
                    #st.write('HANDSTROKE FIRST')
                    st.session_state.cached_strikes[update_index] = st.session_state.allstrikes
                    st.session_state.cached_certs[update_index] = st.session_state.allcerts
                    st.session_state.cached_data[update_index] = [st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename]
                    st.session_state.cached_rawdata[update_index] = []
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])
                else:
                    #st.write('BACKSTROKE FIRST')
                    st.session_state.cached_strikes[update_index] = st.session_state.allstrikes[:,1:]
                    st.session_state.cached_certs[update_index] = st.session_state.allcerts[:,1:]
                    st.session_state.cached_data[update_index] = [st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename]
                    st.session_state.cached_rawdata[update_index] = []
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])  - 1
                st.session_state.incache = True
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
                    st.session_state.cached_rawdata.append([])
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])
                else:
                    #st.write('BACKSTROKE FIRST')
                    st.session_state.cached_strikes.append(st.session_state.allstrikes[:,1:])
                    st.session_state.cached_certs.append(st.session_state.allcerts[:,1:])
                    st.session_state.cached_data.append([st.session_state.tower_name, len(st.session_state.allstrikes[0]), st.session_state.audio_filename])
                    st.session_state.cached_rawdata.append([])
                    st.session_state.touch_length = len(st.session_state.allstrikes[0])  - 1
                st.session_state.incache = True
                st.rerun()
        else:
            st.page_link("pages/2_Analyse_Striking.py", label = ":blue[Analyse this striking]")


        if len(st.session_state.cached_strikes) == 1:
            st.write('%d set of striking data currently in the cache - view using the tab on the left or the link above' % len(st.session_state.cached_strikes))
        elif len(st.session_state.cached_strikes) > 1:
            st.write('%d sets of striking data currently in the cache - view using the tab on the left or the link above' % len(st.session_state.cached_strikes))
        
        #Give options to save to the cache (so this works on the analysis page) or to download as a csv
        #Create strike data in the right format (like the Strikeometer)
        striking_df, orders = save_strikes(Paras)
        #striking_df.attrs = {"Tower Name": st.session_state.tower_name, "Touch Length": st.session_state.touch_length, "File Name": st.session_state.audio_filename}
        @st.cache_data
        def convert_for_download(df):
            return df.to_csv().encode("utf-8")

        csv = convert_for_download(striking_df)
        st.download_button("Download raw timing data to device as .csv", csv, file_name = st.session_state.audio_filename + '.csv', mime="text/csv")

        st.write("**To analyse another touch, change the tower or upload a new file**")

        if False:
            with st.expander("View all rows and detection confidences"):
                for ri, row in enumerate(orders):
                    rowconf = st.session_state.allcerts[:,ri]
                    string = ''
                    for bell in row:
                        if bell < 10:
                            string = string + str(bell)
                        elif bell == 10:
                            string = string + '0'
                        elif bell == 11:
                            string = string + 'E'
                        elif bell == 12:
                            string = string + 'T'
                        elif bell == 13:
                            string = string + 'A'
                        elif bell == 14:
                            string = string + 'B'
                        elif bell == 15:
                            string = string + 'C'
                        elif bell == 16:
                            string = string + 'D'
                            
                    c = find_colour(np.mean(rowconf))
                    confstring = ('%d' % np.mean(rowconf)*100 )
                    st.write(':%s[%s -- %d %%]' % (c ,string, np.mean(rowconf)*100))



