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
import random

import random
import string
from datetime import datetime

import matplotlib.pyplot as plt
from methods import find_method_things

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
                uploaded_files.pop(uploaded_files.index(uploaded_file))
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
                    st.session_state.cached_score.append('')
                    st.session_state.cached_ms.append('')
                    #st.write(strike_data)
        os.system('rm -r ./tmp/%s' % uploaded_file.name)
           
    if len(uploaded_files) > 0:
        st.session_state.uploader_key += 1
        st.session_state.current_touch = -1

        st.rerun()
    return
    
def make_longtitle_cache(ti):
    def get_nice_date(date_string):
        def get_day_suffix(day):
            if 11 <= day <= 13:
                return 'th'
            last_digit = day % 10
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(last_digit, 'th')
        dt = datetime.strptime(date_string, "%m/%d/%Y, %H:%M:%S")   
        day = dt.day
        suffix = get_day_suffix(day)
        nice_day = f"{day}{suffix}"    
        return f"{nice_day} {dt.strftime('%B')} at {dt.strftime('%H:%M')}"
    #Creates a descriptive, readable title from the metadata
    longtitle = ''
    longtitle = longtitle + "**%s**: " % st.session_state.cached_read_id[ti]
    if st.session_state.cached_nchanges[ti] != '' and st.session_state.cached_methods[ti] != '':
        longtitle = longtitle + "%d %s" % (int(st.session_state.cached_nchanges[ti]), st.session_state.cached_methods[ti])
    if st.session_state.cached_tower[ti] != '':
        longtitle = longtitle + " at %s" % (st.session_state.cached_tower[ti])
    longtitle = longtitle + " (%s)" % get_nice_date(st.session_state.cached_datetime[ti])

    return longtitle

def sort_current_collection(saved_index_list, oldfirst = False):
    #Sorts based on recorded time (most recent at the top, I presume - but this is backwards later)
    list_array = np.array(saved_index_list)
    date_objects = []
    for i in range(len(saved_index_list)):
        date_objects.append(datetime.strptime(saved_index_list[i,5], "%m/%d/%Y, %H:%M:%S"))
    if oldfirst:
        sorted_list = np.array([val for _, val in sorted(zip(date_objects,list_array), reverse = True)])
    else:
        sorted_list = np.array([val for _, val in sorted(zip(date_objects,list_array), reverse = False)])
    np.savetxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, np.array(sorted_list, dtype = str), fmt = '%s', delimiter = ';')
    st.rerun()
    return


def make_longtitle_collection(touch_info):
    def get_nice_date(date_string):
        def get_day_suffix(day):
            if 11 <= day <= 13:
                return 'th'
            last_digit = day % 10
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(last_digit, 'th')
        dt = datetime.strptime(date_string, "%m/%d/%Y, %H:%M:%S")   
        day = dt.day
        suffix = get_day_suffix(day)
        nice_day = f"{day}{suffix}"    
        return f"{nice_day} {dt.strftime('%B')} at {dt.strftime('%H:%M')}"
    #Creates a descriptive, readable title from the metadata
    longtitle = ''
    longtitle = longtitle + "**%s**: " % touch_info[1]
    if touch_info[2] != '' and touch_info[3] != ' ':
        longtitle = longtitle + "%d %s" % (int(touch_info[2]), touch_info[3])
    if touch_info[4] != '':
        longtitle = longtitle + " at %s" % (touch_info[4])
    longtitle = longtitle + " (%s)" % get_nice_date(touch_info[5])
    if (touch_info[6] != '') and (touch_info[7] != ' '):
        longtitle = longtitle +  ' - **%.2f%%** (%dms)' % (float(touch_info[6]), float(touch_info[7]))
    return longtitle


def find_existing_names():
    #Finds a list of existing collection names
    names_raw = os.listdir('./saved_touches/')
    names_lower = []
    for name in names_raw:
        lower_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), name)   
        names_lower.append(lower_name)
    return names_lower

def change_addition_mode():
    st.session_state.addition_mode *= -1
    st.session_state.subtraction_mode = -1
    st.session_state.rename_mode = -1
    st.session_state.rename_collection_mode = -1
    return

def change_subtraction_mode():
    st.session_state.addition_mode = -1
    st.session_state.subtraction_mode *= -1
    st.session_state.rename_mode = -1
    st.session_state.rename_collection_mode = -1
    return

def change_rename_mode():
    st.session_state.addition_mode = -1
    st.session_state.subtraction_mode = -1
    st.session_state.rename_mode *= -1
    st.session_state.rename_collection_mode = -1
    return

def change_rename_collection_mode():
    st.session_state.addition_mode = -1
    st.session_state.subtraction_mode = -1
    st.session_state.rename_mode = -1
    st.session_state.rename_collection_mode *= -1
    return


def add_new_folder(name):
    os.system('mkdir "saved_touches/%s"' % name)
    np.savetxt("./saved_touches/%s/index.csv" % name, np.array([[' ',' ',' ',' ',' ',' ',' ',' ']], dtype = 'str'), fmt = '%s', delimiter = ';')
    st.session_state.collection_status = 0
    st.session_state.current_collection_name = name
    st.session_state.input_key += 1
    return

if not os.path.exists('./tmp/'):
    os.system('mkdir tmp')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir frequency_data')
if not os.path.exists('./striking_data/'):
    os.system('mkdir striking_data')
if not os.path.exists('./saved_touches/'):
    os.system('mkdir saved_touches')

st.set_page_config(page_title="Touch Library", page_icon="📖")
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "selected_library_touches" not in st.session_state:   #List in index space of the currently-selected touches
    st.session_state.selected_library_touches = []
if "addition_mode" not in st.session_state:
    st.session_state.addition_mode = -1 
if "subtraction_mode" not in st.session_state:
    st.session_state.subtraction_mode = -1 
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = -1 
if "rename_collection_mode" not in st.session_state:
    st.session_state.rename_collection_mode = -1 
if "new_index_list" not in st.session_state:
    st.session_state.new_index_list = None 
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
if 'analysis_button' not in st.session_state:
    st.session_state.analysis_button = True
if 'expanded' not in st.session_state:
    st.session_state.expanded = False
    #Remove the large things from memory -- need a condition on this maybe?
# st.session_state.trimmed_signal = None
# st.session_state.audio_signal = None
# st.session_state.raw_file = None
st.session_state.Paras = None
st.session_state.Data = None

touch_titles = []
raw_titles = []

#FInd the touches currently in the cache
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = st.session_state.cached_read_id[i]
    touch_titles.append(title)
    raw_titles.append(st.session_state.cached_data[i][2])

if len(touch_titles) > 0:
    selection = st.pills("Currently loaded touches:", touch_titles, default = touch_titles[-1])
    st.session_state.current_touch = touch_titles.index(selection)
    longtitle = make_longtitle_cache(st.session_state.current_touch)
    st.write(longtitle)
    selected_title = selection

    with st.expander("Upload more touches from device"):
        uploaded_files = st.file_uploader(
            "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
else:
    st.write('No touches are currently loaded. Load them from the library, upload a .csv from your device or from analysing a recording.')
    uploaded_files = st.file_uploader(
        "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")

dealwith_upload()
       
existing_names = find_existing_names()
#Open or create a method collection
st.write('Open an existing or create a new collection:')
            
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
        
url_collection = determine_collection_from_url(existing_names)

def determine_url_params():
    if 'current_collection_name' in st.session_state:
        if st.session_state.current_collection_name is not None:
            st.query_params.from_dict({"collection": [st.session_state.current_collection_name]})
    return
determine_url_params()   #I think disable this on this page at least 

if url_collection is not None:
    st.session_state.collection_status = 0
    st.session_state.current_collection_name = url_collection

cols = st.columns(2)
with cols[0]:
    if st.button('Open an existing collection', disabled = st.session_state.collection_status == 2):
        st.session_state.collection_status = 2   #2 is selecting a new name
        st.session_state.current_collection_name = None
        st.session_state.selected_library_touches = []
        st.session_state.input_key += 1
        st.query_params.clear()
        st.rerun()
with cols[1]:
    if st.button('Create a new collection', disabled = st.session_state.collection_status == 1):
        st.session_state.collection_status = 1 
        st.query_params.clear()        
        st.rerun()

if st.session_state.collection_status == 2:
    selected_name = st.text_input('Enter existing collection name:', key = st.session_state.input_key + 1000 )
    selected_name = re.sub(r'[^\w\-]', '_', selected_name)
    selected_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), selected_name)   

    if selected_name in existing_names:
        st.session_state.current_collection_name = selected_name
        st.session_state.selected_library_touches = []
        st.session_state.collection_status = 0       
        st.rerun()     
    elif len(selected_name) > 0:
        st.session_state.current_collection_name = selected_name
        st.session_state.selected_library_touches = []
        st.write("Can't find a collection with this name... Apologies")
        st.stop()
    else:
        st.stop()

new_collection_name = None    
#Create a new one
if st.session_state.collection_status == 1:
    st.write("Choose a name (a single word) for the new collection. If you'd like it not to be found by anyone else, choose something unguessable.")
    new_collection_name = st.text_input('New collection name (no capitals, spaces or special characters)', key = st.session_state.input_key)
    if new_collection_name is not None:
        new_collection_name = re.sub(r'[^\w\-]', '_', new_collection_name)
        new_collection_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), new_collection_name)

    if new_collection_name is not None:
        if new_collection_name not in existing_names and new_collection_name and len(new_collection_name) > 0:
            st.write('"%s" is valid and unused. Good good.' % new_collection_name)
            st.write("**Remember this name -- it can't be recovered if you forget it!**")
            if st.button('Create new collection called "%s"' % new_collection_name):
                st.write('New collection created with name "%s"' % new_collection_name)
                add_new_folder(new_collection_name)
                st.session_state.collection_status = 0
                st.session_state.current_collection_name = new_collection_name
                st.session_state.selected_library_touches = []                
                st.rerun()
            else:
                st.stop()
        elif new_collection_name in existing_names:
            st.write('This collection already exists... Click the relevant button above to open it.')
            st.stop()
        elif len(new_collection_name) > 0:
            st.write('Chosen name invalid for some reason.')
            st.stop()
        else:
            st.stop()

elif st.session_state.collection_status == 0:
    st.write('Viewing collection **%s**' % st.session_state.current_collection_name)  
    st.write('Share this collection with the url https://brenda.oekrice.com/Analyse_Striking?collection=%s' % st.session_state.current_collection_name)
else:
    st.stop()  

#I think the above logic is all sound. It's a bit tricky. Now to present a list of touches in said collection.  
if not os.path.exists("./saved_touches/%s/index.csv" % st.session_state.current_collection_name):  
    st.error("This collection is somehow corrupted... Apologies")
    st.stop()

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

def add_collection_to_cache(ntouches, saved_index_list):
    if ntouches!= 0 and saved_index_list[0][0] != ' ':
        if st.button('**Analyse touches from this collection**'):
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
            st.switch_page(page="pages/2_Analyse_Striking.py")
            st.rerun()

add_collection_to_cache(ntouches, saved_index_list)

st.divider()

with st.expander('Add/remove touches and change touch metadata'):

    if st.session_state.addition_mode < 0 and st.session_state.subtraction_mode < 0 and st.session_state.rename_mode < 0:
        if ntouches == 0 or saved_index_list[0][0] == ' ':
            disabled_flag = True
        else:
            disabled_flag = False
    
        cols = st.columns(3)    
        with cols [0]:
            if st.button("Add touches"):
                change_addition_mode()        
                st.rerun()
        with cols [1]:
            if st.button("Remove touches", disabled = disabled_flag):
                change_subtraction_mode()        
                st.rerun()
        with cols [2]:
            if st.button("Change touch data", disabled = disabled_flag):
                change_rename_mode()        
                st.rerun()
        with cols [0]:
            if st.button("Rename this collection"):
                change_rename_collection_mode()        
                st.rerun()
        with cols [1]:
            if st.button("Sort (oldest first)", disabled = disabled_flag):
                sort_current_collection(saved_index_list, oldfirst = True)
                st.rerun()
        with cols [2]:
            if st.button("Sort (newest first)", disabled = disabled_flag):
                sort_current_collection(saved_index_list, oldfirst = False)
                st.rerun()
    #Listitems will be ID (random string), readable ID (nonunique), nchanges, method(s), tower, datetime. All as strings.
    #This lists the loaded touches, hopefully with some metadata...            
    if st.session_state.addition_mode > 0: 
        #Replace these with long, metadataed titles at some point
        if len(touch_titles) == 0:
            st.write('No touches are loaded. Either analyse a recording, upload a csv. or load from another collection.')
            st.stop()
        else:
            st.write("**Choose touches to add:**")
            if saved_index_list[0][0] != ' ':
                current_entries = saved_index_list[:,0]
            else:
                current_entries = []
            select_list = []
            for ti, title in enumerate(touch_titles):
                longtitle = make_longtitle_cache(ti)
                if st.session_state.cached_touch_id[ti] not in current_entries:
                    checked = st.checkbox(label = longtitle)
                else:
                    checked = st.checkbox(label = longtitle, disabled = True)
                if checked:
                    select_list.append(ti)

            st.write("If the touch you want isn't in this list, either analyse the recording or upload a .csv")
            if len(select_list) > 0:
                text = "Add the selected touches"
            else:
                text = "Cancel"
            if st.button(text):
                change_addition_mode()
                current_ids = [listitem[0] for listitem in saved_index_list]
                if saved_index_list[0][0] != ' ':
                    new_index_list = saved_index_list.tolist()
                else:
                    new_index_list = []
                for ti in select_list:
                    #Figure out methods (just do it, it's cached...)
                    methods, hunt_types, calls, start_row, end_row, allrows_correct, quality = find_method_things(st.session_state.cached_rawdata[ti]["Bell No"])
                    nbells = np.max(st.session_state.cached_rawdata[ti]["Bell No"])
                    if len(methods) > 0:
                        nchanges = len(allrows_correct) - 1
                        end_row = int(np.ceil((start_row + len(allrows_correct))/2)*2)
                        if quality > 0.7:
                            if len(methods) == 1:   #Single method
                                method_title = methods[0][0]
                                if method_title.rsplit(' ')[0] == "Stedman" or method_title.rsplit(' ')[0] == "Erin":
                                    method_title = method_title.rsplit(' ')[0] + " " + method_title.rsplit(' ')[1]
                                    lead_length = 12
                                else:
                                    lead_length = 4*int(hunt_types[0][1] + 1)

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

                            st.session_state.cached_methods[ti] = method_title
                            st.session_state.cached_nchanges[ti] = nchanges

                        else:
                            st.session_state.cached_methods[ti] = "Unknown Method"
                            st.session_state.cached_nchanges[ti] = int(len(st.session_state.cached_rawdata[ti]["Bell No"])/nbells)
                    else:
                        st.session_state.cached_methods[ti] = "Rounds and/or Calls"
                        st.session_state.cached_nchanges[ti] = int(len(st.session_state.cached_rawdata[ti]["Bell No"])/nbells)

                    if st.session_state.cached_touch_id[ti] not in current_entries:

                        new_list_entry = [st.session_state.cached_touch_id[ti], st.session_state.cached_read_id[ti], st.session_state.cached_nchanges[ti], st.session_state.cached_methods[ti],st.session_state.cached_tower[ti], st.session_state.cached_datetime[ti], st.session_state.cached_score[ti], st.session_state.cached_ms[ti]]
                        new_index_list.append(new_list_entry)

                        @st.cache_data(ttl=300)
                        def convert_for_download(df):
                            return df.to_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,st.session_state.cached_touch_id[ti]))
                        convert_for_download(st.session_state.cached_rawdata[ti])

                np.savetxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, np.array(new_index_list, dtype = str), fmt = '%s', delimiter = ';')

                st.rerun()

    if st.session_state.subtraction_mode > 0: 
        #Replace these with long, metadataed titles at some point
        st.write("**Choose touches to remove:**")

        select_list = []
        for ti, info in enumerate(saved_index_list):
            longtitle = make_longtitle_collection(info)

            checked = st.checkbox(label = longtitle)

            if checked:
                select_list.append(ti)

        if len(select_list) > 0:
            text = "Remove the selected touches"
        else:
            text = "Cancel"

        if st.button(text):
            change_subtraction_mode()
            current_ids = [listitem[0] for listitem in saved_index_list]
            if saved_index_list[0][0] != ' ':
                new_index_list = saved_index_list.tolist()
            else:
                new_index_list = []
            for ti in select_list[::-1]:
                new_index_list.pop(ti)
            if len(new_index_list) == 0:
                new_index_list = np.array([[' ',' ',' ',' ',' ',' ',' ',' ']])

            np.savetxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, np.array(new_index_list, dtype = str), fmt = '%s', delimiter = ';')

            st.rerun()

    if st.session_state.rename_mode > 0: 
        #Replace these with long, metadataed titles at some point
        st.write("**Choose which touch data to alter:**")

        #Change title
        select_options = []
        for ti, info in enumerate(saved_index_list):
            longtitle = make_longtitle_collection(info)
            select_options.append(longtitle)
            
        rename_touch = st.radio('Select touch', options = select_options)
        new_name = st.text_input('Enter new name', value = saved_index_list[select_options.index(rename_touch),1])

        #Change time
        date_object = datetime.strptime(saved_index_list[select_options.index(rename_touch),5], "%m/%d/%Y, %H:%M:%S")
        new_proper_time = st.text_input('Enter time', date_object.strftime("%d-%m-%Y %H:%M:%S"))
        try:
            new_date_object = datetime.strptime(new_proper_time, "%d-%m-%Y %H:%M:%S")
        except:
            st.error('Incorrect date format, or incorrect date. Try again. Format is DD-MM-YYY HH:MM:SS')
            st.stop()

        @st.cache_data(ttl=300)               
        def read_bell_data():
            nominal_import = pd.read_csv('./bell_data/nominal_data.csv')
            return nominal_import
        #Chenge tower

        #Determine if this tower is in the list? Nah, just have a string!
        new_tower_name = st.text_input('Enter new tower name', saved_index_list[select_options.index(rename_touch),4])

        cols = st.columns(2)
        with cols[0]:
            if st.button("Apply Changes"):
                saved_index_list[select_options.index(rename_touch),1] = new_name
                saved_index_list[select_options.index(rename_touch),4] = new_tower_name
                saved_index_list[select_options.index(rename_touch),5] = new_date_object.strftime("%m/%d/%Y, %H:%M:%S")
                change_rename_mode()
                np.savetxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, np.array(saved_index_list, dtype = str), fmt = '%s', delimiter = ';')
                st.rerun()

        with cols[1]:
            if st.button("Cancel"):
                change_rename_mode()
                st.rerun()

    if st.session_state.rename_collection_mode > 0: 
        #Replace these with long, metadataed titles at some point

        new_name = st.text_input('Enter new collection name')

        cols = st.columns(2)
        with cols[0]:
            if len(new_name) < 1:
                if st.button('Rename collection', disabled = True):
                    pass
            else:
                #Check if it already exists
                if new_name in existing_names:
                    st.write('A collection with this name already exists...')
                    if st.button('Rename collection to "%s"' % new_name, disabled = True):
                        pass
                else:
                    os.system("mv ./saved_touches/%s ./saved_touches/%s" % (st.session_state.current_collection_name, new_name))
                    st.session_state.current_collection_name = new_name
                    change_rename_collection_mode()
                    st.rerun()

        with cols[1]:
            if st.button("Cancel"):
                change_rename_collection_mode()
                st.rerun()


if ntouches == 0 or saved_index_list[0][0] == ' ':
    st.write('No touches currently in this collection. Will need to add some.')
    st.stop()
elif ntouches == 1:
    st.write('One touch currently in this collection:')
else:
    st.write('%d touches currently in this collection:' % ntouches)

for ti, info in enumerate(saved_index_list[::-1]):
    longtitle = make_longtitle_collection(info)
    st.write(longtitle)
            
            
            
    
