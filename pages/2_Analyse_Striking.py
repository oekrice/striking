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
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
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
                    #st.write(strike_data)
        os.system('rm -r ./tmp/%s' % uploaded_file.name)
           
    if len(uploaded_files) > 0:
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

st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("## Analyse Striking")

st.write(
    """
    This page is for analysing the striking from the strike times either generated with the "Analyse Recording" tab, from the Touch Library or from an uploaded .csv file. \\
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


#Remove the large things from memory -- need a condition on this maybe?
# st.session_state.trimmed_signal = None
# st.session_state.audio_signal = None
# st.session_state.raw_file = None
Paras = None
Data = None

touch_titles = []
raw_titles = []
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = st.session_state.cached_read_id[i]
    touch_titles.append(title)
    raw_titles.append(st.session_state.cached_data[i][2])

if len(touch_titles) > 0:
    selection = st.pills("Choose a touch to analyse:", touch_titles, default = touch_titles[-1])

    with st.expander("Upload more touches from device"):
        uploaded_files = st.file_uploader(
            "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
else:
    uploaded_files = st.file_uploader(
        "Upload file:", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")
    
dealwith_upload()

if len(touch_titles) == 0:
    st.write('No data currently loaded: either upload a .csv file with striking data or generate some using the Analyse Recording page')
    st.stop()


if selection is None:
    st.stop()

st.session_state.current_touch = touch_titles.index(selection)
selected_title = selection

if len(touch_titles) > 0 and st.session_state.current_touch < 0:
      st.session_state.current_touch = len(touch_titles) - 1  

if st.session_state.current_touch < 0:
    st.write('**Select a touch from the options on the left, or upload a new one**')
else:
    st.write('Analysing ringing from "%s"' % touch_titles[st.session_state.current_touch])

if len(touch_titles) == 0:
    st.session_state.current_touch = -1
    
if st.session_state.current_touch >= 0:
    #Write in to a local bit to actually do the analysis
    strikes = st.session_state.cached_strikes[st.session_state.current_touch]
    certs = st.session_state.cached_certs[st.session_state.current_touch]
    
    available_models = []
    #If data is uploaded, treat it slightly differently to otherwise. Can just output various things immediately without calculation
    if len(strikes) == 0:
        #This is from a .csv
        raw_data = st.session_state.cached_rawdata[st.session_state.current_touch]
        cols = raw_data.columns.tolist()

        existing_models = [val for val in cols if val not in ["Bell No", "Confidence", "Actual Time"]]
        existing_models = [val for val in existing_models if val[:7] != "Unnamed"]
        existing_models = [val for val in existing_models if val[:17] != "Corrected Bells"]

        raw_actuals = raw_data["Actual Time"]
        
        nbells = np.max(raw_data["Bell No"])
        
        if "Confidence" not in  raw_data.columns.tolist():
            #st.write(len(raw_data['Actual Time'])//nbells)
            raw_data['Confidence'] = np.ones(len(raw_data["Actual Time"]))

    else:        
        #I think it would be easiest to just plonk this into a dataframe and treat it like an imported one, given I've already written code for that
        allbells = []
        allcerts_save = []
        allstrikes = []
        yvalues = np.arange(len(st.session_state.cached_strikes[st.session_state.current_touch][:,0])) + 1
        orders = []
        for row in range(len(st.session_state.cached_strikes[st.session_state.current_touch][0])):
            order = np.array([val for _, val in sorted(zip(st.session_state.cached_strikes[st.session_state.current_touch][:,row], yvalues), reverse = False)])
            certs = np.array([val for _, val in sorted(zip(st.session_state.cached_strikes[st.session_state.current_touch][:,row], st.session_state.cached_certs[st.session_state.current_touch][:,row]), reverse = False)])
            allstrikes = allstrikes + sorted((st.session_state.cached_strikes[st.session_state.current_touch][:,row]).tolist())
            allcerts_save = allcerts_save + certs.tolist()
            allbells = allbells + order.tolist()
            orders.append(order)

        allstrikes = 1000*np.array(allstrikes)*0.01
        allbells = np.array(allbells)
        allcerts_save = np.array(allcerts_save)
        orders = np.array(orders)

        raw_data = pd.DataFrame({'Bell No': allbells, 'Actual Time': allstrikes, 'Confidence': allcerts_save})

        raw_actuals = raw_data["Actual Time"]
        nbells = np.max(raw_data["Bell No"])

        existing_models = []
        
        #st.write(len(raw_data)//nbells)
    
    class striking_data():
        #Empty class with the raw striking data and errors etc.
        def __init__(self):
            return

    st.method_message = st.empty()
    st.method_message.write("Figuring out methods and composition...")

    methods, hunt_types, calls, start_row, end_row, allrows_correct, quality = find_method_things(raw_data["Bell No"])

    if len(methods) > 0:
        call_string, comp_html = print_composition(methods, hunt_types, calls, allrows_correct)
        method_flag = True
    else:
        method_flag = False

    composition_flag = False
    if len(methods) > 0:
        nchanges = len(allrows_correct) - 1
        end_row = int(np.ceil((start_row + len(allrows_correct))/2)*2)
        if quality > 0.7:
            composition_flag = True
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
            if quality > 0.85:
                st.method_message.write("**Method(s) detected: " + str(nchanges) + " " + method_title + "**")
            else:
                st.method_message.write("**Method(s) detected: " + str(nchanges) + " " + method_title + " (sort of)**")

            st.session_state.cached_methods[st.session_state.current_touch] = method_title
            st.session_state.cached_nchanges[st.session_state.current_touch] = nchanges

            if nchanges/int(len(raw_actuals)/nbells) < 0.25:
                start_row = 0; end_row = int(len(raw_actuals)/nbells)

        else:
            st.write("**Probably a method but not entirely sure what...**")
            st.session_state.cached_methods[st.session_state.current_touch] = "Unknown Method"
            st.session_state.cached_nchanges[st.session_state.current_touch] = int(len(raw_actuals)/nbells)
            method_flag = False
            lead_length = 24
            start_row = 0; end_row = int(len(raw_actuals)/nbells)
    else:
        st.method_message.write("**No method detected**")
        st.session_state.cached_methods[st.session_state.current_touch] = "Rounds and/or Calls"
        st.session_state.cached_nchanges[st.session_state.current_touch] = int(len(raw_actuals)/nbells)
        start_row = 0; end_row = int(len(raw_actuals)/nbells)
        lead_length = 24

    if "Team Model" not in  raw_data.columns.tolist():
        count_test = st.session_state.rhythm_variation_time*nbells; gap_test = st.session_state.handstroke_gap_variation_time
        ideal_times = find_ideal_times_band(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, key = st.session_state.current_touch)
        raw_data['Team Model'] = ideal_times
        existing_models.append('Team Model')
    
    ideal_times = find_ideal_times_rwp(raw_data['Actual Time'], nbells, key = st.session_state.current_touch)
    if "RWP Model" not in  raw_data.columns.tolist():
        #st.write(len(raw_data['Actual Time'])//nbells)
        ideal_times = find_ideal_times_rwp(raw_data['Actual Time'], nbells, key = st.session_state.current_touch)
        raw_data["RWP Model"] = ideal_times
        existing_models.append("RWP Model")

    if "Corrected Bells" not in  raw_data.columns.tolist():
        #Add the correct order the bells should be in, based on the method analysis
        corrected_order = raw_data["Bell No"].values.copy()  #The original order
        start_index = start_row*nbells
        corrected_order[start_index:start_index + np.size(allrows_correct)] = np.ravel(allrows_correct)
        raw_data["Corrected Bells"] = corrected_order
        
    if "Metronomic Model" not in  raw_data.columns.tolist():
        @st.cache_data(ttl=300)
        def find_metronomic(raw_data):
            nrows = int(len(raw_data['Actual Time'])//nbells)
            all_metros = []
            for row in range(nrows):
                actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
                start = np.min(actual)
                end = np.max(actual)
                metronomic_target = np.linspace(start, end, nbells)
                all_metros = all_metros + metronomic_target.tolist()

            return all_metros

        all_metros = find_metronomic(raw_data)
        if len(all_metros) == len(raw_data["Actual Time"]):  #Bodge for a bug.
            raw_data['Metronomic Model'] = all_metros
            existing_models.append('Metronomic Model')
    
    if "Individual Model" not in  raw_data.columns.tolist():
        count_test = st.session_state.rhythm_variation_time*nbells; gap_test = st.session_state.handstroke_gap_variation_time
        ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, key = st.session_state.current_touch)
        raw_data['Individual Model'] = ideal_times
        existing_models.append('Individual Model')

    if len(existing_models) > 0:
                
        with st.expander("Change Statistical Options"):

            selection = st.selectbox("Select striking model:", options = existing_models, index = existing_models.index("Team Model"))   #Can set default for this later?
            #st.write(raw_data["Actual Time"][0:100:12])
            raw_target = np.array(raw_data[selection])
            raw_bells = np.array(raw_data["Bell No"])
            correct_bells = np.array(raw_data["Corrected Bells"])
            #Plot blue line
            nstrikes = len(raw_actuals)
            nrows = int(nstrikes//nbells)

            if selection == "Individual Model":
                rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 6, format = "%d changes", step = 2, key = 100 + st.session_state.current_touch)
                st.session_state.rhythm_variation_time = rhythm_variation_time
                ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = st.session_state.rhythm_variation_time*nbells, ngaps = st.session_state.handstroke_gap_variation_time)
                if "Individual Model" not in  raw_data.columns.tolist():
                    existing_models.append('Individual Model')

                raw_data['Individual Model'] = ideal_times

            if selection == "Team Model":

                rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 6, format = "%d changes", step = 2, key = 200 + st.session_state.current_touch)
                st.session_state.rhythm_variation_time = rhythm_variation_time
                ideal_times = find_ideal_times_band(raw_data['Actual Time'], nbells, ncount = st.session_state.rhythm_variation_time*nbells, ngaps = st.session_state.handstroke_gap_variation_time)
                if "Team Model" not in  raw_data.columns.tolist():
                    existing_models.append('Team Model')
                
                raw_data['Team Model'] = ideal_times
                
            remove_mistakes = st.checkbox("Remove presumed method mistakes from the stats?", value = True)
            remove_confidence = st.checkbox("Remove not-confident strike times from the stats?", value = True)
            if method_flag:
                use_method_info = st.checkbox("Use presumed composition to identify correct times?", value = True)
            else:
                use_method_info = False
            min_include_change, max_include_change = st.slider("For the stats, include changes in range:", min_value = 0, max_value = nrows, value=(start_row+1, end_row), format = "%d", step = 2, key = 300 + st.session_state.current_touch)

        Strike_Data = striking_data()  #Initialise empty thing
        Strike_Data.method_flag = method_flag
        Strike_Data.use_method_info = use_method_info
        Strike_Data.remove_confidence = remove_confidence
        Strike_Data.remove_mistakes= remove_mistakes
        Strike_Data.nbells = nbells
        Strike_Data.raw_data = raw_data
        Strike_Data.raw_actuals = raw_actuals
        Strike_Data.raw_target = raw_target
        Strike_Data.raw_bells = raw_bells
        Strike_Data.correct_bells = correct_bells
        Strike_Data.min_include_change = min_include_change
        Strike_Data.max_include_change = max_include_change

        Strike_Data = calculate_stats(Strike_Data)

        st.message = st.empty()
        st.message_2 = st.empty()
        st.message.write("Calculating stats and things...")
        #Blue Line
        st.blueline = st.empty()
        
        titles = ['All blows', 'Handstrokes', 'Backstrokes']

        overall_quality = 1.0 - np.mean(Strike_Data.alldiags[2,2,:])/Strike_Data.cadence

        k = 17.5; x0 = 0.727
        shifted_quality = 1.0/(1.0 + np.exp(-k*(overall_quality - x0)))
        st.message.write("Standard deviation from ideal for this touch: %.1fms" % np.mean(Strike_Data.alldiags[2,2,:]))
        st.message_2.write("Overall striking quality: **%.2f%%**" % (100*shifted_quality))

        if composition_flag:
            with st.expander("View Composition"):
                st.html(comp_html)

        with st.expander('View Plaintext Striking Report'):
            st.empty()
            obtain_striking_markdown(Strike_Data.alldiags, Strike_Data.time_errors, Strike_Data.lead_times, Strike_Data.cadence, Strike_Data.remove_mistakes)

         
        with st.expander("View Grid/Blue Line"):
            st.empty()

            min_plot_change, max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = nrows, value=(start_row, min(300, end_row)), format = "%d", step = 2, key = 400 + st.session_state.current_touch)
            view_numbers = st.checkbox("View Bell Numbers", value = False)
            options = ["Bell %d" % bell for bell in range(1,nbells+1)]
            highlight_bells = st.pills("Highlight Bells", options, selection_mode="multi")
            plot_blue_line(raw_target, raw_actuals, Strike_Data.raw_bells, nbells, lead_length, min_plot_change, max_plot_change, highlight_bells, view_numbers = view_numbers)

        #Bar Chart
        with st.expander("View Error Bar Charts"):
            st.empty()
            plot_bar_charts(Strike_Data.alldiags, Strike_Data.nbells, titles)
                        
        with st.expander("View Bell Errors Through Time"):
            st.empty()
            min_plot_change, max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = nrows, value=(0, nrows), format = "%d", step = 2, key = 500 + st.session_state.current_touch)
            
            absvalues = st.radio("Use absolute values?", ["Absolute Error", "Relative Error"])
            options = ["Average"] + ["Bell %d" % bell for bell in range(1,nbells+1)]
            highlight_bells = st.pills("Plot Bells", options, default = ["Average"], selection_mode="multi", key = 600 + st.session_state.current_touch)
            
            smooth = st.checkbox("Smooth data?", value = False)
                    
            strokes = ["Both Strokes", "Handstrokes", "Backstrokes"]
            if len(highlight_bells) == 1:
                strokes_plot = st.pills("Select Strokes", strokes, default = "Both Strokes", selection_mode="multi", key = 700 + st.session_state.current_touch)
            elif len(highlight_bells) > 1:
                strokes_plot = st.pills("Select Strokes", strokes, default = "Both Strokes", selection_mode="single", key = 800 + st.session_state.current_touch)
                strokes_plot = [strokes_plot]
            else:
                strokes_plot = None
                
            if len(highlight_bells) > 0 and strokes_plot is not None:
                    plot_errors_time(Strike_Data.time_errors, min_plot_change, max_plot_change, absvalues, highlight_bells, strokes_plot, smooth)
        
        with st.expander("View Histograms"):
            st.empty()
            x_range = st.slider("Histogram x range:", min_value = 50, max_value = 250, value= 160, format = "%dms")
            nbins_default = min(100, max(int(len(Strike_Data.errors)/2.5),10))
            nbins = st.slider("Number of histogram bins", min_value = 10, max_value = 100, value= nbins_default, format = "%d", step = 1)
            plot_histograms(Strike_Data.errors, x_range, nbins, nbells, Strike_Data.raw_bells, Strike_Data.correct_bells, Strike_Data.min_include_change, Strike_Data.max_include_change, Strike_Data.use_method_info, Strike_Data.remove_mistakes, Strike_Data.cadence, Strike_Data.raw_actuals, Strike_Data.raw_target, titles)

        with st.expander("View Box Plots"):
            st.empty()
            plot_boxes(Strike_Data.time_errors, nbells, titles)

        @st.cache_data(ttl=300)
        def convert_for_download(df):
            return df.to_csv().encode("utf-8")

        csv = convert_for_download(raw_data)
        if st.session_state.current_touch > -1 and len(raw_titles) > 0:
            st.download_button("Download analysis to device as .csv", csv, file_name = raw_titles[st.session_state.current_touch] + '.csv', mime="text/csv")
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
