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

from strike_model import find_ideal_times
from strike_model_band import find_ideal_times_band
from methods import find_method_things, print_composition

from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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
        st.rerun()
    return
    
if not os.path.exists('./tmp/'):
    os.system('mkdir ./tmp/')
if not os.path.exists('./frequency_data/'):
    os.system('mkdir ./frequency_data/')
if not os.path.exists('./striking_data/'):
    os.system('mkdir ./striking_data/')

st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("## Analyse Striking")

st.write(
    """This page is for analysing the striking from the strike times either generated with the "Analyse Audio" tab or from an uploaded .csv."""
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
    st.session_state.handstroke_gap_variation_time = 10
    
#Remove the large things from memory -- need a condition on this!
st.session_state.trimmed_signal = None
st.session_state.audio_signal = None
Paras = None
Data = None

touch_titles = []
raw_titles = []
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = '' + st.session_state.cached_data[i][2] + ''# + ': ' + str(st.session_state.cached_data[i][1]) + ' changes'
    touch_titles.append(title)
    raw_titles.append(st.session_state.cached_data[i][2])

uploaded_files = st.file_uploader(
    "Upload timing data from device", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}", type = "csv")

dealwith_upload()

if len(touch_titles) == 0:
    st.write('No data currently loaded: either upload a .csv file with striking data or generate some using the Analyse Audio page')
    st.stop()

selection = st.pills("Choose a touch to analyse:", touch_titles, default = touch_titles[-1])

if selection is None:
    st.stop()

st.session_state.current_touch = touch_titles.index(selection)
selected_title = selection

if len(touch_titles) > 0 and st.session_state.current_touch < 0:
      st.session_state.current_touch = len(touch_titles) - 1  

if st.session_state.current_touch < 0:
    st.write('**Select a touch from the options on the left, or upload a new one**')
else:
    st.write('Analysing ringing from file "%s"' % touch_titles[st.session_state.current_touch])

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
    
    st.method_message = st.empty()
    st.method_message.write("Figuring out methods and composition...")

    methods, hunt_types, calls, start_row, end_row, allrows_correct, quality = find_method_things(raw_data["Bell No"])

    if len(methods) > 0:
        call_string, comp_html = print_composition(methods, hunt_types, calls, allrows_correct)
        method_flag = True
    else:
        method_flag = False

    if len(methods) > 0:
        nchanges = len(allrows_correct) - 1
        end_row = int(np.ceil((start_row + len(allrows_correct))/2)*2)
        if quality > 0.8:

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
            st.method_message.write("**Method(s) detected: " + str(nchanges) + " " + method_title + "**")
            with st.expander("View Composition"):
                st.html(comp_html)
        else:
            st.write("**Probably a method but not entirely sure what...**")
            method_flag = False
            lead_length = 24
            start_row = 0; end_row = len(allrows_correct)
    else:
        st.method_message.write("**No method detected**")
        start_row = 0; end_row = len(allrows_correct)
        lead_length = 24

    if "Individual Model" not in  raw_data.columns.tolist():
        #st.write(len(raw_data['Actual Time'])//nbells)
        count_test = st.session_state.rhythm_variation_time*nbells; gap_test = st.session_state.handstroke_gap_variation_time
        ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, key = st.session_state.current_touch)
        raw_data['Individual Model'] = ideal_times
        existing_models.append('Individual Model')

    if "Team Model" not in  raw_data.columns.tolist():
        #st.write(len(raw_data['Actual Time'])//nbells)
        count_test = st.session_state.rhythm_variation_time*nbells; gap_test = st.session_state.handstroke_gap_variation_time
        ideal_times = find_ideal_times_band(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, key = st.session_state.current_touch)
        raw_data['Team Model'] = ideal_times
        existing_models.append('Team Model')
    
    if "Corrected Bells" not in  raw_data.columns.tolist():
        #Add the correct order the bells should be in, based on the method analysis
        corrected_order = raw_data["Bell No"].values.copy()  #The original order
        start_index = start_row*nbells
        corrected_order[start_index:start_index + np.size(allrows_correct)] = np.ravel(allrows_correct)
        raw_data["Corrected Bells"] = corrected_order
        
    if "Metronomic Model" not in  raw_data.columns.tolist():
        @st.cache_data
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
    
    if len(existing_models) > 0:
                
        selection = st.selectbox("Select striking model:", options = existing_models, index = existing_models.index("Team Model"))   #Can set default for this later?
        
        #st.write(raw_data["Actual Time"][0:100:12])
        raw_target = np.array(raw_data[selection])
        raw_bells = np.array(raw_data["Bell No"])
        correct_bells = np.array(raw_data["Corrected Bells"])
        #Plot blue line
        nstrikes = len(raw_actuals)
        nrows = int(nstrikes//nbells)
    
        with st.expander("Statistical Options"):
            if selection == "Individual Model":
                rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 10, format = "%d changes", step = 2, key = 100 + st.session_state.current_touch)
                st.session_state.rhythm_variation_time = rhythm_variation_time
                ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = st.session_state.rhythm_variation_time*nbells, ngaps = st.session_state.handstroke_gap_variation_time)
                if "Individual Model" not in  raw_data.columns.tolist():
                    existing_models.append('Individual Model')

                raw_data['Individual Model'] = ideal_times

            if selection == "Team Model":

                rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
                st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 10, format = "%d changes", step = 2, key = 200 + st.session_state.current_touch)
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

        
        st.message = st.empty()
        st.message_2 = st.empty()
        st.message.write("Calculating stats things...")
        #Blue Line
        st.blueline = st.empty()
        
        #Do stats now
        alldiags = np.zeros((3,3,nbells))   #Type, stroke, bell
        allerrors = []
        diffs = np.array(raw_actuals)[1:] - np.array(raw_actuals)[:-1]
        cadence = np.mean(diffs)*(2*nbells)/(2*nbells + 1)   #Mean interbell gap (ish)

        time_errors = np.zeros((nbells, int(len(np.array(raw_actuals))/nbells)))   #Errors through time for the whole touch

        for plot_id in range(3):
            for bell in range(1,nbells+1):#nbells):
                
                bellstrikes = np.where(raw_bells == bell)[0]
                targetstrikes = np.where(correct_bells == bell)[0]

                if use_method_info:
                    errors = np.array(raw_actuals[bellstrikes] - raw_target[targetstrikes])
                else:
                    errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])

                confs = np.array(raw_data["Confidence"][bellstrikes])
                
                time_errors[bell-1, :] = errors
                if remove_confidence:
                    time_errors[bell-1, confs < 0.9] = 0.0

                bellstrikes = bellstrikes[bellstrikes/nbells >= min_include_change]
                bellstrikes = bellstrikes[bellstrikes/nbells <= max_include_change]
                   
                if len(bellstrikes) < 2:
                    st.error('Increase range -- stats are all wrong')
                    st.stop()
                    
                errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])
                confs = np.array(raw_data["Confidence"][bellstrikes])
                
                #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
                maxlim = cadence*0.75
                minlim = -cadence*0.75
    
                #Trim for the appropriate stroke
                if plot_id == 1:
                    errors = errors[::2]
                    confs = confs[::2]
                if plot_id == 2:
                    errors = errors[1::2]
                    confs = confs[1::2]
                count = len(errors)
    
                if remove_confidence:
                    errors[confs < 0.9] = 0.0
                    count -= np.sum(confs < 0.9)

                                        
                if remove_mistakes:
                    #Adjust stats to disregard these properly
                    count -= np.sum(errors > maxlim)
                    count -= np.sum(errors < minlim)
    
                    errors[errors > maxlim] = 0.0
                    errors[errors < minlim] = 0.0

                if count > 2:
                #Diagnostics
                    alldiags[0,plot_id,bell-1] = np.sum(errors)/count
                    alldiags[1,plot_id,bell-1] = np.sqrt(np.sum((errors-np.sum(errors)/count)**2)/count)
                    alldiags[2,plot_id,bell-1] = np.sqrt(np.sum(errors**2)/count)
            
        titles = ['All blows', 'Handstrokes', 'Backstrokes']

        overall_quality = 1.0 - np.mean(alldiags[2,2,:])/cadence

        k = 17.5; x0 = 0.727
        shifted_quality = 1.0/(1.0 + np.exp(-k*(overall_quality - x0)))
        st.message.write("Standard deviation from ideal for this touch: %.1fms" % np.mean(alldiags[2,2,:]))
        st.message_2.write("Overall striking quality: **%.2f%%**" % (100*shifted_quality))

        #Want error through time for each bell, and do as a snazzy plot?
        @st.cache_data
        def plot_errors_time(time_errors, min_plot_change, max_plot_change, absvalues, highlight_bells, strokes_plot, smooth):
            #Just do both strokes to start with because easy
            fig5, ax5 = plt.subplots(1, figsize = (10,5))
            all_xs = np.arange(min_plot_change, max_plot_change)
            hand_xs = np.arange(min_plot_change, max_plot_change, 2)
            back_xs = np.arange(min_plot_change + 1, max_plot_change, 2)
            #Just calculate everything. Doesn't take long...
            if absvalues == "Absolute Error":
                time_errors = np.abs(time_errors[:,min_plot_change:max_plot_change])
            else:  
                time_errors = time_errors[:,min_plot_change:max_plot_change]
            
            all_ys = time_errors[:,:]
            hand_ys = time_errors[:,::2]
            back_ys = time_errors[:,1::2]
            
            if smooth:
                all_ys = gaussian_filter1d(all_ys, 6, axis = 1)
                hand_ys = gaussian_filter1d(hand_ys, 3, axis = 1)
                back_ys = gaussian_filter1d(back_ys, 3, axis = 1)
                
            avg_ys = np.mean(all_ys, axis = 0)
            hand_avg_ys = np.mean(hand_ys, axis = 0)
            back_avg_ys = np.mean(back_ys, axis = 0)
            
            if len(strokes_plot) > 1:
                mode = 2  #Multiple strokes, one bell
            else:
                mode = 1  #Multiple bells, one stroke
            
            for bell in highlight_bells:
                for stroke in strokes_plot:
                    if bell == "Average":
                        if stroke == "Both Strokes":
                            if mode == 2:
                                ax5.plot(all_xs, avg_ys, c = 'gray', label = 'Both Strokes')
                            else:
                                ax5.plot(all_xs, avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
                        if stroke == "Handstrokes":
                            if mode == 2:
                                ax5.plot(hand_xs, hand_avg_ys, c = 'red', label = 'Handstrokes')
                            else:
                                ax5.plot(hand_xs, hand_avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
                        if stroke == "Backstrokes":
                            if mode == 2:
                                ax5.plot(back_xs, back_avg_ys, c = 'green', label = 'Backstrokes')
                            else:
                                ax5.plot(back_xs, back_avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
                    else:
                        if stroke == "Both Strokes":
                            if mode == 2:
                                ax5.plot(all_xs, all_ys[int(bell[5:])-1], c = 'gray', label = 'Both Strokes')
                            else:
                                ax5.plot(all_xs, all_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
                        if stroke == "Handstrokes":
                            if mode == 2:
                                ax5.plot(hand_xs, hand_ys[int(bell[5:])-1], c = 'red', label = 'Handstrokes')
                            else:
                                ax5.plot(hand_xs, hand_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
                        if stroke == "Backstrokes":
                            if mode == 2:
                                ax5.plot(back_xs, back_ys[int(bell[5:])-1], c = 'green', label = 'Backstrokes')
                            else:
                                ax5.plot(back_xs, back_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
            
            plt.xlabel('Change Number')
            plt.ylabel('Error (ms)')
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig5)
            plt.clf()
            plt.close()

            return

        @st.cache_data
        def plot_blue_line(raw_target_plot, min_plot_change, max_plot_change, highlight_bells, view_numbers = False):
            
            bell_names = ['1','2','3','4','5','6','7','8','9','0','E','T','A','B','C','D']
            highlight_bells = [int(highlight_bells[val][5:]) for val in range(len(highlight_bells))]
            raw_actuals = np.array(raw_data["Actual Time"])
    
            toprint = []
            orders = []; starts = []; ends = []
            for row in range(nrows):
                actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
                target = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                bells =  np.array(raw_bells[row*nbells:(row+1)*nbells])  
                toprint.append(actual-target)
                orders.append(bells)
                starts.append(np.min(target))
                ends.append(np.max(target))
    
            nrows_plot = max_plot_change - min_plot_change
            rows_per_plot = 61#6*int(nrows_plot//24)
            if nbells < 9:
                nplotsk = max(3, nrows_plot//rows_per_plot + 1)
            else:
                nplotsk = max(2, nrows_plot//rows_per_plot + 1)
            nplotsk = min(nplotsk, 3)
            min_rows_per_plot = int(nrows_plot/nplotsk) + 2
            rows_per_plot = int(np.ceil(min_rows_per_plot / lead_length) * lead_length)

            nplotsk = int(np.ceil((nrows_plot-1)/rows_per_plot))
            #Either plot lines or nmbers -- but guides will matter for that...
            dotext = True

            #fig,axs = plt.subplots(1,ncols, figsize = (15,4*nrows/(nbells + 4)))
            fig1, axs1 = plt.subplots(1,nplotsk, figsize = (10, 100))
            for plot in range(nplotsk):
                if dotext:
                    xtexts = []; ytexts = []; stexts = []
                
                if nplotsk > 1:
                    ax = axs1[plot]
                else:
                    ax = axs1
                maxrow = max_plot_change
                for bell in range(1,nbells+1):#nbells):

                    if bell in highlight_bells:
                        points = []  ; changes = []
                        bellstrikes = np.where(raw_bells == bell)[0]
                        for row in range(plot*rows_per_plot+ min_plot_change, min((plot+1)*rows_per_plot + min_plot_change + 1, maxrow)):
                            #Find linear position... Linear interpolate?
                            target_row = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                            if len(target_row) == nbells:
                                ys = np.arange(1,nbells+1)
                                f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                                rat = float(f(raw_actuals[bellstrikes][row]))
                                points.append(rat); changes.append(row)
                                
                                if view_numbers:
                                    ax.text(rat, row, bell_names[bell-1], horizontalalignment = 'center', verticalalignment = 'center')

                        if len(highlight_bells) > 0:
                            ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10], linewidth = 2)
                        else:
                            ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10])
                    else:
                        points = []  ; changes = []
                        bellstrikes = np.where(raw_bells == bell)[0]
                        for row in range(plot*rows_per_plot+ min_plot_change, min((plot+1)*rows_per_plot + min_plot_change + 1, maxrow)):
                            #Find linear position... Linear interpolate?
                            target_row = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                            if len(target_row) == nbells:
                                ys = np.arange(1,nbells+1)
                                f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                                rat = float(f(raw_actuals[bellstrikes][row]))
                                points.append(rat); changes.append(row)
                                if view_numbers:
                                    ax.text(rat, row, bell_names[bell-1], horizontalalignment = 'center', verticalalignment = 'center')
                                    
                        if len(highlight_bells) > 0:
                            ax.plot(points, changes,label = bell, c = 'grey', linewidth  = 0.5)
                        else:
                            if view_numbers:
                                ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10], linewidth = 0.5)
                            else:
                                ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10])

                        
                    ax.plot((bell)*np.ones(len(points)), changes, c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
                    
                row_guides = [np.column_stack([np.arange(-1,nbells+3), row*np.ones(nbells+4)]) for row in range(plot*rows_per_plot+ min_plot_change, (plot+1)*rows_per_plot + min_plot_change + 1)]
                #for row in range(min_plot_change, max_plot_change):
                #    ax.plot(np.arange(-1,nbells+3), row*np.ones(nbells+4), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
                line_collection = LineCollection(row_guides, color = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
                ax.add_collection(line_collection)
                
                plt.gca().invert_yaxis()
                ax.set_ylim((plot+1)*rows_per_plot + min_plot_change, plot*rows_per_plot+ min_plot_change )
                ax.set_xlim(-1,nbells+2)
                ax.set_xticks([])
                ax.set_aspect('equal')
                #if plot == nplotsk-1:
                #    plt.legend()
                #ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig1)
            plt.clf()
            plt.close()
         
        with st.expander("View Grid/Blue Line"):

            min_plot_change, max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = nrows, value=(start_row, min(300, end_row)), format = "%d", step = 2, key = 400 + st.session_state.current_touch)
            view_numbers = st.checkbox("View Bell Numbers", value = False)
            options = ["Bell %d" % bell for bell in range(1,nbells+1)]
            highlight_bells = st.pills("Highlight Bells", options, selection_mode="multi")
            plot_blue_line(raw_target, min_plot_change, max_plot_change, highlight_bells, view_numbers = view_numbers)

        with st.expander("View Bell Errors Through Time"):

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
                    plot_errors_time(time_errors, min_plot_change, max_plot_change, absvalues, highlight_bells, strokes_plot, smooth)
        
        #Bar Chart
        with st.expander("View Error Bar Charts"):
            @st.cache_data
            def plot_bar_charts(alldiags):
                fig3, axs3 = plt.subplots(3, figsize = (12,7))
                bar_width = 0.3
        
                data_titles = ['Avg. Error', 'Std. Dev. from Average', 'Std. Dev. From Ideal']
        
                x = np.arange(nbells)
                for plot_id in range(3):
                    ax = axs3[plot_id]
        
                    xmin = np.min(alldiags[plot_id,:,:])*0.9
                    xmax = np.max(alldiags[plot_id,:,:])*1.1
                    
        
                    rects0 = ax.bar(x-bar_width*1,alldiags[plot_id,0,:],bar_width,label = titles[0], color='lightgray')
                    ax.bar_label(rects0, padding = 3, fmt = '%d')
        
                    rects1 = ax.bar(x-bar_width*0,alldiags[plot_id,1,:],bar_width,label = titles[1], color='red')
                    ax.bar_label(rects1, padding = 3, fmt = '%d')
        
                    rects2 = ax.bar(x+bar_width*1.0,alldiags[plot_id,2,:],bar_width,label = titles[2], color='blue')
                    ax.bar_label(rects2, padding = 3, fmt = '%d')
        
                    ax.set_xticks(np.arange(nbells), np.arange(1,nbells+1))
                    ax.set_title(data_titles[plot_id])
                    if plot_id > 0:
                        ax.set_ylim(xmin, xmax)
                    if plot_id == 0:
                        ax.legend()
        
                plt.tight_layout()
                st.pyplot(fig3)
                plt.clf()
                plt.close()
            plot_bar_charts(alldiags)            

        with st.expander("View Histograms"):

            @st.cache_data
            def plot_histograms(errors, x_range, nbins):

                for plot_id in range(3):
                    #Everything, then handstrokes, then backstrokes
                    nrows = max(2,(nbells)//3)
                    ncols = int((nbells-1e-6)/nrows) + 1
                    if nrows*ncols < nbells:
                        ncols += 1
                    fig2, axs2 = plt.subplots(nrows,ncols, figsize = (10,7))
                    for bell in range(1,nbells+1):#nbells):
                        #Extract data for this bell
                        bellstrikes = np.where(raw_bells == bell)[0]

                        bellstrikes = bellstrikes[bellstrikes/nbells >= min_include_change]
                        bellstrikes = bellstrikes[bellstrikes/nbells <= max_include_change]

                        if len(bellstrikes) < 2:
                            st.error('Increase range -- stats are all wrong')
                            st.stop()

                        errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])

                        #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
                        maxlim = cadence*0.75
                        minlim = -cadence*0.75

                        #Trim for the appropriate stroke
                        if plot_id == 1:
                            errors = errors[::2]
                        if plot_id == 2:
                            errors = errors[1::2]

                        count = len(errors)

                        if remove_mistakes:
                            #Adjust stats to disregard these properly
                            count -= np.sum(errors > maxlim)
                            count -= np.sum(errors < minlim)

                            errors[errors > maxlim] = 0.0
                            errors[errors < minlim] = 0.0

                        ax = axs2[(bell-1)//ncols, (bell-1)%ncols]

                        ax.set_title('Bell %d' % bell)
                        bin_bounds = np.linspace(-x_range, x_range, nbins+1)
                        n, bins, _ = ax.hist(errors, bins = bin_bounds)

                        curve = gaussian_filter1d(n, sigma = nbins/20)
                        ax.plot(0.5*(bins[1:] + bins[:-1]),curve, c= 'black')
                        ax.set_xlim(-x_range, x_range)
                        ax.set_ylim(0,np.max(n))
                        ax.plot([0,0],[0,max(n)], linewidth = 2)
                        ax.set_yticks([])
                    plt.suptitle(titles[plot_id])
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.clf()
                    plt.close()

            x_range = st.slider("Histogram x range:", min_value = 50, max_value = 250, value= 160, format = "%dms")
            nbins_default = min(100, max(int(len(errors)/5),10))
            nbins = st.slider("Number of histogram bins", min_value = 10, max_value = 100, value= nbins_default, format = "%d", step = 1)
            plot_histograms(errors,x_range,nbins)

        with st.expander("View Box Plots"):
            
            @st.cache_data
            def plot_boxes(raw_data):
                fig4, axs4 = plt.subplots(3, figsize = (10,7))

                for plot_id in range(3):
                    #Everything, then handstrokes, then backstrokes

                    ax = axs4[plot_id]

                    
                    for bell in range(1,nbells+1):#nbells):
                        #Extract data for this bell
                        belldata = raw_data.loc[raw_data['Bell No'] == bell]

                        errors = np.array(belldata['Actual Time'] - belldata[selection])

                        #Trim for the appropriate stroke
                        if plot_id == 1:
                            errors = errors[::2]
                        if plot_id == 2:
                            errors = errors[1::2]

                        #Box plot data
                        ax.boxplot(errors,positions = [bell], sym = 'x', widths = 0.35, zorder = 1)
                    #ax.axhline(0.0, c = 'black', linestyle = 'dashed')

                    ax.set_ylim(-150,150)
                    ax.set_title(titles[plot_id])

                plt.tight_layout()
                st.pyplot(fig4)
                plt.clf()
                plt.close()
            
            plot_boxes(raw_data)

        @st.cache_data
        def convert_for_download(df):
            return df.to_csv().encode("utf-8")

        csv = convert_for_download(raw_data)
        if st.session_state.current_touch > -1 and len(raw_titles) > 0:
            st.download_button("Download analysis to device as .csv", csv, file_name = raw_titles[st.session_state.current_touch] + '.csv', mime="text/csv")
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
