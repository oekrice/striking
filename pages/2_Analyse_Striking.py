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
import numpy as np
import pandas as pd

from strike_model import find_ideal_times
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
            st.write(uploaded_file.name)
            with open('./tmp/%s' % uploaded_file.name, 'wb') as f: 
                f.write(uploaded_file.getvalue())        
            try:
                raw_data = pd.read_csv('./tmp/%s' % uploaded_file.name)
                #Convert this into rows? Nah. Do it in a bit, if at all
                #st.session_state.selected_data = raw_data
                #Present as an option on the side.
                if "Bell No" not in raw_data.columns or "Actual Time" not in raw_data.columns:
                    isfine = False
                st.write(raw_data.columns)
                strike_data = ["Unknown Tower", int(len(raw_data)/np.max(raw_data["Bell No"]))]
            except:
                st.error('Cannot interpret %s as readable data' % uploaded_file.name)
                uploaded_files.pop(uploaded_files.index(uploaded_file))
                isfine = False
            
            st.write(isfine, "Bell No" not in raw_data.columns, "Actual Time " not in raw_data.columns)
            if isfine:
                if strike_data not in st.session_state.cached_data:
                    st.session_state.cached_data.append(strike_data)
                    st.session_state.cached_strikes.append([])
                    st.session_state.cached_certs.append([])
                    st.session_state.cached_rawdata.append(raw_data)
                    st.write(strike_data)
    if len(uploaded_files) > 0:
        os.system('rm -r ./tmp/%s' % uploaded_file.name)
        st.session_state.uploader_key += 1
        st.rerun()
    return
    
st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("# Analyse Striking")
st.sidebar.header("Choose a Touch:")
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
    st.session_state.rhythm_variation_time = 48
if "handstroke_gap_variation_time" not in st.session_state:
    st.session_state.handstroke_gap_variation_time = 10
    
titles = []
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = st.session_state.cached_data[i][0] + ': ' + str(st.session_state.cached_data[i][1]) + ' changes'
    if st.sidebar.button(title, key = i):
        st.session_state.current_touch = i
        selected_title = title
    titles.append(title)
#Give option to upload things
st.sidebar.write("**Upload timing data from device:**")

uploaded_files = st.sidebar.file_uploader(
    "Choose a .csv file", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")

dealwith_upload()
    
if st.session_state.current_touch < 0:
    st.write('**Select a touch from the options on the left, or upload a new one**')
else:
    st.write('**Analysing ringing from %s**' % titles[st.session_state.current_touch])

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
        
        raw_actuals = raw_data["Actual Time"]
        
        nbells = np.max(raw_data["Bell No"])
    else:        
        #I think it would be easiest to just plonk this into a dataframe and treat it like an imported one, given I've already written code for that
        allbells = []
        allcerts_save = []
        allstrikes = []
        yvalues = np.arange(len(st.session_state.allstrikes[:,0])) + 1
        orders = []
        for row in range(len(st.session_state.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], yvalues), reverse = False)])
            certs = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], st.session_state.allcerts[:,row]), reverse = False)])
            allstrikes = allstrikes + sorted((st.session_state.allstrikes[:,row]).tolist())
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
        
    if "Individual Model" not in  raw_data.columns.tolist():
        count_test = st.session_state.rhythm_variation_time; gap_test = st.session_state.handstroke_gap_variation_time
        ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test)
        raw_data['Individual Model'] = ideal_times
        existing_models.append('Individual Model')

    if "Metronomic Model" not in  raw_data.columns.tolist():
        @st.cache_data
        def find_metronomic():
            nrows = int(len(raw_actuals)//nbells)
    
            ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test)
            raw_data['Individual Model'] = ideal_times
            all_metros = []
            for row in range(nrows):
                actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
                start = np.min(actual)
                end = np.max(actual)
                metronomic_target = np.linspace(start, end, nbells)
                all_metros = all_metros + metronomic_target.tolist()

            return all_metros

        all_metros = find_metronomic()
        raw_data['Metronomic Model'] = all_metros
        existing_models.append('Metronomic Model')

    if len(existing_models) > 0:
                
        selection = st.selectbox("Select striking model:", options = existing_models)   #Can set default for this later?
        
        raw_target = np.array(raw_data[selection])
        raw_bells = np.array(raw_data["Bell No"])
        #Plot blue line
        nstrikes = len(raw_actuals)
        nrows = int(nstrikes//nbells)
    
        if selection == "Individual Model":
            rhythm_variation_time = st.slider("Rhythm variation time:", min_value = 2, max_value = 10, value=4, format = "%d changes", step = 1)
            st.session_state.handstroke_gap_variation_time = st.slider("Handstroke gap variation time:", min_value = 4, max_value = 20, value = 10, format = "%d changes", step = 2)
            st.session_state.rhythm_variation_time = rhythm_variation_time*nbells
            ideal_times = find_ideal_times(raw_data['Actual Time'], nbells, ncount = st.session_state.rhythm_variation_time, ngaps = st.session_state.handstroke_gap_variation_time)
            raw_data['Individual Model'] = ideal_times

            
        remove_mistakes = st.checkbox("Remove presumed method mistakes from the stats? (Not foolproof -- I'm working on it)", value = True)
        
        min_include_change, max_include_change = st.slider("Include changes in range:", min_value = 0, max_value = nrows, value=(0, nrows), format = "%d", step = 2)

        
        st.message = st.empty()
        st.message.write("Calculating stats things...")
        #Blue Line
        st.blueline = st.empty()
        with st.blueline.expander("View Blue Line"):
        
            def plot_blue_line(raw_target_plot, min_plot_change, max_plot_change):
        
                raw_actuals = np.array(raw_data["Actual Time"])
        
                toprint = []
                orders = []; starts = []; ends = []
                for row in range(nrows):
                    actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
                    target = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                    bells =   np.array(raw_bells[row*nbells:(row+1)*nbells])  
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
                rows_per_plot = int(nrows_plot/nplotsk) + 2
                
                #fig,axs = plt.subplots(1,ncols, figsize = (15,4*nrows/(nbells + 4)))
                fig,axs = plt.subplots(1,nplotsk, figsize = (10, 30))
                for plot in range(nplotsk):
                    if nplotsk > 1:
                        ax = axs[plot]
                    else:
                        ax = axs
                    for bell in range(1,nbells+1):#nbells):
                        points = []  ; changes = []
                        bellstrikes = np.where(raw_bells == bell)[0]
                        for row in range(min_plot_change, max_plot_change):
                            #Find linear position... Linear interpolate?
                            target_row = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                            ys = np.arange(1,nbells+1)
                            f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                            rat = float(f(raw_actuals[bellstrikes][row]))
                            points.append(rat); changes.append(row)
                        ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10])
                        ax.plot((bell)*np.ones(len(points)), changes, c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
                    for row in range(min_plot_change, max_plot_change):
                        ax.plot(np.arange(-1,nbells+3), row*np.ones(nbells+4), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
                    
                    plt.gca().invert_yaxis()
                    ax.set_ylim((plot+1)*rows_per_plot + min_plot_change, plot*rows_per_plot+ min_plot_change )
                    ax.set_xlim(-1,nbells+2)
                    ax.set_xticks([])
                    ax.set_aspect('equal')
                    #if plot == nplotsk-1:
                    #    plt.legend()
                    #ax.set_yticks([])
                plt.tight_layout()
                st.pyplot(fig)
                
            min_plot_change, max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = nrows, value=(0, min(240, nrows)), format = "%d", step = 2)
            plot_blue_line(raw_target, min_plot_change, max_plot_change)
            
        diffs = np.array(raw_actuals)[1:] - np.array(raw_actuals)[:-1]
        cadence = np.mean(diffs)*(2*nbells)/(2*nbells + 1)

        with st.expander("View Histograms"):
            
            alldiags = np.zeros((3,3,nbells))   #Type, stroke, bell

            titles = ['All blows', 'Handstrokes', 'Backstrokes']

            x_range = st.slider("Histogram x range:", min_value = 0, max_value = 250, value= 150, format = "%dms")
            nbins = st.slider("Number of histogram bins", min_value = 0, max_value = 100, value= 50, format = "%d", step = 1)

            for plot_id in range(3):
                #Everything, then handstrokes, then backstrokes

                fig, axs = plt.subplots(3,4, figsize = (10,7))
                allerrors = []
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


                    #Diagnostics
                    alldiags[0,plot_id,bell-1] = np.sum(errors)/count
                    alldiags[1,plot_id,bell-1] = np.sqrt(np.sum((errors-np.sum(errors)/count)**2)/count)
                    alldiags[2,plot_id,bell-1] = np.sqrt(np.sum(errors**2)/count)

                    allerrors += np.sum(errors)/count
                    ax = axs[(bell-1)//4, (bell-1)%4]

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
                st.pyplot(fig)
        
        st.message.write("Standard deviation from ideal for this touch: %dms" % np.mean(alldiags[2,2,:]))

        #Bar Chart
        with st.expander("View Error Bar Charts"):
            fig, axs = plt.subplots(3, figsize = (12,7))
            bar_width = 0.3

            data_titles = ['Avg. Error', 'Std. Dev. from Average', 'Std. Dev. From Ideal']

            x = np.arange(nbells)
            for plot_id in range(3):
                ax = axs[plot_id]

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
            st.pyplot(fig)

