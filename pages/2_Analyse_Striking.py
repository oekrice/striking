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

from strike_model import find_ideal_times
from scipy import interpolate
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
        st.session_state.uploader_key += 1
        st.rerun()
    return
    
    
st.set_page_config(page_title="Analyse Striking", page_icon="📈")
st.markdown("# Analyse Striking")
st.sidebar.header("Choose a Touch:")
st.write(
    """This page is for analysing the striking from the strike times either generated with the "Analyse Audio" tab or for an uploaded .csv"""
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
    
titles = []
#Write out the touch options from the cache --  can theoretically load in more
for i in range(len(st.session_state.cached_data)):
    #Title should be number of changes and tower
    title = st.session_state.cached_data[i][0] + ' -- ' + str(st.session_state.cached_data[i][1]) + ' changes'
    if st.sidebar.button(title):
        st.session_state.current_touch = i
        selected_title = title
    titles.append(title)
#Give option to upload things
st.sidebar.write("**Upload timing data from device:**")

uploaded_files = st.sidebar.file_uploader(
    "Choose a .csv file", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")

dealwith_upload()
    
st.write(st.session_state.current_touch)
if st.session_state.current_touch < 0:
    st.write('**Select a touch from the options on the left, or upload a new one**')
else:
    st.write('**Analysing ringing from %s:**' % titles[st.session_state.current_touch])

if st.session_state.current_touch >= 0:
    #Write in to a local bit to actually do the analysis
    strikes = st.session_state.cached_strikes[st.session_state.current_touch]
    certs = st.session_state.cached_certs[st.session_state.current_touch]
    
    available_models = []
    #If data is uploaded, treat it slightly differently to otherwise. Can just output various things immediately without calculation
    if len(strikes) == 0:
        st.write('from csv')
        #This is from a .csv
        raw_data = st.session_state.cached_rawdata[st.session_state.current_touch]
        cols = raw_data.columns.tolist()

        existing_models = [val for val in cols if val not in ["Bell No", "Confidence", "Actual Time"]]
        existing_models = [val for val in existing_models if val[:7] != "Unnamed"]
        
        raw_actuals = raw_data["Actual Time"]
        
        nbells = np.max(raw_data["Bell No"])
    else:        
        st.write('From this app. Dont use yet.')
        existing_models = []
        
    if len(existing_models) > 0:
        
        selection = st.selectbox("Select striking model:", options = existing_models)   #Can set default for this later?
        
        raw_target = np.array(raw_data[selection])
        raw_bells = np.array(raw_data["Bell No"])
        #Plot blue line
        nstrikes = len(raw_actuals)
        nrows = int(nstrikes//nbells)
    
        #Blue Line
        with st.expander("View Blue Line"):
        
            raw_actuals = np.array(raw_actuals)
    
            toprint = []
            orders = []; starts = []; ends = []
            for row in range(nrows):
                yvalues = np.arange(nbells) + 1
                actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
                target = np.array(raw_target[row*nbells:(row+1)*nbells])
                bells =   np.array(raw_bells[row*nbells:(row+1)*nbells])  
                order = np.array([val for _, val in sorted(zip(actual, yvalues), reverse = False)])
                targets = np.array([val for _, val in sorted(zip(actual, actual-target), reverse = False)])
                toprint.append(actual-target)
                orders.append(bells)
                starts.append(np.min(target))
                ends.append(np.max(target))
                #starts.append(np.min(actual))
                #ends.append(np.max(actual))
                
            min_plot_change, max_plot_change = st.slider("View changes in range:", min_value = 0, max_value = nrows, value=(0, 120), format = "%d", step = 2)
    
            columns = ["Bell %d" % val for val in np.arange(1, nbells+1)]
    
            nrows_plot = max_plot_change - min_plot_change
            rows_per_plot = 61#6*int(nrows_plot//24)
            nplotsk = nrows_plot//rows_per_plot + 1
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
                    errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])
                    targets = np.array(raw_target[bellstrikes])
                    for row in range(min_plot_change, max_plot_change):
                        #Find linear position... Linear interpolate?
                        target_row = np.array(raw_target[row*nbells:(row+1)*nbells])
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


    



#plotting_demo()

#show_code(plotting_demo)
if False:   #Trying to be interactive, sort of works OK but not really...

    #An attempt to plot the method?
    rows_per_plot = 60
    nplotsk = nrows//rows_per_plot + 1
    rows_per_plot = int(nrows/nplotsk) + 2
    blueline_data = pd.DataFrame({})
    raw_actuals = np.array(raw_actuals)
    
    #Make raw dataframe with each ROW as the things
    row_data = pd.DataFrame({})
    

    fig = go.Figure()
    plot_array = np.zeros((nbells, int(len(raw_actuals)/nbells)))
    for bell in range(1, nbells + 1):
        points = []  ; changes = []
        bellstrikes = np.where(raw_bells == bell)[0]
        errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])
        targets = np.array(raw_target[bellstrikes])
        for row in range(min_plot_change, max_plot_change+1):
            #Find linear position... Linear interpolate?
            target_row = np.array(raw_target[row*nbells:(row+1)*nbells])
            ys = np.arange(1,nbells+1)
            f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
            rat = float(f(raw_actuals[bellstrikes][row]))
            points.append(rat); changes.append(row)
            plot_array[bell-1,row] = rat
            
        fig.add_trace(go.Line(x = points, y = changes, name = columns[bell-1]))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(autorange ="reversed")
    fig.update_layout(height = 800)
    st.plotly_chart(fig)
