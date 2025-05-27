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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import minimize_scalar, curve_fit, OptimizeWarning
import warnings
import streamlit as st
      
#This one is for the RWP model, using the principle of the code provided by Ian McCallion writte in OCtave (but completely rewritten)

@st.cache_data
def find_limits(alltimes_trim, nbells):
    nwholepulls = int(len(alltimes_trim)/(nbells*2))
    limits = np.zeros((nwholepulls, 2))
    xvalues = np.arange(nbells*2) 
    for wholepull in range(nwholepulls):
        strikes = alltimes_trim[(wholepull)*nbells*2:(wholepull+1)*nbells*2]
        slope, intercept, _, _, _ = stats.linregress(xvalues, strikes)
        limits[wholepull] = [intercept + 0*slope, intercept + xvalues[-1]*slope]
    return np.array(limits)

@st.cache_data
def find_centres(alltimes_trim, nbells):
    #Finds the ideal centres of each whole pull as midway between the centres of those either side
    nwholepulls = int(len(alltimes_trim)/(nbells*2))
    centres = np.zeros(nwholepulls)
    ideal_durations = np.zeros(nwholepulls)
    ideal_centres = np.zeros(nwholepulls)
    for wholepull in range(nwholepulls):
        strikes = alltimes_trim[(wholepull)*nbells*2:(wholepull+1)*nbells*2]
        centres[wholepull] = np.mean(strikes)
    ideal_centres = centres.copy()
    for wholepull in range(1,nwholepulls-1):
        ideal_centres[wholepull] = 0.5*(centres[wholepull + 1] + centres[wholepull - 1])
        ideal_durations[wholepull] = 0.5*(centres[wholepull + 1] - centres[wholepull - 1])
    ideal_durations[0] = ideal_durations[1]; ideal_durations[-1] = ideal_durations[-2]
    return ideal_centres, ideal_durations

@st.cache_data
def find_all_times(nbells, ideal_centres, ideal_interbell_gaps):
    #Finds all the ideal times using the ideal centres and interbell gaps
    nwholepulls = len(ideal_centres)
    ideal_times = np.zeros(nbells*2*nwholepulls)
    for wholepull in range(nwholepulls):
        row_times = ideal_centres[wholepull] + ideal_interbell_gaps[wholepull]*(np.arange(nbells*2) - nbells + 0.5)
        ideal_times[(wholepull)*nbells*2:(wholepull+1)*nbells*2] = row_times
    return ideal_times

    
@st.cache_data
def find_ideal_times_rwp(alltimes, nbells, key = -1):

    #Finds the ideal times based on the Rod Pipe model. Requires a whole number of whole pulls, which won't necessarily be the case for me
    alltimes = np.array(alltimes)

    nrows = int(len(alltimes)/nbells)

    #Get rid of any stray handstrokes at the end, if any
    if nrows%2 == 0:
        alltimes_trim = alltimes[:]
    else:
        alltimes_trim = alltimes[:-nbells]

    nrows_trim = int(len(alltimes_trim)/nbells)
    #Determine the timing of the end of each whole pull, using linear regression
    wholepull_limits = find_limits(alltimes_trim, nbells)

    #Caculate interbell gaps etc. for the whole piece
    avg_interbell_gap = np.mean(wholepull_limits[:,1] - wholepull_limits[:,0])/(nbells*2 - 1)
    avg_handstroke_gap = np.mean(wholepull_limits[1:,0] - wholepull_limits[:-1,1])
    gap_ratio = avg_handstroke_gap/avg_interbell_gap

    ideal_centres, ideal_durations = find_centres(alltimes_trim, nbells)   #The ideal centres and durations of each whole pull

    ideal_interbell_gaps = ideal_durations/(nbells*2 - 1 + gap_ratio)

    all_ideal_times = find_all_times(nbells, ideal_centres, ideal_interbell_gaps)

    if nrows%2 != 0:  #Append the last row as perfection
        all_ideal_times = np.concatenate((all_ideal_times, alltimes[-nbells:]))

    return all_ideal_times


