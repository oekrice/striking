import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import minimize_scalar, curve_fit
import streamlit as st
import time

@st.cache_data
def find_ideal_hgap_band(cut_init, gap_init, row, nbells):
    #Finds ideal handstroke gap for two rows (given in cut_init)
    def find_r(gap):
        cut = np.array(cut_init)
        cut[nbells:] = cut_init[nbells:] - gap
        _, _, r, _, _ = stats.linregress(np.arange(nbells*2), cut)
        return (1-r)

    res = minimize_scalar(find_r, bounds=(10, 400), method='bounded')

    return res.x

#I think this is the easiest way to do it, and quickest. Then go through again with a priori data

@st.cache_data
def find_all_gaps_band(alltimes, nbells, nrows):
    all_gaps = np.zeros(nrows) #Handstroke gaps BEFORE each row. Backstrokes will just be zero
    gap_init = (alltimes[nbells*2-1] - alltimes[0])/(nbells*2)
    all_gaps[0] = gap_init

    for row in range(1,nrows-1,1): #Starting the cut on each backstroke
        start = row*nbells
        end   = (row+2)*nbells
        cut = np.array(alltimes[start:end])   #Contains 24 blows with a backstroke change and handstroke change
        #Find ideal handstroke gap
        gap = find_ideal_hgap_band(cut, gap_init, row, nbells)

        cut[nbells:] = cut[nbells:] - gap
        all_gaps[row+1] = gap

    #print('Ideal handstroke gaps found...')
    return all_gaps

@st.cache_data
def find_predicted_gaps_band(all_ideal_gaps, nbells, nrows, ngaps):
    #Ussing data before AND after, find the ideal handstroke gap times
    #Perhaps fit a quadratic to the previous few, if possible
    def f(x,a,b,c):
        return a*x**2 + b*x + c

    #print('Finding best possible handstroke gaps in the moment')
    all_gaps = np.zeros(nrows)
    nfits = ngaps #Number of handstroke rows to establish the rhythm from.
    #Given doing before and after, can always do the curve fit? That's a lot neater
    nfits = max(nfits, 4)
    #st.line_chart(all_ideal_gaps[::2])
    for row in range(0,nrows,2):
        back = min(row, nfits)
        forward = min(len(all_ideal_gaps)-row, nfits)
        alldata = all_ideal_gaps[row-back:row+forward:2]
        popt, pcov = curve_fit(f, np.arange(len(alldata)), alldata)
        all_gaps[row] = f(back//2, *popt)
    #st.line_chart(all_gaps[::2])

    return all_gaps

@st.cache_data
def find_ideal_times_band(alltimes, nbells, ncount = 24, ngaps = 6, key = -1):

    alltimes = np.array(alltimes)

    nrows = int(len(alltimes)/nbells)

    all_ideal_gaps = find_all_gaps_band(alltimes, nbells, nrows)

    all_predicted_gaps = find_predicted_gaps_band(all_ideal_gaps, nbells, nrows, ngaps = ngaps)

    #print('Handstroke gaps determined...')
    #Actually finds the ideal strike time of each bell, based on the predicted handstroke gaps and the preceding strikes (up to a point to be determined.)
    #Bell number is irrelevant
    def f1(x,a,b):
        return a*x + b

    def f2(x,a,b,c):
        return a*x**2 + b*x + c

    def adjust_times(data, row_number, n_adjust, position, cut_start_distance):
        #Remove the effect of the handstroke gaps -- keep the individual times the same but rewrite the preceding few as necessary, essentially assuming everything before comes earlier
        first_change = max(0,row_number - n_adjust)
        last_change = min(row_number + n_adjust, nrows)
        count = 0   #Retrofit previous changes
        for row_change in range(row_number,first_change,-1):
            #Figure out which ones to retrofit
            limit = count*nbells + position - cut_start_distance
            #print(row_change, first_change, position, all_predicted_gaps[row_change], limit)
            count += 1
            #print(data)
            #print(all_predicted_gaps[row_change])
            #print(data[1:] - data[:-1])
            #print('a',data)

            if limit == 0:
                data[:] = data[:] + all_predicted_gaps[row_change]
            elif limit < len(data):
                data[:-limit] = data[:-limit] + all_predicted_gaps[row_change]
                
        count = 0   #Retrofit next few changes
        for row_change in range(row_number+1,last_change,1):
            #Figure out which ones to retrofit
            limit = count*nbells + (nbells - position) + cut_start_distance
            #st.write(limit, row_position)
            #print(row_change, first_change, position, all_predicted_gaps[row_change], limit)
            count += 1
            #print(data)
            #print(all_predicted_gaps[row_change])
            #print(data[1:] - data[:-1])
            #print('a',data)

            if limit == 0:
                data[:] = data[:] - all_predicted_gaps[row_change]
            elif limit < len(data):
                data[limit:] = data[limit:] - all_predicted_gaps[row_change]
           
        
            #print('b',data)
        return data

    nback = int(ncount//2)  
    all_ideals = np.zeros(nrows*nbells)
    n_adjust = int(nback/nbells)

    if len(alltimes) != len(all_ideals):
        st.error('Not a complete number of changes -- aborting')
        st.stop()
    #print('Finding individual strikes')

    for strike in range(len(all_ideals)):
        row_position = strike%(nbells)  #Row position up to nbells
        row_number = strike//nbells
    
        back = max(0, strike - nback)
        forward = min(len(alltimes), strike+nback)
        
        #Is always enough for quadratic now
        
        data = np.array(alltimes[back:forward])

        data = adjust_times(data, row_number, n_adjust, row_position, strike-back)  #Adjust to take into account handstroke gaps

        basis = np.arange(back, forward)
        popt, pcov = curve_fit(f2, basis, data)
        all_ideals[strike] = int(f2(strike, *popt))
        
        
        
    return all_ideals

#all_ideal_times = find_ideal_times(alltimes)

