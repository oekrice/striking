import streamlit as st

import numpy as np
import time
import pandas as pd
from listen_classes import data
import matplotlib.pyplot as plt

from listen_other_functions import find_ringing_times, find_strike_probabilities, find_first_strikes, do_frequency_analysis, find_strike_times_rounds, find_colour
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


def establish_initial_rhythm_test(Paras, final = False):

    if not final:
        Data = data(Paras, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    else:
        Data = data(Paras, tmin = 0.0, tmax = 60.0) #This class contains all the important stuff, with outputs and things
        
    Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)
    Paras.reinforce_tmax = Paras.ringing_start*Paras.dt + Paras.reinforce_tmax
    Paras.rounds_tmax = Paras.ringing_start*Paras.dt  + Paras.rounds_tmax

    if not final:
        st.current_log.write('Ringing detected from approx. time %d seconds' % (Paras.ringing_start*Paras.dt))

    if not final:
        Data = data(Paras, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    else:
        Data = data(Paras, tmin = 0.0, tmax = 60.0 + Paras.ringing_start*Paras.dt) #This class contains all the important stuff, with outputs and things

    #Find strike probabilities from the nominals
    Data.strike_probabilities = find_strike_probabilities(Paras, Data, init = True, final = final)
    #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
    Paras.local_tmin = Paras.overall_tmin
    Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
    Paras.stop_flag = False

    Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)
    
    Paras.first_strikes, Paras.first_strike_certs = find_first_strikes_test(Paras, Data)
    
    Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
        
    return Data


def find_rough_cadence(Paras, Data):
    #Takes the first 20 seconds or so and smooths to hell, finds peaks and figures out when times could be.
    #Probably reasonable to use the tenor here? Or just all
    #REDO this for loudness, rather than fancy probability things
    cadences = []
    fig = plt.figure()
    back_bell_cutoff = max(2,int(len(Data.strike_probabilities)/4))
    for i in range(len(Data.strike_probabilities)):
        verysmooth_probs = gaussian_filter1d(Data.strike_probabilities[i],100)
        plt.plot(verysmooth_probs)
        peaks, _ = find_peaks(verysmooth_probs) 
        avg_peak_distance = np.sum(peaks[1:] - peaks[:-1])/len(peaks[1:])
        cadences.append(avg_peak_distance)
        print(avg_peak_distance)
    st.pyplot(fig)
    return np.mean(cadences[-back_bell_cutoff:])

def find_first_strikes_test(Paras, Data):
    #Takes normalised wave vector, and does some fourier things
    #This function is the one which probably needs improving the most...

    #Want to plot prominences and things to check if these is an obvious pattern
    rough_cadence = find_rough_cadence(Paras, Data)   #This gives an impression of how long it is between successive changes
    npeaks_ish = int((Paras.rounds_tmax/Paras.dt - Paras.ringing_start*Paras.dt)/rough_cadence)

    fig = plt.figure()
    ratios = []
    nbells = len(Data.strike_probabilities)
    significant_peaks = []
    for bell in range(nbells):   #Establish peaks and reliability
        probs = Data.strike_probabilities[bell][:]
        probs = gaussian_filter1d(probs, 5)
        peaks, _ = find_peaks(probs) 
        peaks = peaks[peaks < Paras.rounds_tmax/Paras.dt]
        peaks = peaks[peaks > Paras.ringing_start + int(2.5/Paras.dt)]
        prominences = peak_prominences(probs, peaks)[0]
        peaks = np.array([val for _, val in sorted(zip(prominences,peaks), reverse = True)]).astype('int')
        prominences = np.array(sorted(prominences, reverse = True))
        if len(peaks) > 2*npeaks_ish: 
            #This will work for comparisons
            confidence_ratio = np.mean(prominences[:npeaks_ish])/np.mean(prominences[npeaks_ish:2*npeaks_ish])
        else:
            confidence_ratio = 0
        ratios.append(confidence_ratio)

        threshold = 0.9*sorted(prominences, reverse = True)[min(int(0.75*npeaks_ish), len(peaks)-1)]
        significant_peaks.append(np.array(peaks[prominences > threshold]))  
    #These should be slightly too many 
    best_bells = np.array([val for _, val in sorted(zip(ratios,np.arange(nbells)), reverse = True)]).astype('int')
    for bell in best_bells[:3]:
        plt.scatter(significant_peaks[bell], bell*np.ones(len(significant_peaks[bell])))

    plt.gca().invert_yaxis()
    plt.legend()
    #plt.xlim(0,2000)
    st.pyplot(fig)
    plt.close()

    tenor_probs = Data.strike_probabilities[-1]
    tenor_probs = gaussian_filter1d(tenor_probs, 5)
    tenor_peaks, _ = find_peaks(tenor_probs) 
    tenor_peaks = tenor_peaks[tenor_peaks < Paras.rounds_tmax/Paras.dt]
    tenor_peaks = tenor_peaks[tenor_peaks > Paras.ringing_start + int(2.5/Paras.dt)]
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]
    tenor_peaks = np.array([val for _, val in sorted(zip(prominences,tenor_peaks), reverse = True)]).astype('int')
    prominences = np.array(sorted(prominences, reverse = True))
    #Test the first few tenor peaks to see if the following diffs are fine...    
    npeaks_ish = int((Paras.rounds_tmax - Paras.ringing_start*Paras.dt)/2.0)   #How many strikes expected in this time
    
    if npeaks_ish < 5:
        st.error("Not enough reliable rounds detected. Make sure rounds starts within a minute of the start of the recording,"
        " and there are at least a few decent whole pulls.")
        st.session_state.reinforce_status = 0
        time.sleep(5.0)

        st.rerun()
        
    threshold = 0.5*sorted(prominences, reverse = True)[min(npeaks_ish - 1, len(tenor_peaks)-1)]
        
    tenor_big_peaks = np.array(tenor_peaks[prominences > threshold])  
    tenor_peaks = np.array(tenor_peaks[prominences > 0.01]) 
    
    tenor_big_peaks = sorted(tenor_big_peaks)
                        
    if len(tenor_big_peaks) < Paras.nrounds_min:
        st.error('Reliable tenor strikes not found within the required time... Try cutting out audio at the start or increasing rounds time?')
        st.session_state.reinforce_status = 0
        time.sleep(2.0)
        st.stop()
        
    tenor_strikes = []; best_length = 0; go = True
    
    #print('Big peaks', tenor_big_peaks)
    for first_test in range(min(8, len(tenor_big_peaks))):
        if not go:
            break
        first_strike = tenor_big_peaks[first_test]
              
        teststrikes = [first_strike]

        start = first_strike + 1
        end = first_strike + int(Paras.max_change_time/Paras.dt)
        
        rangestart = int(1.0/Paras.dt)   #Minimum time to check from previous strike
        rangeend = int(Paras.max_change_time/Paras.dt) 
        
        for ri in range(Paras.nrounds_max):  #Try to find as many as is reasonable here
            #Find most probable tenor strikes
            poss = tenor_peaks[(tenor_peaks > start)*(tenor_peaks < end)]  #Possible strikes in range -- just pick biggest
            prominences = peak_prominences(tenor_probs, poss)[0]
            poss = np.array([val for _, val in sorted(zip(prominences,poss), reverse = True)]).astype('int')
            if ri > 1:
                avg_change_length = np.mean(np.array(teststrikes)[1:] - np.array(teststrikes)[:-1])
                last_twochange_length = np.array(teststrikes)[-1] - np.array(teststrikes)[-3] 
                avg_twochange_length = np.mean(np.array(teststrikes)[2:] - np.array(teststrikes)[:-2])
                #Look at the one detected TWO blows ago and figure that out. Hopefully fine...
                rangestart = int(0.9*last_twochange_length)
                rangeend = int(1.1*last_twochange_length)

            if len(poss) < 1:
                break
            

            teststrikes.append(poss[0])

            if ri > 1:
                start = teststrikes[-2] + rangestart
                end = teststrikes[-2] + rangeend
            else:
                start = poss[0] + rangestart
                end = poss[0] + rangeend
            
        teststrikes = np.array(teststrikes)
        diff1s = teststrikes[1:] - teststrikes[:-1]
        diff2s = teststrikes[2:] - teststrikes[:-2]

        twostroke_max_variance = 1.0*avg_change_length/(Paras.nbells - 1)   #Relative difference in the length of changes
        handstroke_max_variance = 2.0*avg_change_length/(Paras.nbells - 1)   #Relative difference accounting for handstroke gaps (should be at least 1)

        #print(first_test, diff1s, diff2s)
        for tests in range(2, len(diff2s)):
            if max(diff2s[:tests]) - min(diff2s[:tests]) < twostroke_max_variance:   
                if max(diff1s[:tests]) - min(diff1s[:tests]) < handstroke_max_variance:
                    if tests + 2 > best_length:
                        best_length = tests + 2
                        tenor_strikes = teststrikes[:tests+2]
                 
    if len(tenor_strikes) < 5:
        #print(tenor_big_peaks, tenor_peaks)
        #st.session_state.reinforce = 0
        st.error('Reliable tenor strikes not found within the required time... Try cutting out start silence or increasing rounds time?')
        st.session_state.reinforce_status = 0
        time.sleep(5.0)
        st.rerun()
    #print('Tenor strikes in rounds (check these are reasonable): ', np.array(tenor_strikes)*Paras.dt)
    
    diff1s = tenor_strikes[1::2] - tenor_strikes[0:-1:2]
    diff2s = tenor_strikes[2::2] - tenor_strikes[1:-1:2]
    
    if np.mean(diff1s) < np.mean(diff2s):
        handstroke_first = False  #first CHANGE is a handstroke, but this counts from the previous tenor
    else:
        handstroke_first = True
            
    
    Paras.first_change_limit = tenor_strikes[0]*np.ones(Paras.nbells) + 10
    Paras.reinforce_tmax = Paras.reinforce_tmax + tenor_strikes[0]
    nrounds_test = len(tenor_strikes) - 1
    
    handstroke = handstroke_first
    
    init_aims = []; cadences = []

    for rounds in range(nrounds_test):
        #Interpolate the bells smoothly (assuming steady rounds)
        if not handstroke:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], Paras.nbells + 1)
        else:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], Paras.nbells + 2)
            
        cadences.append(belltimes[1] - belltimes[0])
        belltimes = belltimes[-Paras.nbells:]
                
        init_aims.append(belltimes)
                  
        handstroke = not(handstroke)
          
    #print('Attempting to find ', len(init_aims), ' rows for rounds...')
    
    cadence = np.mean(cadences)
    Data.cadence = cadence
    Paras.cadence = cadence
    #Do this just like the final row finder! But have taims all nicely. Also use same confidences.

    #Obtain VERY smoothed probabilities, to compare peaks against
    
    #Use these guesses to find the ACTUAL peaks which should be nearby...
    init_aims = np.array(init_aims)
    
    strikes = np.zeros(init_aims.T.shape)
    strike_certs = np.zeros(strikes.shape)
    spacings = np.mean(cadences)*np.ones(Paras.nbells) #Distances from adjacent bells
    
    Paras.nrounds_max = len(init_aims)

    probs_raw = Data.strike_probabilities[:]
    probs_raw = gaussian_filter1d(probs_raw, Paras.strike_smoothing, axis = 1)

    tcut = 1 #Be INCREDIBLY fussy with these picks or the wrong ones will get nicked
    
    for bell in range(Paras.nbells):
        #Find all peaks in the probabilities for this individual bell
        probs_adjust = probs_raw[bell,:]**(Paras.probs_adjust_factor + 1)/(np.sum(probs_raw[:,:], axis = 0) + 1e-6)**Paras.probs_adjust_factor
        #Adjust for when the rounds is a bit shit
        
        peaks, _ = find_peaks(probs_adjust) 
        sigs = peak_prominences(probs_adjust, peaks)[0]
        sigs = sigs/np.max(sigs)

        for ri in range(Paras.nrounds_max): 
            #Actually find the things. These should give reasonable options
            aim = init_aims[ri, bell]
            
            poss = peaks[(peaks > aim - 1.0*cadence)*(peaks < aim + 1.0*cadence)]   #These should be accurate strikes
            yvalues = sigs[(peaks > aim - 1.0*cadence)*(peaks < aim + 1.0*cadence)]

            scores = []
            for k in range(len(poss)):  #Many options...
                tvalue = 1.0/(abs(poss[k] - aim)/tcut + 1)**Paras.strike_alpha
                yvalue = yvalues[k]
                scores.append(tvalue*yvalue**Paras.strike_gamma_init)
                
            if len(scores) > 0:
                
                kbest = scores.index(max(scores))
                
                strikes[bell, ri] = poss[kbest]
                strike_certs[bell,ri] = scores[k]

            else:
                strikes[bell, ri] = aim
                strike_certs[bell, ri] = 0.0
               
        
    del probs_raw; del probs_adjust; del peaks; del sigs
    
    strikes = np.array(strikes)
    strike_certs = np.array(strike_certs)    

    all_spacings = []
    for ri in range(Paras.nrounds_max):
        for bell in range(0,Paras.nbells):
            if bell == 0:
                spacings[bell] = strikes[bell+1,ri] - strikes[bell,ri]
            elif bell == Paras.nbells - 1:
                spacings[bell] = strikes[bell,ri] - strikes[bell-1,ri]
            else:
                spacings[bell] = min(strikes[bell+1,ri] - strikes[bell,ri], strikes[bell,ri] - strikes[bell-1,ri])
        all_spacings.append(spacings.copy())

    all_spacings = np.array(all_spacings).T/np.max(spacings)

    strike_certs = strike_certs*all_spacings
    
    #Check this is indeed handstroke or not, in case of an oddstruck tenor
    diff1s = strikes[:,1::2] - strikes[:,0:-1:2]
    diff2s = strikes[:,2::2] - strikes[:,1:-1:2]
    
    #st.write('Initial diffs', diff1s, diff2s)
    kback = len(diff2s[0])
    kback = min((2*kback)//2, 4)   #MUST BE EVEN
    if np.mean(diff1s[:]) < np.mean(diff2s[:]):
        handstroke_first = True
    else:
        handstroke_first = False
    #st.write('First change', strikes[:,0], handstroke_first)
    #st.write(diff1s, diff2s)
    #st.write(handstroke_first)
    
    
    nrounds_per_bell = 2
    row_ids = []
    final_strikes = []; final_certs = []
    for bell in range(Paras.nbells):
        threshold = 0.0
        allcerts = []; count = 0
        for row in range(len(strikes[0])):
            allcerts.append(strike_certs[bell,row])
        if len(allcerts) > Paras.nreinforce_rows:
            threshold = max(threshold, sorted(allcerts, reverse = True)[nrounds_per_bell])
        #Threshold for THIS BELL
        for row in range(len(strikes[0])):
            if strike_certs[bell,row] > threshold and count < nrounds_per_bell:
                if row not in row_ids:
                    row_ids.append(row)
                    final_strikes.append(strikes[:,row])
                    final_certs.append(strike_certs[:,row])
                    count += 1
                    
    final_strikes = np.array([val for _, val in sorted(zip(row_ids, final_strikes))]).astype('int')
    final_certs = np.array([val for _, val in sorted(zip(row_ids, final_certs))])

    if np.min(row_ids)%2 == 1:
        handstroke_first = not(handstroke_first)
        
    strikes = np.array(final_strikes).T
    strike_certs = np.array(final_certs).T
    
    Data.handstroke_first = handstroke_first
    st.session_state.handstroke_first = handstroke_first

    #Determine how many rounds there actually are? Nah, it's probably fine...
    Paras.first_change_start = np.min(strikes[:,0])
    Paras.first_change_end = np.max(strikes[:,0])

    return strikes, strike_certs
