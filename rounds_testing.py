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

    #Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)  #Why does this happen twice? Forgot...
    
    Paras.first_strikes, Paras.first_strike_certs = find_first_strikes_test(Paras, Data)
    
    Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
        
    return Data


def find_rough_cadence(Paras, Data):
    #Takes the first 20 seconds or so and smooths to hell, finds peaks and figures out when times could be.
    #Probably reasonable to use the tenor here? Or just all
    #REDO this for loudness, rather than fancy probability things. Ian said so.
    #Need to isolate for frequencies though
    cadences = []
    fig = plt.figure()

    for bell in range(Paras.nbells):
        freq_test = Data.test_frequencies[bell]

        loudness = Data.transform[:,freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        
        loudness = gaussian_filter1d(loudness, int(0.1/Paras.dt),axis = 0)
        loudsum = np.sum(loudness, axis = 1)
        loudsmooth = gaussian_filter1d(loudsum, int(0.5/Paras.dt), axis = 0)
        plt.plot(loudsmooth, label = bell)

        back_bell_cutoff = max(2,int(len(Data.strike_probabilities)/4))

        peaks, _ = find_peaks(loudsmooth) 
        avg_peak_distance = np.sum(peaks[1:] - peaks[:-1])/len(peaks[1:])
        cadences.append(avg_peak_distance)
    plt.close()
    #st.pyplot(fig)
    return np.mean(cadences[-back_bell_cutoff:])

def find_first_strikes_test(Paras, Data):
    #Takes normalised wave vector, and does some fourier things
    #This function is the one which probably needs improving the most...

    #Want to plot prominences and things to check if these is an obvious pattern
    rough_cadence = find_rough_cadence(Paras, Data)   #This gives an impression of how long it is between successive changes
    print('Peal speed (hours):', 0.01*rough_cadence*5000/3600, rough_cadence)
    st.write('Peal speed (hours):', 0.01*rough_cadence*5000/3600)
    npeaks_ish = int((Paras.rounds_tmax/Paras.dt - Paras.ringing_start*Paras.dt)/rough_cadence)

    fig = plt.figure()
    ratios = []
    nbells = len(Data.strike_probabilities)
    significant_peaks = []; significant_proms = []
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
        #Set threshold such that the 'second round' of significant peaks are not included
        #Will still be a bit messy, unfortunately
        threshold = 0.9*sorted(prominences, reverse = True)[min(int(0.75*npeaks_ish), len(peaks)-1)]
        significant_peaks.append(np.array(peaks[prominences > threshold]))
        #plt.scatter(significant_peaks[bell], bell*np.ones(len(significant_peaks[bell])))

        significant_proms.append(np.array(prominences[prominences > threshold]))  

    print('Confidence ratios', np.array(ratios))
    #These should be slightly too many 
    best_bells = np.array([val for _, val in sorted(zip(ratios,np.arange(nbells)), reverse = True)]).astype('int')

    #Identify rhythm using these peaks -- that's the tricky bit...
    #Look for peaks which are on their own for some time and regard them as gospel, use as a basis for the rest
    allpeaks = [[] for _ in range(nbells)]; allconfs = []   #These are for the COMPLETE set of rounds. So will guess those values which don't appear nicely.
    allpeaks_guess = [[] for _ in range(nbells)]; allconfs = []   #These are for the COMPLETE set of rounds. So will guess those values which don't appear nicely.
    all_lonesome_peaks = [[] for _ in range(nbells)]
    for bi, bell in enumerate(best_bells):
        cadence_cutoff = 0.75*rough_cadence #Look for lonesome peaks
        lonesome_peaks = []
        sigpeaks = sorted(significant_peaks[bell])
        for pi in range(0,len(sigpeaks)):
            if pi == 0:
                nearest_distance = abs(sigpeaks[pi + 1] - sigpeaks[pi])
            elif pi == len(significant_peaks[bell]) - 1:
                nearest_distance = abs(sigpeaks[pi - 1] - sigpeaks[pi])
            else:
                nearest_distance = min(abs(sigpeaks[pi - 1] - sigpeaks[pi]), abs(sigpeaks[pi + 1] - sigpeaks[pi]))
            if nearest_distance > cadence_cutoff:
                lonesome_peaks.append(sigpeaks[pi])
        all_lonesome_peaks[bell] = lonesome_peaks
    
    for bi, bell in enumerate(best_bells):
        if len(all_lonesome_peaks[bell]) < 3:
            continue
        #Figure out distances between these peaks (in 'change space')
        #Don't want to rely on one bell here, take the 'best'
        #Can also probably figure out hand/back at this point, though it might be incorrect
        allpeaks.append([])
        allpeaks[bell].append(all_lonesome_peaks[bell][0])
        allpeaks_guess[bell].append(all_lonesome_peaks[bell][0])   #This is an interpolated 'guess' based on no gaps. 
        #Necessary to do this to determine the gaps, then can figure out allpeaks properly
        for li in range(1, len(all_lonesome_peaks[bell])):
            gap = int(round((all_lonesome_peaks[bell][li] - all_lonesome_peaks[bell][li-1])/rough_cadence, 0))
            for k in range(max(0, gap - 1)):
                allpeaks[bell].append(-1)
                allpeaks_guess[bell].append(all_lonesome_peaks[bell][li-1] + (k+1)*(all_lonesome_peaks[bell][li] - all_lonesome_peaks[bell][li-1])/gap)
            allpeaks[bell].append(all_lonesome_peaks[bell][li])
            allpeaks_guess[bell].append(all_lonesome_peaks[bell][li])


        if bell == 5:
            print(allpeaks_guess[bell])
        #Cut off things here as all the bells might not start at the same time (establish start time)
        #Use best_bell for this I think. Begin with the change AFTER that one as that's the only way to be certain
        if bi == 0:
            actual_start_time = min(allpeaks_guess[bell])
            allpeaks_guess[bell] = allpeaks_guess[bell][1:].copy()
            print('Actual start', actual_start_time)
        else:
            if bell < best_bells[0]: #Cut off the same as before
                allpeaks_guess[bell] = [val for val in allpeaks_guess[bell].copy() if val > actual_start_time]
            else:
                allpeaks_guess[bell] = [val for val in allpeaks_guess[bell].copy() if val > actual_start_time + rough_cadence*0.75]
    
        plt.scatter(allpeaks[bell], bell*np.ones(len(allpeaks[bell])), c = 'green', s = 20)

    rounds_start_time = min(allpeaks_guess[0])

    print('rounds start time', rounds_start_time)
    sum_1 = 0; sum_2 = 0
    for bi, bell in enumerate(best_bells[:3]):
        #At this point there are half-decent guesses for each of the rows. 
        #Now need to adjust for handstroke gaps, and redo the above step
        diff1s = np.array(allpeaks_guess[bell])[1:-1:2] - np.array(allpeaks_guess[bell])[0:-2:2]
        diff2s = np.array(allpeaks_guess[bell])[2::2] - np.array(allpeaks_guess[bell])[1:-1:2]
        #The non-confident ones won't really contribute but also won't be detrimental, so keep them in
        sum_1 += np.mean(diff1s)
        sum_2 += np.mean(diff2s)

    if sum_1 > sum_2: 
        handstroke_first = False
    else:
        handstroke_first = True
    print('Handstroke?', sum_1, sum_2, handstroke_first)
    allpeaks_betterguess = [[] for _ in range(nbells)]

    for bell in range(nbells):
        if len(all_lonesome_peaks[bell]) < 3:
            continue
        handstroke = handstroke_first

        allpeaks.append([])
        if all_lonesome_peaks[bell][0] >= rounds_start_time + 0.375*bell*rough_cadence/nbells:
            allpeaks_betterguess[bell].append(all_lonesome_peaks[bell][0])  
            handstroke = not(handstroke)
        #Necessary to do this to determine the gaps, then can figure out allpeaks properly
        for li in range(1, len(all_lonesome_peaks[bell])):
            gap = int(round((all_lonesome_peaks[bell][li] - all_lonesome_peaks[bell][li-1])/rough_cadence, 0))

            if gap > 1: 
                #Need to do some interpolation
                start = all_lonesome_peaks[bell][li-1]; end = all_lonesome_peaks[bell][li]
                position = start
                if handstroke:
                    total_interbell_gaps = (nbells*2 + 1)*(gap//2) + (gap%2)*(nbells + 1)
                else:
                    total_interbell_gaps = (nbells*2 + 1)*(gap//2) + (gap%2)*(nbells)
                avg_gap = (end - start)/total_interbell_gaps
                for k in range(gap-1):
                    #Run through and add things
                    if handstroke:
                        position += avg_gap*(nbells + 1)
                        allpeaks_betterguess[bell].append(position)  
                        handstroke = not(handstroke)
                    else:
                        position += avg_gap*(nbells)
                        allpeaks_betterguess[bell].append(position)
                        handstroke = not(handstroke)

            if all_lonesome_peaks[bell][li] >= rounds_start_time + 0.375*bell*rough_cadence/nbells:
                allpeaks_betterguess[bell].append(all_lonesome_peaks[bell][li])
                handstroke = not(handstroke)

        plt.scatter(allpeaks_betterguess[bell], bell*np.ones(len(allpeaks_betterguess[bell])), c = 'red', s = 5)


    #Start with the bells with the most peaks?
    plt.gca().invert_yaxis()
    plt.xlim(0,4000)
    st.pyplot(fig)
    plt.close()
    st.stop()

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
