import streamlit as st

import numpy as np
import time
import pandas as pd
from listen_classes import data
import matplotlib.pyplot as plt

from listen_other_functions import find_ringing_times, find_strike_probabilities
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


def establish_initial_rhythm_test(Paras, final = False):

    if not final:
        Data = data(Paras, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    else:
        Data = data(Paras, tmin = 0.0, tmax = 60.0) #This class contains all the important stuff, with outputs and things
        
    Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)

    Paras.reinforce_tmax = Paras.ringing_start*Paras.dt + Paras.reinforce_tmax

    if not final:
        st.current_log.write('Ringing detected from approx. time %d seconds' % (Paras.ringing_start*Paras.dt))

    if not final:
        Data = data(Paras, tmin = Paras.ringing_start*Paras.dt, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    else:
        Data = data(Paras, tmin = Paras.ringing_start*Paras.dt, tmax = 60.0 + Paras.ringing_start*Paras.dt) #This class contains all the important stuff, with outputs and things

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
    npeaks_ish = int((len(Data.strike_probabilities[0]) - Paras.ringing_start*Paras.dt)/rough_cadence)

    fig = plt.figure()
    ratios = []
    nbells = len(Data.strike_probabilities)
    significant_peaks = []; significant_proms = []
    for bell in range(nbells):   #Establish peaks and reliability
        probs = Data.strike_probabilities[bell][:]
        probs = gaussian_filter1d(probs, 5)
        peaks, _ = find_peaks(probs) 
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

    #Try superposing all the strike probabilities based on the assumed cadence?
    probability_shift = np.zeros(len(Data.strike_probabilities[0]))
    rough_interbell_gap = rough_cadence*2/(Paras.nbells*2 + 1)
    alpha = 1.5   #trust larger bells more by this factor
    fig = plt.figure(figsize = (10,7))
    for bell in range(nbells):
        shift = int((nbells - bell - 1)*rough_interbell_gap)
        toplot = np.zeros(len(Data.strike_probabilities[0]))
        if shift > 0:
            probability_shift[shift:] = probability_shift[shift:] + Data.strike_probabilities[bell][:-shift]*((bell + 1)/nbells)**alpha
            toplot[shift:] = Data.strike_probabilities[bell][:-shift]
        else:
            probability_shift[:] = probability_shift[:] + Data.strike_probabilities[bell]*((bell + 1)/nbells)**alpha
            toplot[shift:] = Data.strike_probabilities[bell][:]
    
    plt.plot(probability_shift, c = 'black')
    plt.plot(gaussian_filter1d(probability_shift, 3), c = 'red')
    #plt.legend()
    plt.xlim(0,7500)

    #probability_shift should give a good probability of the location of the TENOR. 
    #Keep in mind times are all now relative to the start time, whatever that is

    #Identify rhythm using these peaks -- that's the tricky bit...
    #Look for peaks which are on their own for some time and regard them as gospel, use as a basis for the rest
    shiftpeaks, _ = find_peaks(gaussian_filter1d(probability_shift, 3)) 
    shiftproms = peak_prominences(gaussian_filter1d(probability_shift, 3), shiftpeaks)[0]
    shiftpeaks = np.array([val for _, val in sorted(zip(shiftproms,shiftpeaks), reverse = True)]).astype('int')

    cadence_cutoff = 0.75*rough_cadence #Look for lonesome peaks
    lonesome_peaks = []
    npeaks_ish = int(len(probability_shift)/rough_cadence + 1)
    significant_peaks = shiftpeaks[:npeaks_ish]
    significant_peaks = sorted(significant_peaks)

    peak_threshold = 0.25*np.max(shiftproms)
    shiftpeaks = shiftpeaks[sorted(shiftproms, reverse = True) > peak_threshold]
    shiftproms = peak_prominences(gaussian_filter1d(probability_shift, 3), shiftpeaks)[0]

    for pi in range(0,len(significant_peaks)):
        if pi == 0:
            nearest_distance = abs(significant_peaks[pi + 1] - significant_peaks[pi])
        elif pi == len(significant_peaks) - 1:
            nearest_distance = abs(significant_peaks[pi - 1] - significant_peaks[pi])
        else:
            nearest_distance = min(abs(significant_peaks[pi - 1] - significant_peaks[pi]), abs(significant_peaks[pi + 1] - significant_peaks[pi]))
        if nearest_distance > cadence_cutoff:
            lonesome_peaks.append(significant_peaks[pi])
    
    if len(lonesome_peaks) < 4:
        st.error('Not found reliable enough rounds. Apologies. Try trimming audio from the start?')
        print('Not found reliable enough rounds. Apologies. Try trimming audio from the start?')
        st.session_state.test_counter += 1
        st.rerun()

    plt.scatter(lonesome_peaks,-0.25*np.ones(len(lonesome_peaks)), c = 'green')

    #Figure out distances between these peaks (in 'change space')
    #Can also probably figure out hand/back at this point, though it might be incorrect
    #Interpolate between the correct values to obtain a 'guess'
    #Use this to then determine hand/back, and improve the guess
    #Then backdate to the start of the ringing.
    #THEN search for actual peaks nearby
    #Hopefully that'll be foolproof...
    first_guesses = [lonesome_peaks[0]]

    #Necessary to do this to determine the gaps, then can figure out allpeaks properly
    for li in range(1, len(lonesome_peaks)):
        if lonesome_peaks[li] < 15.0/Paras.dt:   #Adjust for slow pull-off
            gap = int(round((lonesome_peaks[li] - lonesome_peaks[li-1])/rough_cadence - 0.25, 0))
        else:
            gap = int(round((lonesome_peaks[li] - lonesome_peaks[li-1])/rough_cadence, 0))

        for k in range(max(0, gap - 1)):
            first_guesses.append(lonesome_peaks[li-1] + (k+1)*(lonesome_peaks[li] - lonesome_peaks[li-1])/gap)
        first_guesses.append(lonesome_peaks[li])

    plt.scatter(first_guesses,-0.5*np.ones(len(first_guesses)), c = 'orange')

    #Update rough cadence:
    rough_cadence = np.mean(np.array(sorted(first_guesses))[1:] - np.array(sorted(first_guesses)[:-1]))
    print('Peal speed (hours):', Paras.dt*rough_cadence*5000/3600)

    if len(first_guesses) < 6:
        st.error('Not enough changes detected to proceed...')
        print('Not enough changes detected to proceed...')
        st.session_state.test_counter += 1
        st.rerun()
    #At this point there are half-decent guesses for each of the rows. 
    #Now need to adjust for handstroke gaps, and redo the above step
    nrows_check = int(min(6, 2*len(first_guesses)//2 - 4))
    diff1s = np.array(first_guesses)[2:][1:nrows_check:2] - np.array(first_guesses)[2:][0:nrows_check-1:2]
    diff2s = np.array(first_guesses)[2:][2:nrows_check+1:2] - np.array(first_guesses)[2:][1:nrows_check:2]

    if np.mean(diff1s) < np.mean(diff2s): 
        handstroke_first = True
    else:
        handstroke_first = False
    stroke_difference = abs(np.mean(diff2s) - np.mean(diff1s))

    handstroke = handstroke_first   #This is the stroke of the next change 
    second_guesses = [lonesome_peaks[0]]
    handstroke = not(handstroke)
    #Necessary to do this to determine the gaps, then can figure out allpeaks properly
    for li in range(1, len(lonesome_peaks)):
        if lonesome_peaks[li] < 15.0/Paras.dt:   #Adjust for slow pull-off
            gap = int(round((lonesome_peaks[li] - lonesome_peaks[li-1])/rough_cadence - 0.25, 0))
        else:
            gap = int(round((lonesome_peaks[li] - lonesome_peaks[li-1])/rough_cadence, 0))

        if gap > 1: 
            #Need to do some interpolation
            start = lonesome_peaks[li-1]; end = lonesome_peaks[li]
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
                    second_guesses.append(position)  
                    handstroke = not(handstroke)
                else:
                    position += avg_gap*(nbells)
                    second_guesses.append(position)
                    handstroke = not(handstroke)

        second_guesses.append(lonesome_peaks[li])
        handstroke = not(handstroke)

    #Almost done -- now append to the start, and find the 'final' guesses for the tenor times
    handstroke = handstroke_first
    while min(shiftpeaks) < min(second_guesses) - 0.75*rough_cadence and min(second_guesses) - 1.5*rough_cadence > Paras.ringing_start:
        #Check for strikes which are closer to the start than we've already looked at (unsteady rounds problem)
        if handstroke:
            second_guesses.insert(0,second_guesses[0]-rough_interbell_gap*(nbells + 1))
        else:
            second_guesses.insert(0,second_guesses[0]-rough_interbell_gap*(nbells))
        handstroke = not(handstroke)

    handstroke_first = handstroke   #This is actually the stroke of the first recorded change, rather than the first reliable one
    #BUT if there is silence beforehand and not much confidence, it's safe to assume it's handstroke first probably

    if Paras.ringing_start > rough_cadence * 1.5 and stroke_difference*Paras.dt < 10.0:
        handstroke_first = True

    #Check for peaks close to these assumed times (deals with some variation in ringing time). Just pick the most prominent within a reasonable range, if such a thing exists
    final_guesses = []
    for strike in second_guesses:
        min_limit = strike - (nbells/3)*rough_interbell_gap
        max_limit = strike + (nbells/3)*rough_interbell_gap
        options = [val for val in shiftpeaks if min_limit <= val <= max_limit]
        if len(options) == 0:
            final_guesses.append(strike)
        else:
            prom_options = peak_prominences(gaussian_filter1d(probability_shift, 3), options)[0]
            index = np.argmax(prom_options)
            final_guesses.append(options[index])

    plt.scatter(second_guesses, -0.75*np.ones(len(second_guesses)), c = 'blue')
    plt.scatter(final_guesses, -1.0*np.ones(len(second_guesses)), c = 'red')
    plt.close()
    
    handstroke = handstroke_first

    cadences = []; init_aims = []
    cadence_guess = (final_guesses[2] - final_guesses[0])/(2*Paras.nbells + 1)
    belltimes = np.linspace(final_guesses[0] - cadence_guess*(nbells-1), final_guesses[0], Paras.nbells)
    belltimes = belltimes[-Paras.nbells:]    
    init_aims.append(belltimes)

    for ri in range(0,len(final_guesses)-1):
        #Interpolate the bells smoothly (assuming steady rounds)

        if handstroke:
            belltimes = np.linspace(final_guesses[ri], final_guesses[ri + 1], Paras.nbells + 1)
        else:
            belltimes = np.linspace(final_guesses[ri], final_guesses[ri + 1], Paras.nbells + 2)
            
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


    tcut = 1 #Be INCREDIBLY fussy with these picks or the wrong ones will get picked
    absolute_cutoff = cadence*0.5
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
                if abs(poss[k] - aim) < absolute_cutoff:
                    tvalue = 1.0/(abs(poss[k] - aim)/tcut + 1)**Paras.strike_alpha
                else:
                    tvalue = 0
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


    #If the first change is before the ringing time, ged rid of it (make sure to change stroke)
    if np.min(strikes) < 0:
        strikes = strikes[:,1:]
        strike_certs = strike_certs[:,1:]
        Paras.nrounds_max = Paras.nrounds_max - 1
        handstroke = not(handstroke)

    #print('Change ends', np.array(final_guesses[:5]) + Paras.ringing_start)
    #print(strikes[:,:5] + Paras.ringing_start)

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
    
    row_confidences = np.mean(strike_certs, axis = 0)
    bell_confidences = np.mean(strike_certs, axis = 1)
    #print('Row confidences', row_confidences)
    #print('Bell confidences', bell_confidences)
    #Check this is indeed handstroke or not, in case of an oddstruck tenor
    diff1s = strikes[:,1:-1:2] - strikes[:,0:-2:2]
    diff2s = strikes[:,2::2] - strikes[:,1:-1:2]
    
    if np.mean(diff1s[:]) < np.mean(diff2s[:]):
        handstroke_first = True
    else:
        handstroke_first = False

    #print('Handstroke first?', handstroke_first)

    #Filter out rows which probably aren't rounds from these results -- otherwise bells will get mixed up for sure
    #Establish a standard based on all the bells which aren't the tenor
    navg = min(8, 2*len(strike_certs)//2)
    stroke1_standard = np.mean(strike_certs[:-1,2:navg:2])
    stroke2_standard = np.mean(strike_certs[:-1,3:navg:2])
    upto_standard = np.zeros(Paras.nrounds_max)
    last_rounds_change = 0; offrounds_count = 0
    for ri in range(Paras.nrounds_max):
        if ri%2 == 0:
            if np.mean(strike_certs[:-1,ri]) > 0.5*stroke1_standard:
                upto_standard[ri] = 1.
                offrounds_count = 0
        else:
            if np.mean(strike_certs[:-1,ri]) > 0.5*stroke2_standard:
                upto_standard[ri] = 1.
                offrounds_count = 0
        if upto_standard[ri] > 0.5:
            last_rounds_change = ri
        elif np.sum(upto_standard) > 0:
            offrounds_count += 1
        if offrounds_count > 3 or (offrounds_count > 1 and ri > 10):
            break
    if last_rounds_change < 6:   #Otherwise it's probably not going to work...
        last_rounds_change = 5

    strikes = np.array(strikes[:,:last_rounds_change+1])
    strike_certs = np.array(strike_certs[:,:last_rounds_change+1])
    
    Data.handstroke_first = handstroke_first
    st.session_state.handstroke_first = handstroke_first

    #Determine how many rounds there actually are? Nah, it's probably fine...
    Paras.first_change_start = np.min(strikes[:,0])
    Paras.first_change_end = np.max(strikes[:,0])

    return strikes, strike_certs
