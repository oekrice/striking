# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 10:49:58 2025

@author: eleph
"""
import streamlit as st

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
import time

def test_error(text):
    print(text)
    st.session_state.test_counter += 1
    st.rerun()

#Added text so Streamlit detects a commit again again
def find_colour(value):
    #For prettiness purposes
    colour_thresholds = [0.95,0.98]; colours = ['red', 'orange', 'green']
    c = colours[0]
    if value > colour_thresholds[0]:
        c = colours[1]
    if value > colour_thresholds[1]:
        c = colours[2]
    return  c
    
def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))

def check_initial_rounds(strikes):
    isfines = np.zeros(len(strikes[0]))
    ncheck = 6
    for ri in range(len(strikes[0])):
        row = strikes[:,ri]
        diffs = row[1:] - row[:-1]
        mean_diff = np.mean(diffs)
        if np.min(diffs) < mean_diff*0.25:
            isfines[ri] = 0
        else:
            isfines[ri] = 1
    if np.sum(isfines[:ncheck])/ncheck < 0.5:
        return False
    else:
        return True


def find_ringing_times(Paras, Data):
    
    freq_int_max = round(5000*Paras.fcut_length)
    
    loudness = Data.transform[:,:freq_int_max]
    loudness = loudness
    
    loudness = gaussian_filter1d(loudness, round(0.1/Paras.dt),axis = 0)
    loudsum = np.sqrt(np.sum(loudness, axis = 1))

    loudsmooth = gaussian_filter1d(loudsum, round(2.0/Paras.dt), axis = 0)
    loudsmooth[0] = 0.0; loudsmooth[-1] = 0.0 #For checking peaks
    
    threshold = np.max(loudsmooth)*0.8
    #Use this to determine the start time of the ringing -- time afte
    peaks, _= find_peaks(loudsmooth, width = round(10.0/Paras.dt))  #Prolonged peak in noise - probably ringing
    
    if len(peaks) == 0:
        return 0, len(Data.transform)
    #I can't find an inbuilt finctino to do this, bafflingly
    peak = sorted(peaks)[-1]
    rlim = peak; llim = peak
    while rlim < len(loudsmooth):
        if loudsmooth[rlim] > threshold:
            rlim = rlim + 1
        else:
            break
        
    while llim > 0:
        if loudsmooth[llim] > threshold:
            llim = llim - 1
        else:
            break
    start_time = llim
    end_time   = rlim
    
    start_time = round(max(0, start_time - 3.0/Paras.dt))
    del loudness; del loudsmooth; del loudsum; del peaks 
    return start_time, end_time

def find_rough_cadence(Paras, Data):
    #Takes the first 20 seconds or so and smooths to hell, finds peaks and figures out when times could be.
    #Probably reasonable to use the tenor here? Or just all
    #REDO this for loudness, rather than fancy probability things. Ian said so.
    #Need to isolate for frequencies though
    cadences = []
    #fig = plt.figure()

    for bell in range(Paras.nbells):
        freq_test = Data.test_frequencies[bell]

        loudness = Data.transform[:,freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        
        loudness = gaussian_filter1d(loudness, int(0.1/Paras.dt),axis = 0)
        loudsum = np.sum(loudness, axis = 1)
        loudsmooth = gaussian_filter1d(loudsum, int(0.5/Paras.dt), axis = 0)
        #plt.plot(loudsmooth, label = bell)

        back_bell_cutoff = max(2,int(len(Data.strike_probabilities)/4))

        peaks, _ = find_peaks(loudsmooth) 
        if len(peaks) < 2:
            return np.nan
        avg_peak_distance = np.sum(peaks[1:] - peaks[:-1])/len(peaks[1:])
        cadences.append(avg_peak_distance)
    #plt.close()
    #st.pyplot(fig)
    return np.mean(cadences[-back_bell_cutoff:])


def find_first_strikes(Paras, Data):
    #Takes normalised wave vector, and does some fourier things
    #This function is the one which probably needs improving the most...

    #Want to plot prominences and things to check if these is an obvious pattern
    rough_cadence = find_rough_cadence(Paras, Data)   #This gives an impression of how long it is between successive changes
    if np.isnan(rough_cadence):
        st.error("Can't detect any ringing... Apologies")
        if st.session_state.testing_mode:
            test_error("Can't detect any ringing... Apologies")
        else:
            st.stop()

    npeaks_ish = int((len(Data.strike_probabilities[0]) - Paras.ringing_start*Paras.dt)/rough_cadence)

    #fig = plt.figure()
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
    #fig = plt.figure(figsize = (10,7))
    for bell in range(nbells):
        shift = int((nbells - bell - 1)*rough_interbell_gap)
        toplot = np.zeros(len(Data.strike_probabilities[0]))
        if shift > 0:
            probability_shift[shift:] = probability_shift[shift:] + Data.strike_probabilities[bell][:-shift]*((bell + 1)/nbells)**alpha
            toplot[shift:] = Data.strike_probabilities[bell][:-shift]
        else:
            probability_shift[:] = probability_shift[:] + Data.strike_probabilities[bell]*((bell + 1)/nbells)**alpha
            toplot[shift:] = Data.strike_probabilities[bell][:]
    
    #plt.plot(probability_shift, c = 'black')
    #plt.plot(gaussian_filter1d(probability_shift, 3), c = 'red')
    #plt.legend()
    #plt.xlim(0,7500)

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
        if st.session_state.testing_mode:
            test_error('Not found reliable enough rounds. Apologies. Try trimming audio from the start?')
        else:
            st.stop()


    #plt.scatter(lonesome_peaks,-0.25*np.ones(len(lonesome_peaks)), c = 'green')

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

    #plt.scatter(first_guesses,-0.5*np.ones(len(first_guesses)), c = 'orange')

    #Update rough cadence:
    rough_cadence = np.mean(np.array(sorted(first_guesses))[1:] - np.array(sorted(first_guesses)[:-1]))
    #print('Peal speed (hours):', Paras.dt*rough_cadence*5000/3600)

    if len(first_guesses) < 6:
        st.error('Not enough changes detected to proceed...')
        if st.session_state.testing_mode:
            test_error('Not enough changes detected to proceed...')
        else:
            st.stop()
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

    #plt.scatter(second_guesses, -0.75*np.ones(len(second_guesses)), c = 'blue')
    #plt.scatter(final_guesses, -1.0*np.ones(len(second_guesses)), c = 'red')
    #plt.close()
    
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


def find_strike_times(Paras, Data, final = False):
    #Go through the rounds in turn instead of doing it bellwise
    #Allows for nicer plotting and stops mistakely hearing louder bells. Hopefully.
        
    #Determine whether this is the start of the ringing or not... Actually, disregard the first change whatever. Usually going to be wrong...
    
    allstrikes = []; allconfs = []; allcadences = []
            
    start = 0; end = 0
    
    tcut = Paras.rounds_tcut*round(Paras.cadence)

    strike_probs = Data.strike_probabilities

    #Obtain adjusted probs
    strike_probs_adjust = np.zeros(strike_probs.shape)
    strike_probs_adjust = strike_probs[:, :]**(Paras.probs_adjust_factor + 1)/(np.sum(strike_probs[:,:], axis = 0) + 1e-6)**Paras.probs_adjust_factor
    
    strike_probs_adjust = gaussian_filter1d(strike_probs_adjust, Paras.rounds_probs_smooth, axis = 1)

    allpeaks = []; allbigs = []; allsigs = []
    for bell in range(Paras.nbells):
        
        probs = strike_probs_adjust[bell]  

        probs_smooth = 1.0*gaussian_filter1d(probs, round(Paras.smooth_time/Paras.dt))

        peaks, _ = find_peaks(probs)
        
        if len(Paras.allstrikes) == 0 and not final:
            peaks = peaks[peaks > np.min(Paras.first_strikes) - round(1.0/Paras.dt)]
        
        prominences = peak_prominences(probs, peaks)[0]
        
        bigpeaks = peaks[prominences > 0.5*probs_smooth[peaks]]  #For getting first strikes, need to mbe more significant
        peaks = peaks[prominences > 0.1*probs_smooth[peaks]]

        sigs = peak_prominences(probs, peaks)[0]/probs_smooth[peaks]
        
        sigs = sigs/np.max(sigs)
        
        allpeaks.append(peaks); allbigs.append(bigpeaks); allsigs.append(sigs)

    #Find all peaks to begin with
    #Run through each set of rounds 
    handstroke = Data.handstroke_first
    #st.write('Test', handstroke, st.session_state.handstroke_first, len(Paras.allstrikes))
    next_end = 0
    
    count = 0
    
    if len(Paras.allstrikes) == 0:
        taims = np.zeros(Paras.nbells)
    else:
        change_start = np.mean(Data.last_change) - Data.cadence_ref*((Paras.nbells - 1)/2)
        change_end = np.mean(Data.last_change) + Data.cadence_ref*((Paras.nbells - 1)/2)
        
        rats = (Data.last_change - change_start)/(change_end - change_start)
        if not handstroke:
            taims  = np.array(Data.last_change) + round(Paras.nbells*Data.cadence_ref)
            next_start = change_start + round(Paras.nbells*Data.cadence_ref)
            next_end = change_end + round(Paras.nbells*Data.cadence_ref)
        else:
            taims  = np.array(Data.last_change) + round((Paras.nbells + 1)*Data.cadence_ref)
            next_start = change_start + round((Paras.nbells+1)*Data.cadence_ref)
            next_end = change_end + round((Paras.nbells+1)*Data.cadence_ref)

        taims = next_start + (next_end - next_start)*rats
                                   
        start = next_start - 3.0*round(Data.cadence_ref)
        end  =  next_end   + 3.0*round(Data.cadence_ref)

    go = True
    while go:
        strikes = np.zeros(Paras.nbells)
        confs = np.zeros(Paras.nbells)
        certs = np.zeros(Paras.nbells) #To know when to stop
        
        count += 1
        peaks_range = []; sigs_range = []
        if len(Paras.allstrikes) == 0 and len(allstrikes) < 4:  #Establish first strikes overall.
            #IMPROVE ON THIS - DETERMINE FROM THE INIT AUDIO BIT
            #print(Paras.first_strikes[:,0])
            change_number = len(allstrikes)
            for bell in range(Paras.nbells): #This is a bit shit -- improve it?
                #taim = Paras.first_change_start + Paras.cadence*bell
                taim = Paras.first_strikes[bell,change_number]
                start_bell = taim - round(3.5*Paras.cadence)  #Aim within the change
                end_bell = taim + round(3.5*Paras.cadence)

                poss = allpeaks[bell][(allpeaks[bell] > start_bell)*(allpeaks[bell] < end_bell)]
                sigposs = allsigs[bell][(allpeaks[bell] > start_bell)*(allpeaks[bell] < end_bell)]
                
                poss = np.array([val for _, val in sorted(zip(sigposs, poss), reverse = True)])
    
                if len(poss) < 1:
                    strikes[bell] = taim
                    confs[bell] = 0.0
                else:
                    strikes[bell] = poss[0]
                    confs[bell] = 1.0
                    
                del poss; del sigposs
        else:  #Find options in the correct range
            for bell in range(Paras.nbells):
                peaks = allpeaks[bell]
                sigs = allsigs[bell]
                
                peaks_range = peaks[(peaks > start)*(peaks < end)]
                sigs_range = sigs[(peaks > start)*(peaks < end)]
                
                start_bell = taims[bell] - round(3.5*Paras.cadence)  #Aim within the change
                end_bell = taims[bell] + round(3.5*Paras.cadence)
                #Check physically possible...
                if len(allstrikes) == 0:
                    start_bell = max(start_bell, Data.last_change[bell] + round(3.0*Paras.cadence))
                else:
                    start_bell = max(start_bell, allstrikes[-1][bell] + round(3.0*Paras.cadence))
                    
                sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]

                if len(peaks_range) == 1:   #Only one time that it could reasonably be
                    strikes[bell] = peaks_range[0]
                    tvalue = 1.0/(abs(peaks_range[0] - taims[bell])/tcut + 1)**Paras.strike_alpha
                    if final:
                        confs[bell]  = 1.0
                    else:
                        confs[bell] = 1.0  #Timing doesn't really matter, but prominence does -- don't want ambiguity
                    certs[bell] = tvalue*sigs_range[0]/np.max(sigs)
                    
                elif len(peaks_range) > 1:
                                          
                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        if abs(peaks_range[k] - taims[bell]) < round(Paras.rounds_leeway*Paras.cadence):
                            tvalue = 1.0
                        else:
                            tvalue = 1.0/(abs(abs(peaks_range[k] - taims[bell]) - round(Paras.rounds_leeway*Paras.cadence))/tcut + 1)**(Paras.strike_alpha)
                            
                        if final:
                            yvalue = sigs_range[k]
                        else:
                            yvalue = sigs_range[k]
                            
                        scores.append(tvalue*sigs_range[k]/np.max(sigs))
                                                
                        
                    kbest = scores.index(max(scores))
                    
                    strikes[bell] = peaks_range[kbest]
                    if final:
                        confs[bell] = (sigs_range[kbest]/np.sum(sigs_range))**2
                    else:
                        confs[bell] = (sigs_range[kbest]/np.sum(sigs_range))**2
                        
                    certs[bell] = scores[kbest]
                            
                else:
                    #Pick best peak in the change? Seems to work when things are terrible
                    peaks = allpeaks[bell]
                    sigs = allsigs[bell]
                    peaks_range = peaks[(peaks > start)*(peaks < end)]
                    sigs_range = sigs[(peaks > start)*(peaks < end)]

                    start_bell = np.min(taims)
                    end_bell = np.max(taims)

                    sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                    peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]

                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        tvalue = 1.0/(abs(peaks_range[k] - np.mean(taims))/tcut + 1)**Paras.strike_alpha
                        yvalue = sigs_range[k]/np.max(sigs_range)
                        scores.append(tvalue*yvalue**2.0)
                        
                    if len(scores) > 0:
                        kbest = scores.index(max(scores))
                        
                        strikes[bell] = peaks_range[kbest]
                        confs[bell] = 0.0
                        certs[bell] = 0.0

                    else:
                        #Pick average point in the change

                        strikes[bell] = round(0.5*(start + end))
                        confs[bell] = 0.0
                        certs[bell] = 0.0
                     
                    
            if np.median(certs) < 0.01:
                
                strikes = []   #Not confident enough -- the difference between certs and confs was lost on me here...
                if len(allconfs) > 1:
                    bellconfs_individual = np.mean(np.array(allconfs)[1:,:], axis = 0)

                Data.freq_data = np.array([Paras.dt, Paras.fcut_length, 0., 0.])
                Data.freq_data = np.concatenate((Data.freq_data, np.zeros(Paras.nbells)))

        if len(strikes) > 0 and np.max(strikes) - np.min(strikes) > 10:
            if np.median(confs) > 0.5 or len(allstrikes) == 0:
                allstrikes.append(strikes)
                allconfs.append(confs)
            else:
                go = False
        else:
            go = False
            continue
         
        #Determine likely location of the next change END
        #Need to be resilient to method mistakes etc... 
        #Log the current avg. bell cadences
        allcadences.append((max(strikes) - min(strikes))/(Paras.nbells - 1))     

        nrows_count = int(min(len(allcadences), 20))
        Data.cadence_ref = np.mean(allcadences[-nrows_count:])

        change_start = np.mean(strikes) - Data.cadence_ref*((Paras.nbells - 1)/2)
        change_end = np.mean(strikes) + Data.cadence_ref*((Paras.nbells - 1)/2)

        rats = (strikes - change_start)/(change_end - change_start)
                
        if handstroke:
            taims  = np.array(allstrikes[-1]) + round(Paras.nbells*Data.cadence_ref)
            next_start = change_start + round(Paras.nbells*Data.cadence_ref)
            next_end = change_end + round(Paras.nbells*Data.cadence_ref)
        else:
            taims  = np.array(allstrikes[-1]) + round((Paras.nbells + 1)*Data.cadence_ref)
            next_start = change_start + round((Paras.nbells+1)*Data.cadence_ref)
            next_end = change_end + round((Paras.nbells+1)*Data.cadence_ref)

        taims = next_start + (next_end - next_start)*rats
                   
        handstroke = not(handstroke)
                
        start = next_start - 1.5*round(Data.cadence_ref)
        end  =  next_end   + 3.5*round(Data.cadence_ref)

        if not Data.end_flag:
            if end > len(Data.strike_probabilities[0]) - 2.5/Paras.dt:   #This is nearing the end of the reasonable time for this cut, so don't bother any more. UNLESS it's the final one
                go = False

    if len(allconfs) > 1:
        
        bellconfs_individual = np.mean(np.array(allconfs)[1:,:], axis = 0)
        Data.freq_data = np.array([Paras.dt, Paras.fcut_length, np.mean(allconfs[1:]), np.min(allconfs[1:])])
        Data.freq_data = np.concatenate((Data.freq_data, bellconfs_individual))
        
    if len(allstrikes) < 2:
        #Paras.ringing_finished = True
        return [], []
    
    spacings = 1e6*np.ones((len(allstrikes), Paras.nbells, 2))
    yvalues = np.arange(Paras.nbells)
        
    for ri, row in enumerate(allstrikes):
        #Sort out ends
        order = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
        for si in range(len(row)):
            if si == 0:
                if ri == 0:
                    spacings[ri,order[si],0] = Paras.cadence*2
                else:
                    spacings[ri,order[si],0] = row[order[si]] - np.max(allstrikes[ri-1])
            else:
                spacings[ri,order[si],0] = row[order[si]] - row[order[si-1]]
            
            if si == len(row)- 1:    
                if ri == len(allstrikes) - 1:
                    spacings[ri,order[si],1] =  Paras.cadence*2
                else:
                    spacings[ri,order[si],1] =  np.min(allstrikes[ri+1])  - row[order[si]]  
            else:
                spacings[ri,order[si],1] = row[order[si+1]] - row[order[si]]
      
    if not final:
        allconfs = allconfs*(np.min(spacings, axis = 2)/ np.max(spacings))
            
    del allpeaks; del allsigs; del peaks_range; del sigs_range
    del probs; del probs_smooth

    nstrikes_done = len(allstrikes)

    if nstrikes_done < len(Paras.first_strikes[0]):
         Paras.first_strikes[:,:nstrikes_done] = np.array(allstrikes).T
    else:
         Paras.first_strikes[:,:] = np.array(allstrikes).T[:,:len(Paras.first_strikes[0])]

    return np.array(allstrikes).T, np.array(allconfs).T   

def check_for_misses(allstrikes, allcerts, last_switch):
    #This function checks if a bell has found its way into the wrong change. If so it will produce the 'correct' strikes up to that point and disregard the rest. One hopes.
    #Can look through allstrikes in its entirety!  Will need to implement some kind of checker to stop it getting stuck in a loop

    #Two ways to check -- either have the 'rounds problem' of a bell staying at the front/back too long
    #OR look for an outlier in the cadences, and work backwards until the last point at which that bell wasn't at the respective end
    #The former seems to probably be the better approach
    allrows = np.array(allstrikes)
    nrows_cut = len(allrows)
    nbells = len(allrows[0])
    yvalues = np.arange(nbells) + 1
    allcadences = []
    prev_order = yvalues
    ndiffs = 0
    for ri, row in enumerate(allrows):
        allcadences.append((np.max(row) - np.min(row))/(nbells-1))
        order = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
        ndiffs += (1 - len(np.where(order == prev_order)[0])/nbells)
        prev_order = order
    if ndiffs/len(allrows) < 0.2:   #This is probably just rounds and calls -- leave it be
        return allstrikes, allcerts, last_switch
    
    #Try running through to get an overall maxcounts
    maxcounts = np.zeros((len(allrows), nbells), dtype = int)   #Counts of bells in each position (other than rounds)
    prev_order = yvalues
    orders = []; lengths = []
    for ri, row in enumerate(allrows):
        order = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
        #print(ri, order)
        orders.append(order)
        lengths.append(np.max(row) - np.min(row))
        if (not (order == yvalues).all()) and ri > 2:   #Obviously rounds is probably fine
            sameplaces = np.where(order == prev_order)[0]
            diffplaces = [val for val in yvalues-1 if val not in sameplaces]
            maxcounts[ri, sameplaces] = maxcounts[ri - 1][sameplaces] + 1
            if order[-1] == nbells-1:   #Account for tenor behind
                maxcounts[ri,-1] = 0
            maxcounts[ri, diffplaces] = 0
        else:
            maxcounts[ri] = np.zeros(nbells, dtype = int)
        prev_order = order
    navg = min(len(allrows), 10)
    avg_length = np.mean(np.array(lengths)[:navg])
    avg_cadence = avg_length/(nbells - 1)
    static_counts = np.zeros(len(allrows), dtype = int)
    prev_order = yvalues
    for ri, length in enumerate(lengths):
        row = allrows[ri]
        order = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
        if (order == prev_order).all() and ri > 2 and (not (order == yvalues).all()):   
            static_counts[ri] = static_counts[ri-1] + 1
        prev_order = order
    
    def find_lengths(row_section):
        lengths = np.zeros(len(row_section))
        for ril, row in enumerate(row_section):
            lengths[ril] = np.max(row) - np.min(row)
        return lengths

    def lookfor_rounds(min_index, static_counts, allrows):
        #Checks for a rotation of rounds after the min_index threshold.
        isnotrounds = False
        if np.max(static_counts) > 2:
            #This may be some rotated rounds -- have a look for such.
            for fi_rounds, roundsbutnot_count in enumerate(static_counts):
                if roundsbutnot_count  > 2 and fi_rounds >= min_index: #In the same position for three blows. Take this one as the first occasion.
                    isnotrounds = True
                    for backward_count in range(fi_rounds,0,-1):
                        if static_counts[backward_count] == 0:
                            start_index = backward_count
                            break
                    for forward_count in range(fi_rounds, len(static_counts),1):
                        if static_counts[forward_count] > 0:
                            end_index = forward_count
                        else:
                            break
                    #print('rounds switch range', fi_rounds, start_index, end_index)
                    break
        #With this information, check for bells before the tenor
        if isnotrounds:
            reference_rows = allrows[start_index:end_index]  #Take one fewer than the end in case this does go as far as the end
            bells_after = reference_rows[0] > reference_rows[0,-1]
            shift_rows = reference_rows.copy()
            if np.sum(bells_after) < nbells/2:
                #Use an earlier time for these bells
                shift_rows[:,bells_after] = allrows[start_index-1:end_index-1, bells_after]
            else:
                bells_before = np.invert(bells_after)
                shift_rows[:,bells_before] = allrows[start_index+1:end_index+1, bells_before]
            #Test these orders and if they're fine, continue without anything else. Should do a stroke checker too maybe? Will check with Saltburn.
            new_orders = np.zeros(np.shape(shift_rows))
            roundscount = 0
            for rir, row in enumerate(shift_rows):
                new_orders[rir] = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
                if (new_orders[rir] == yvalues).all():
                    roundscount += 1
            if roundscount/(len(shift_rows) - 1) > 1.0:
                #Trim correct allstrikes down
                return True, start_index, shift_rows[0]   #Flag for genuine shift, index if so and the rows which have shifted
            else:
                return False, end_index, shift_rows[0]
        else:
            return False, -1, allrows[0]

    def lookfor_front(min_index, maxcounts, allrows):

        #THREE options: Either look for front, back or reset to rounds. Whichever is first or more prominent will be chosen for action. Do not do more than one!
        isfront = False
        if np.max(maxcounts[:,0]) > 2:
            for fi_front, front_count in enumerate(maxcounts[:,0]):
                if front_count > 2 and fi_front >= min_index: #In the same position for three blows. Take this one as the first occasion.
                    isfront = True
                    for backward_count in range(fi_front,0,-1):
                        if maxcounts[backward_count,0] == 0:
                            start_index = backward_count
                            break
                    for forward_count in range(fi_front, len(maxcounts),1):
                        if maxcounts[forward_count,0] > 0:
                            end_index = forward_count
                        else:
                            break
                    #print('front switch range', fi_front, start_index, end_index)
                    break
        #Now need to rate these options based on swapping things around
        front_diff = 0
        if isfront:
            #Determine reference rows etc.
            reference_rows = allrows[start_index:end_index]  #Take one fewer than the end in case this does go as far as the end
            shift_bell = int(orders[start_index][0] - 1)
            #print('front shift bell', shift_bell)
            shift_rows = reference_rows.copy()
            shift_rows[:,shift_bell] = allrows[start_index+1:end_index+1, shift_bell]
            old_lengths = find_lengths(reference_rows)
            new_lengths = find_lengths(shift_rows)
            front_diff = np.mean(new_lengths[1:]) - np.mean(old_lengths[1:])
            #print('front difference', front_diff)
            if front_diff < -avg_cadence*1.25:
                return True, start_index, shift_rows[0]   #Flag for genuine shift, index if so and the rows which have shifted
            else:
                return False, end_index, shift_rows[0]
        else:
            return False, -1, allrows[0]

    def lookfor_back(min_index, maxcounts, allrows):
        isback = False
        if np.max(maxcounts[:,-1]) > 2:
            for fi_back, back_count in enumerate(maxcounts[:,-1]):
                if back_count  > 2 and fi_back >= min_index: #In the same position for three blows. Take this one as the first occasion.
                    isback = True
                    for backward_count in range(fi_back,0,-1):
                        if maxcounts[backward_count,-1] == 0:
                            start_index = backward_count
                            break
                    for forward_count in range(fi_back, len(maxcounts),1):
                        if maxcounts[forward_count,-1] > 0:
                            end_index = forward_count
                        else:
                            break
                    #print('back switch range', fi_back, start_index, end_index)
                    break

        #Now need to rate these options based on swapping things around
        back_diff = 0
        if isback:
            #Determine reference rows etc.
            reference_rows = allrows[start_index:end_index]  #Take one fewer than the end in case this does go as far as the end
            shift_bell = int(orders[start_index][-1] - 1)
            #print('back shift bell', shift_bell)
            shift_rows = reference_rows.copy()
            shift_rows[:,shift_bell] = allrows[start_index-1:end_index-1, shift_bell]
            old_lengths = find_lengths(reference_rows)
            new_lengths = find_lengths(shift_rows)
            back_diff = np.mean(new_lengths[1:]) - np.mean(old_lengths[1:])
            #print('back difference', back_diff)
            if back_diff < -avg_cadence*1.25:
                return True, start_index, shift_rows[0]   #Flag for genuine shift, index if so and the rows which have shifted
            else:
                return False, end_index, shift_rows[0]
        else:
            return False, -1, allrows[0]

    #Container for looking for rounds
    gorounds = True
    rounds_switch_flag, rounds_switch_index, rounds_shift_row = False, -1, allrows[0]
    min_index = last_switch
    while gorounds:
        switch_flag, switch_index, shift_row = lookfor_rounds(min_index, static_counts, allrows)
        if switch_flag:  #This is a genuine switch -- exit the main function? Depends if theres a different genuine switch first!
            rounds_switch_flag, rounds_switch_index, rounds_shift_row = switch_flag, switch_index, shift_row 
            gorounds = False
        elif switch_index > 0:   #There was a potential switch but no good, so keep looking
            min_index = switch_index + 1
        else:
            gorounds = False
    #And back
    goback = True
    back_switch_flag, back_switch_index, back_shift_row = False, -1, allrows[0]
    min_index = last_switch
    while goback:
        switch_flag, switch_index, shift_row = lookfor_back(min_index, maxcounts, allrows)
        if switch_flag:  #This is a genuine switch -- exit the main function? Depends if theres a different genuine switch first!
            back_switch_flag, back_switch_index, back_shift_row = switch_flag, switch_index, shift_row 
            goback = False
        elif switch_index > 0:   #There was a potential switch but no good, so keep looking
            min_index = switch_index + 1
        else:
            goback = False
    #And front
    gofront = True
    front_switch_flag, front_switch_index, front_shift_row = False, -1, allrows[0]
    min_index = last_switch
    while gofront:
        switch_flag, switch_index, shift_row = lookfor_front(min_index, maxcounts, allrows)
        if switch_flag:  #This is a genuine switch -- exit the main function? Depends if theres a different genuine switch first!
            front_switch_flag, front_switch_index, front_shift_row = switch_flag, switch_index, shift_row 
            gofront = False
        elif switch_index > 0:   #There was a potential switch but no good, so keep looking
            min_index = switch_index + 1
        else:
            gofront = False

    #print('Rounds', rounds_switch_flag, rounds_switch_index, rounds_shift_row)
    #print('Back', back_switch_flag, back_switch_index, back_shift_row)
    #print('Front', front_switch_flag, front_switch_index, front_shift_row)

    #Rounds takes priority a little -- if within 5 changes of the others perhaps. 
    if not rounds_switch_flag:
        rounds_switch_index = 100000
    if not front_switch_flag:
        front_switch_index = 100000
    if not back_switch_flag:
        back_switch_index = 100000
    rounds_switch_test = rounds_switch_index - 5

    if rounds_switch_flag and rounds_switch_test <= min(back_switch_index, front_switch_index):
        #print('Switching rounds')
        allstrikes = allstrikes[:rounds_switch_index+1]
        allcerts = allcerts[:rounds_switch_index+1]
        allstrikes[rounds_switch_index] = rounds_shift_row + 2
        #allcerts[rounds_switch_index] = 0.0
        last_switch = rounds_switch_index

    elif back_switch_flag and back_switch_index <= min(front_switch_index, rounds_switch_test):
        #print('Switch back')
        allstrikes = allstrikes[:back_switch_index+1]
        allcerts = allcerts[:back_switch_index+1]
        allstrikes[back_switch_index] = back_shift_row + 2
        #allcerts[back_switch_index][0] = 0.0
        last_switch = back_switch_index

    elif front_switch_flag and front_switch_index <= min(back_switch_index, rounds_switch_test):
        #print('Switch front')
        allstrikes = allstrikes[:front_switch_index+1]
        allcerts = allcerts[:front_switch_index+1]
        allstrikes[front_switch_index] = front_shift_row + 2
        #allcerts[front_switch_index][0] = 0.0
        last_switch = front_switch_index

    return allstrikes, allcerts, last_switch

def do_frequency_analysis(Paras, Data):
    #Now takes existing strikes data to do this (to make reinforcing easier)
    #__________________________________________________
    #Takes strike times and reinforces the frequencies from this. Needs nothing else, so works with the rounds too
     
    tcut = round(Data.cadence*Paras.freq_tcut) #Peak error diminisher
    
    freq_tests = np.arange(0, len(Data.transform[0])//4)
    nstrikes = len(Data.strikes[0])
    allprobs = np.zeros((len(freq_tests), Paras.nbells))
    allvalues = np.zeros((len(freq_tests), len(Data.strikes[0]), Paras.nbells))
        
    bellsums = np.zeros(Paras.nbells)
    for bell in range(Paras.nbells):
        max_cert_threshold = sorted(Data.strike_certs[bell,:])[-4]   #Get rid of massive outliers
        Data.strike_certs[bell,:] = np.minimum(Data.strike_certs[bell,:], max_cert_threshold)
        bellsums[bell] = np.sum(Data.strike_certs[bell,:]**Paras.beta)
        #Need to adjust these for the case where there is literally zero certainty (as for Hursley)
        if np.sum(bellsums[bell]) > 0.0:
            Data.strike_certs[bell,:] = Data.strike_certs[bell,:]/bellsums[bell]
        else:
            Data.strike_certs[bell,:] = 1.0
    #Try going through in turn for each set of rounds? Should reduce the order of this somewhat. BUT don't want to use bad things -- need a threshold for EACH bell
    for si in range(nstrikes):
        #Run through each row
        strikes = Data.strikes[:,si] #Strikes on this row.
        certs = Data.strike_certs[:,si]
        tmin = round(np.min(strikes) - Paras.cadence*(Paras.nbells - 2)); tmax = round(np.max(strikes) + Paras.cadence*(Paras.nbells - 2))
        #print('Examining row %d \r' % si)
        for fi, freq_test in enumerate(freq_tests):
            diff_slice = Data.transform_derivative[tmin:tmax,freq_test]
            diff_slice[diff_slice < 0.0] = 0.0
            diffsum = diff_slice**2
            
            diffsum = gaussian_filter1d(diffsum, Paras.freq_smoothing)
                
            peaks, _ = find_peaks(diffsum)
            
            prominences = peak_prominences(diffsum, peaks)[0]
            
            if len(prominences) == 0:
                continue

            sigs = prominences/np.max(prominences)   #Significance of the peaks relative to the background flow
            
            peaks = peaks + tmin
        
            #For each frequency, check consistency against confident strikes (threshold can be really high for that -- 99%?)
            for bell in range(Paras.nbells):
                best_value = 0.0; min_tvalue = 1e6
                pvalue = certs[bell]**Paras.beta
                for pi, peak_test in enumerate(peaks):
                    tvalue = 1.0/(abs(peak_test - strikes[bell])/tcut + 1)**Paras.strike_alpha
                    best_value = max(best_value, sigs[pi]**Paras.strike_gamma_init*tvalue*pvalue)
                    min_tvalue = min(min_tvalue, tvalue)
                allvalues[fi,si,bell] = best_value*min_tvalue**2
                
    allprobs[:,:] = np.mean(allvalues, axis = 1)
    del strikes; del certs; del diff_slice; del diffsum; del peaks; del prominences; del sigs; del allvalues

    #INSTEAD:: LOOK for frequenc
    
    #for bell in range(Paras.nbells):  #Normalise with the mean value perhaps
    #    allprobs[:,bell] = allprobs[:,bell]/np.mean(allprobs[:,bell])

    #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
    #Need to take into account lots of frequencies, not just the one (which is MUCH easier)
    #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
    
    for bell in range(Paras.nbells):
        allprobs[:,bell] = allprobs[:,bell]/np.max(allprobs[:,bell])
        
    npeaks = Paras.n_frequency_picks
    final_freqs = []   #Pick out the frequencies to test
    best_probs = []   #Pick out the probabilities for each bell for each of these frequencies
    #Filter allprobs nicely
    #Get rid of narrow peaks etc.
    for bell in range(Paras.nbells):
        
        probs_raw = allprobs[:,bell]
        probs_clean = np.zeros(len(probs_raw))   #Gets rid of the horrid random peaks
        for fi in range(1,len(probs_raw)-1):
            #probs_clean[fi] = np.min(probs_raw[fi-1:fi+2])
            probs_clean[fi] = probs_raw[fi]
            
        probs_clean = gaussian_filter1d(probs_clean, Paras.freq_filter) #Stops peaks wiggling around. Can cause oscillation in ability.
        
        #ax.plot(best_freqs/cut_length, probs_clean_smooth, label = bell, zorder = 5)
        
        peaks, _ = find_peaks(probs_clean)
        
        peaks = peaks[peaks > 50]#Data.nominals[bell]*1.1]  #Use nominal frequencies here?
        
        prominences = peak_prominences(probs_clean, peaks)[0]
        
        peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
           
        peaks = peaks[:npeaks]
        prominences = peak_prominences(probs_clean, peaks)[0]

        if len(peaks) == 0:
            st.error('Frequency reinforcement failed. Sorry...')
            if st.session_state.testing_mode:
                test_error('Frequency reinforcement failed. Sorry...')
            else:
                st.stop()
            
        peaks = peaks[prominences > 0.25*np.max(prominences)]
        
        final_freqs = final_freqs + freq_tests[peaks].tolist()
        
    del probs_raw; del probs_clean; del peaks; del prominences
           
    #Determine probabilities for each of the bells, in case of overlap
    #Only keep definite ones?
    
    #Remove repeated indices
    final_freqs = list(set(final_freqs))
    final_freqs = sorted(final_freqs)
    final_freqs = np.array(final_freqs)

    #Sort by height and filter out those which are too nearby -- don't count for the final number
    for freq in final_freqs:
        bellprobs = np.zeros(Paras.nbells)
        for bell in range(Paras.nbells):

            freq_range = 2  #Put quite big perhaps to stop bell confusion. Doesn't like it, unfortunately.

            #Put some blurring in here to account for the wider peaks
            top = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, bell])
            bottom = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, :])
                            
            bellprobs[bell] = (top/bottom)**2.0
        
        best_probs.append(bellprobs)
        
    best_probs = np.array(best_probs)

    nfinals = Paras.n_frequency_picks
    
    for bell in range(Paras.nbells):

        best_probs[:,bell] = best_probs[:,bell]/np.max(best_probs[:,bell])

        freqs_arrange = np.array([val for _, val in sorted(zip(best_probs[:,bell], final_freqs), reverse = True)]).astype('int')
        
        for fi in freqs_arrange[:nfinals*2]:
            #Check for close by alternatives to the biggest peaks and remove them
            others = final_freqs[(abs(final_freqs - fi) < 20) * (abs( final_freqs - fi) > 0)]
            if best_probs[np.where(final_freqs == fi)[0], bell] > 0.0:
                for other in others:
                    ind = np.where(final_freqs == other)[0]
    
                    best_probs[ind, bell] = 0.0
                
    #print('Confident frequency picks for bell', bell+1, ': ', np.sum(best_probs[:,bell] > 0.1), np.array(final_freq_ints)[best_probs[:,bell] > 0.1])

    for bell in range(Paras.nbells):
        #Filter out so there are only a few peaks on each one (will be quicker)
        threshold = sorted(best_probs[:,bell], reverse = True)[nfinals + 1]
        threshold = max(threshold, 5e-2)
        best_probs[:,bell] = best_probs[:,bell]*[best_probs[:,bell] > threshold]
    
    #Then finally run through and remove any picks that are generally useless
    frequencies = []; frequency_probabilities = []
    for fi, freq in enumerate(final_freqs):
        if np.max(best_probs[fi, :]) > 0.05:
            frequencies.append(freq)
            frequency_probabilities.append(best_probs[fi,:])
            
    frequencies = np.array(frequencies)
    frequency_probabilities = np.array(frequency_probabilities)
    
    del best_probs
    
    return frequencies, frequency_probabilities
      
def find_strike_probabilities(Paras, Data, init = False, final = False):
    #Find times of each bell striking, with some confidence
        
    #Make sure that this transform is sorted in EXACTLY the same way that it's done initially.
    #No tbefores etc., just the derivatives.
    #If reinforcing, don't need the whole domain
    
    if not final:
        st.current_log.write('Finding strike probabilities')
    nt_reinforce = Paras.nt
        
    allprobs = np.zeros((Paras.nbells, nt_reinforce))
             
    difflogs = []; all_diffpeaks = []; all_sigs = []
    
    #Produce logs of each FREQUENCY, so don't need to loop
    for fi, freq_test in enumerate(Data.test_frequencies):
        
        raw_slice = Data.transform[:nt_reinforce, freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        rawsum = np.sum(raw_slice**2, axis = 1)

        diff_slice = Data.transform_derivative[:nt_reinforce, freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2, axis = 1)

        diffsum = gaussian_filter1d(diffsum, 5)

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffsum_smooth = gaussian_filter1d(diffsum, round(Paras.smooth_time/Paras.dt))
        
        if init:
            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]  #This is just for plotting...
                
        else:
            sigs = prominences[prominences > diffsum_smooth[diffpeaks]]

            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]
            sigs = sigs/diffsum_smooth[diffpeaks]

        difflogs.append(diffsum)
        all_diffpeaks.append(diffpeaks)
        
        if not init:
            all_sigs.append(sigs)

    del diff_slice; del prominences; del diffsum_smooth
     
    if init:
        #The probabilities for each frequency correspnd exactly to those for each bell -- lovely
        difflogs = np.array(difflogs)

        for bell in range(Paras.nbells):  
            allprobs[bell] = difflogs[bell]/max(difflogs[bell])
    
                
        return allprobs

    else:
        #There are multiple frequency picks to choose from here, so it's more complicated
            
        for bell in range(Paras.nbells):  
            final_poss = []; final_sigs = []; final_probs = []; final_freqs = []
            for fi, freq_test in enumerate(Data.test_frequencies):
                if Data.frequency_profile[fi,bell] > 0.05:  #This is a valid frequency
                    sigs = all_sigs[fi]/np.max(all_sigs[fi])
                    
                    if np.max(sigs) > 0.1: #Maybe this is harsh but it should work...         
                    
                        peaks = all_diffpeaks[fi]
                        final_poss = final_poss + peaks.tolist()
                        final_sigs = final_sigs + sigs.tolist()
                        for k in range(len(sigs)):
                            final_probs = final_probs + [Data.frequency_profile[fi,bell]]
                            final_freqs.append(freq_test)
     
            final_poss = np.array(final_poss)
            final_sigs = np.array(final_sigs)
            final_probs = np.array(final_probs)/np.max(final_probs)
            final_freqs = np.array(final_freqs)

            tcut = round(Paras.prob_tcut/Paras.dt)
            
            overall_probs = np.zeros(len(diffsum))
                         
            #Need to split this up into time slices Ideally...
            t_ints = np.arange(len(diffsum))
            #Want number of significant peaks near the time really
                #Calculate probability at each time
            
            tvalues = 1.0/(np.abs(final_poss[:,np.newaxis] - t_ints[np.newaxis,:])/tcut + 1)**Paras.strike_alpha
            
            if Paras.frequency_skew < 0.5:
                fshift = np.zeros(len(final_freqs))
            else:
                fas = final_freqs/np.max(final_freqs)
                fshift = 1.0 - (1 - fas)**Paras.frequency_skew
            
            #fshift = np.ones(len(final_freqs))
                                    
            allvalues = tvalues*final_sigs[:,np.newaxis]**Paras.prob_beta*final_probs[:,np.newaxis]**Paras.strike_gamma*fshift[:,np.newaxis]

            allvalues = tvalues*final_sigs[:,np.newaxis]**Paras.prob_beta*final_probs[:,np.newaxis]**Paras.strike_gamma
                        
            absvalues = np.sum([tvalues > 0.5], axis = 1)
            
            absvalues = absvalues/np.max(absvalues)
            
            allvalues = allvalues*absvalues**Paras.near_freqs
            
            overall_probs =  np.sum(allvalues, axis = 0)
                
            overall_probs_smooth = gaussian_filter1d(overall_probs, round(Paras.smooth_time/Paras.dt), axis = 0)
                
            allprobs[bell] = overall_probs/(overall_probs_smooth + 1e-6)
            
            allprobs[bell] = allprobs[bell]/np.max(allprobs)
               
            del tvalues; del final_sigs; del absvalues; del allvalues; del overall_probs; del overall_probs_smooth
        return allprobs
