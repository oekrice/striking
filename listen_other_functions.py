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

#Added text so Streamlit detects a commit
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


def find_strike_times(Paras, Data, final = False, doplots = 0):
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
                go = False
                   
                if len(allconfs) > 1:
                    bellconfs_individual = np.mean(np.array(allconfs)[1:,:], axis = 0)

                Data.freq_data = np.array([Paras.dt, Paras.fcut_length, 0., 0.])
                Data.freq_data = np.concatenate((Data.freq_data, np.zeros(Paras.nbells)))

        if len(strikes) > 0:
            if np.median(confs) > 0.5 or len(allstrikes) == 0:
                allstrikes.append(strikes)
                allconfs.append(confs)
                
        if len(allstrikes) == 0:
            Paras.ringing_finished = True
            return [], []
         
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

        if next_end + 0.5*round(Data.cadence_ref) > len(Data.strike_probabilities[0]):   #This is nearing the end of the reasonable time
            go = False

    if len(allconfs) > 1:
        
        bellconfs_individual = np.mean(np.array(allconfs)[1:,:], axis = 0)
        Data.freq_data = np.array([Paras.dt, Paras.fcut_length, np.mean(allconfs[1:]), np.min(allconfs[1:])])
        Data.freq_data = np.concatenate((Data.freq_data, bellconfs_individual))
        
    if len(allstrikes) == 0:
        Paras.ringing_finished = True
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
            st.error('Frequency reinforcement failed. Sorry... Try changing the audio parameters a bit?')
            time.sleep(10.0)
            st.rerun()
            
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
    
    plot_bell = 5
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
        difflogs = np.array(difflogs)
            
        if final:
            doplot = False
        else:
            doplot = True
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
