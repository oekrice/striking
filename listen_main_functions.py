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
import time
import pandas as pd
import matplotlib.pyplot as plt
from listen_classes import data

from listen_other_functions import find_ringing_times, find_strike_probabilities, do_frequency_analysis, find_strike_times, find_colour, check_initial_rounds, find_first_strikes, test_error, check_for_misses


def establish_initial_rhythm(Paras, final = False):

    if not final:
        Data = data(Paras, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    else:
        Data = data(Paras, tmin = 0.0, tmax = 60.0) #This class contains all the important stuff, with outputs and things
        
    # fig = plt.figure()
    # plt.pcolormesh(Data.transform[:500,:1000].T)
    # plt.yscale('log')
    # st.pyplot(fig)
    # plt.close()
    # st.stop()
    
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
    
    Paras.first_strikes, Paras.first_strike_certs = find_first_strikes(Paras, Data)
    
    Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
        
    return Data
        
def do_reinforcement(Paras, Data):
        
    #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
    
    #Check if there is suitable existing frequency data for this tower and at these parameters. 
    #If this calculation comes out better, save it out. If not, don't.
    
    Paras.new_frequencies = False
    Paras.overwrite_existing_freqs = False
    Paras.use_existing_freqs = False

    quality_log = []
    for reinforce_count in range(Paras.n_reinforces):
        
        #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
        #st.write('Doing frequency analysis,  iteration number', count + 1, 'of', Paras.n_reinforces)
        st.main_log.write('**Learning bell frequencies, iteration %d**' % (reinforce_count + 1))

        Data.test_frequencies, Data.frequency_profile = do_frequency_analysis(Paras, Data)  
            
        #Save out frequency data only when finished reinforcing? Yes.
        
        #print('Finding strike probabilities...')
        
        Data.strike_probabilities = find_strike_probabilities(Paras, Data, init = False, final = False)
        Data.end_flag = False

        strikes, strike_certs = find_strike_times(Paras, Data, final = False) #Finds strike times in integer space
    
        #Determine whether this is actually rounds or if something's got mixed up...
        all_is_well = check_initial_rounds(strikes)

        '''
        if all_is_well:
            print("All is well", np.shape(strikes))
        else:
            print("All is not well")
        '''
        if not all_is_well:
            st.error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            if st.session_state.testing_mode:
                test_error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            else:
                st.stop()

        if len(strikes) == 0:
            st.error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            if st.session_state.testing_mode:
                test_error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            else:
                st.stop()

        if np.shape(strikes)[1] < 6:
            st.error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            if st.session_state.testing_mode:
                test_error("Failed to detect rounds. Either there aren't any or the recording isn't good enough...")
            else:
                st.stop()

        #Check and fix handstrokes if necessary
        diff1s = strikes[:,1::2] - strikes[:,0:-1:2]
        diff2s = strikes[:,2::2] - strikes[:,1:-1:2]
    
        #Estalish if the strokes might be the wrong way around... Look at diffs as in initial rounds.
        #st.write('Initial diffs', diff1s, diff2s)
        kback = min(len(diff2s[0]), 8)
        if np.mean(diff1s[:kback,:]) < np.mean(diff2s[:kback,:]):
            handstroke_first = True
        else:
            handstroke_first = False
    
        #print('a', st.session_state.handstroke_first, handstroke_first, Data.handstroke_first)
    
        st.session_state.handstroke_first = handstroke_first
        Data.handstroke_first = handstroke_first
        #print('b', st.session_state.handstroke_first, handstroke_first, Data.handstroke_first)

        
        #Filter these strikes for the best rows, to then be used for reinforcement
        best_strikes = []; best_certs = []; allcerts = []; row_ids = []
        #Pick the ones that suit each bell in turn --but make sure to weight!
        done  = False
        while not done:
            for bell in range(Paras.nbells):
                threshold = 0.01   #Need changes to be at least this good... Need to improve on this really as it's a bit arbitrary.
                allcerts = []; count = 0
                for row in range(len(strikes[0])):
                    allcerts.append(strike_certs[bell,row])
                if len(allcerts) > Paras.nreinforce_rows:
                    threshold = max(threshold, sorted(allcerts, reverse = True)[Paras.nreinforce_rows]) 
                for row in range(len(strikes[0])):
                    if strike_certs[bell,row] > threshold and count < Paras.nreinforce_rows:
                        #st.write(row, row_ids)
                        if row not in row_ids:
                            row_ids.append(row)
                            best_strikes.append(strikes[:,row])
                            best_certs.append(strike_certs[:,row])
                            count += 1
            done = True
            if len(best_strikes) > 3:
                done = True
            else:
                threshold = threshold/2
        
        Data.strikes, Data.strike_certs = strikes, strike_certs
        st.current_log.write('Using ' + str(len(Data.strikes[0])) + ' rows for next reinforcement')

        reinforce_count += 1

        quality_log.append(Data.freq_data[2])

        #This stuff is now a bit different... Need the three arrays as st session variables
        if len(Data.strikes) > 0 and len(Data.strike_certs) > 0:
            #Check if it's worth overwriting the old one? Do this at EVERY STEP, and save out to THIS filename.
            update = False
            if st.session_state.reinforce_frequency_data is not None:
                if st.session_state.reinforce_frequency_data[2] < Data.freq_data[2]:
                    update = True
            else:
                update= True
                
            if update:
                #Experimental -- run through and detect the best yet from EACH bell!
                st.session_state.reinforce_test_frequencies = Data.test_frequencies
                st.session_state.reinforce_frequency_profile = Data.frequency_profile
                st.session_state.reinforce_frequency_data = Data.freq_data
                st.session_state.already_saved = False
                st.session_state.analysis_status = 0
                    
        else:
            #st.session_state.reinforce = 0
            st.error("Frequency analysis failed for some reason. If the percentage is reasonably high this is probably due to bad audio or bad ringing. If it is low then check the correct tower/bells are selected.")
            if st.session_state.testing_mode:
                test_error("Frequency analysis failed for some reason. If the percentage is reasonably high this is probably due to bad audio or bad ringing. If it is low then check the correct tower/bells are selected.")
            else:
                st.stop()
            
        if st.session_state.reinforce_frequency_data is not None:
            #Determine colours:
            toprint = st.session_state.reinforce_frequency_data[2]
            c = find_colour(toprint)
            st.quality_log.write('Best yet frequency match: :%s[%.1f%%]' % (c, 100*toprint))

        if len(quality_log) > 3:
            #Check for lack of increase
            if np.max(quality_log[-3:]) < np.max(quality_log[:-3]):
                break

    return Data

def find_final_strikes(Paras, nested = False):
    
     tmin = Paras.ringing_start*Paras.dt
     tmax = tmin + Paras.overall_tcut + Paras.ringing_start*Paras.dt
     allstrikes = []; allcerts = []
     allcadences = []
     Paras.stop_flag = False
     Paras.local_tmin = Paras.overall_tmin
     Paras.local_tint = round(Paras.overall_tmin/Paras.dt)
     Paras.overall_end = (len(st.session_state.trimmed_signal)/st.session_state.fs)/Paras.dt
     Paras.ringing_finished = False
     length_log = []
     #st.analysis_sublog.write('Initial rhythm established, finding all strikes')
     st.analysis_sublog.progress(0, text = 'Initial rhythm established, finding all strikes')

     counter = 0; last_switch = 0
     while not Paras.stop_flag and not Paras.ringing_finished:
        Data = data(Paras, tmin = tmin, tmax = tmax) #This class contains all the important stuff, with outputs and things
        if tmax >= (Paras.overall_end - 1.0)*Paras.dt:  #Last one
            Paras.stop_flag = True
            Data.end_flag = True
        else:
            Data.end_flag = False

        Paras.local_tmin = tmin + Paras.overall_tmin
        Paras.local_tint = round((tmin+Paras.overall_tmin)/Paras.dt) 

        Data.test_frequencies = st.session_state.final_freqs
        Data.frequency_profile = st.session_state.final_freqprobs
             
        if counter == 0:
            #Adjust the first strikes as appropriate
            Paras.first_strikes = Paras.first_strikes + Paras.ringing_start - round(tmin/Paras.dt)

        Data.strike_probabilities = find_strike_probabilities(Paras, Data, init = False, final = True)
                           
        if len(allstrikes) == 0:  #Look for changes after this time
            Data.handstroke_first = st.session_state.handstroke_first
        else:
            if len(allstrikes)%2 == 0:
                Data.handstroke_first = st.session_state.handstroke_first
            else:
                Data.handstroke_first = not(st.session_state.handstroke_first)
            Data.last_change = np.array(allstrikes[-1]) - round(tmin/Paras.dt)
            Data.cadence_ref = Paras.cadence_ref

        Data.strikes, Data.strike_certs = find_strike_times(Paras, Data, final = True) #Finds strike times in integer space

        if len(Data.strikes) == 0:
            Paras.stop_flag = True
            Paras.ringing_finished = True
        elif len(np.shape(Data.strikes)) == 0:
            Paras.stop_flag = True
            Paras.ringing_finished = True
        elif len(Data.strikes[0]) < 1:
            Paras.stop_flag = True
            Paras.ringing_finished = True
        elif np.median(Data.strike_certs[:,-1]) < 0.25: #Can see if this can be fixed... who knows?
            Paras.stop_flag = True
            Paras.ringing_finished = True
        
        if st.session_state.allstrikes is None:
            all_is_well = check_initial_rounds(Data.strikes)
            if not all_is_well:
                st.error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                if st.session_state.testing_mode:
                    test_error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                else:
                    st.session_state.analysis_status = 0
                    st.stop()

        if len(np.shape(Data.strikes)) > 1:

            if len(Data.strikes[:,0]) == 0:
                st.error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                if st.session_state.testing_mode:
                    test_error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                else:
                    st.session_state.analysis_status = 0
                    st.stop()

            for row in range(0,len(Data.strikes[0])):
                allstrikes.append((Data.strikes[:,row] + round(tmin/Paras.dt)).tolist())
                allcerts.append(Data.strike_certs[:,row].tolist())
                allcadences.append((np.max(allstrikes[-1]) - np.min(allstrikes[-1]))/(Paras.nbells-1))
  
        if counter == 0 and not nested:
             #print('First transform test:')
             diff1s = Data.strikes[:,1::2] - Data.strikes[:,0:-1:2]
             diff2s = Data.strikes[:,2::2] - Data.strikes[:,1:-1:2]
             
             if np.mean(diff1s[:]) < np.mean(diff2s[:]):
                 handstroke_first = True
             else:
                 handstroke_first = False
             #print(handstroke_first, st.session_state.handstroke_first)
             if handstroke_first != st.session_state.handstroke_first:
                 #print('Wrong stroke! Running nested to fix')
                 st.session_state.handstroke_first = handstroke_first
                 find_final_strikes(Paras, nested = True)
                 return

        #This is probably the best place to be retroactive. allstrikes has the strikes in ABSOLUTE form. Not sure where the 'first' one comes from for the next round
        #At this point it's probably wise to add the retroactive checker. Just go back and change the offending row, then continue from there? Can go arbitrarily far back if necessary.
        #The raw Data.strikes are NOT adjusted by the start time
        #Everything else should figure out automatically based on allstrikes, and the session state will do so as well

        if len(length_log) > 0:
            if len(allstrikes) != length_log[-1]:
                allstrikes, allcerts, switch_id = check_for_misses(allstrikes, allcerts, last_switch)   #This is the magic function!
        else:
            allstrikes, allcerts, switch_id = check_for_misses(allstrikes, allcerts, last_switch)   #This is the magic function!

        if switch_id > 0:
            last_switch = switch_id

        length_log.append(len(allstrikes))

        tmin = min(allstrikes[-1])*Paras.dt - 5.0
        tmax = min(tmin + Paras.overall_tcut, Paras.overall_tmax)
             
        #Update global class things

        nrows_count = int(min(len(allcadences), 20))
        Paras.cadence_ref = np.mean(allcadences[-nrows_count:])
        Paras.allstrikes = np.array(allstrikes)
         
        progress_fraction = (np.max(allstrikes[-1])*Paras.dt - Paras.ringing_start*Paras.dt)/(len(st.session_state.trimmed_signal)/st.session_state.fs)
        progress_fraction = min(1, progress_fraction)
        st.analysis_sublog.progress(progress_fraction, text = 'Complete until time %d seconds, after %d rows' % (np.max(allstrikes[-1])*Paras.dt, len(Paras.allstrikes)))
        #st.analysis_sublog.write('Complete until time %d seconds with %d rows' % (np.max(allstrikes[-1])*Paras.dt, len(Paras.allstrikes)))
         
        st.session_state.allstrikes = np.array(allstrikes).T
        st.session_state.allcerts = np.array(allcerts).T
     
        counter += 1

     del allstrikes; del allcerts
     Data = None
     
     return 
     
def filter_final_strikes(Paras):
    #Looks for non-confident blows and attempts to put them somewhere reasonable based on the (hopefully) confident blows either side. Should make the grids look better...
    #Needs to find the ratio through the change for each bell, approximately
    change_ratios = np.zeros(np.shape(st.session_state.allstrikes))
    for ri in range(len(change_ratios[0])):
        change_ratios[:,ri] = (st.session_state.allstrikes[:,ri] - np.min(st.session_state.allstrikes[:,ri]))/(np.max(st.session_state.allstrikes[:,ri]) - np.min(st.session_state.allstrikes[:,ri]))
    
    for bell in range(len(change_ratios)):
        for ri in range(1,len(change_ratios[0])-1):
            if st.session_state.allcerts[bell,ri] < 0.1:   #This is one to be interpolated. Hopefully those either side are fine...
                predicted_ratio = 0.5*(change_ratios[bell,ri-1] + change_ratios[bell,ri + 1])
                predicted_location = np.min(st.session_state.allstrikes[:,ri]) + predicted_ratio*(np.max(st.session_state.allstrikes[:,ri]) - np.min(st.session_state.allstrikes[:,ri]))
                st.session_state.allstrikes[bell,ri] = predicted_location
    #Check that last row is indeed real. For some reason it detects fake rows with much confidence sometimes...
    if np.median(st.session_state.allcerts[:,-1]) < 0.5:
        st.session_state.allstrikes = st.session_state.allstrikes[:,:-1]
        st.session_state.allcerts = st.session_state.allcerts[:,:-1]
    #Check order
    yvalues = np.arange(len(st.session_state.allstrikes[:,0])) + 1
    penultimate_order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,-2], yvalues), reverse = False)])
    ultimate_order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,-1], yvalues), reverse = False)])
    if (penultimate_order == yvalues).all() and not (ultimate_order == yvalues).all():
        st.session_state.allstrikes = st.session_state.allstrikes[:,:-1]
        st.session_state.allcerts = st.session_state.allcerts[:,:-1]
    #Check timing
    diffs = np.sort(st.session_state.allstrikes[:,-1])[1:] - np.sort(st.session_state.allstrikes[:,-1])[:-1]
    if np.min(diffs) < 1:
        st.session_state.allstrikes = st.session_state.allstrikes[:,:-1]
        st.session_state.allcerts = st.session_state.allcerts[:,:-1]

    return

def save_strikes(Paras):
    #Saves as a pandas thingummy like the strikeometer does
    allstrikes = []
    allbells = []
    allcerts_save = []
    yvalues = np.arange(len(st.session_state.allstrikes[:,0])) + 1
    orders = []
    
    if  st.session_state.handstroke_first:
        for row in range(len(st.session_state.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], yvalues), reverse = False)])
            certs = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], st.session_state.allcerts[:,row]), reverse = False)])
            allstrikes = allstrikes + sorted((st.session_state.allstrikes[:,row]).tolist())
            allcerts_save = allcerts_save + certs.tolist()
            allbells = allbells + order.tolist()
            orders.append(order)

    else:
        for row in range(1, len(st.session_state.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], yvalues), reverse = False)])
            certs = np.array([val for _, val in sorted(zip(st.session_state.allstrikes[:,row], st.session_state.allcerts[:,row]), reverse = False)])
            allstrikes = allstrikes + sorted((st.session_state.allstrikes[:,row]).tolist())
            allcerts_save = allcerts_save + certs.tolist()
            allbells = allbells + order.tolist()
            orders.append(order)
            
    allstrikes = 1000*np.array(allstrikes)*0.01
    
    #st.write('saved', allstrikes[:12])
    allbells = np.array(allbells)
    allcerts_save = np.array(allcerts_save)
    orders = np.array(orders)
    
    striking_df = pd.DataFrame({'Bell No': allbells, 'Actual Time': allstrikes, 'Confidence': allcerts_save})
    return striking_df, orders



