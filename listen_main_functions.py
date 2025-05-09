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
from listen_classes import data

from listen_other_functions import find_ringing_times, find_strike_probabilities, find_first_strikes, do_frequency_analysis, find_strike_times_rounds, find_colour


def establish_initial_rhythm(Paras, final = False):
    #Obtain various things about the ringing. What exactlythis does will depend on what's required from the situation
    #Hopefully remove a load of the bugs that seem to have appeared.

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
            
    for count in range(Paras.n_reinforces):
        
        #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
        #st.write('Doing frequency analysis,  iteration number', count + 1, 'of', Paras.n_reinforces)
        st.main_log.write('**Analysing bell frequencies, iteration %d of %d**' % (count + 1, Paras.n_reinforces))

        Data.test_frequencies, Data.frequency_profile = do_frequency_analysis(Paras, Data)  
            
        #Save out frequency data only when finished reinforcing?
        
        #print('Finding strike probabilities...')
        
        Data.strike_probabilities = find_strike_probabilities(Paras, Data, init = False, final = False)
                
        strikes, strike_certs = find_strike_times_rounds(Paras, Data, final = False, doplots = 1) #Finds strike times in integer space
    
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
                threshold = 0.05   #Need changes to be at least this good... Need to improve on this really.
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
            
        st.current_log.write('Using ' + str(len(best_strikes)) + ' rows for next reinforcement')
        Data.strikes, Data.strike_certs = np.array(best_strikes).T, np.array(best_certs).T
        
        count += 1
        
        #This stuff is now a bit different... Need the three arrays as st session variables
        if len(Data.strikes) > 0 and len(Data.strike_certs) > 0:
            #Check if it's worth overwriting the old one? Do this at EVERY STEP, and save out to THIS filename.
            if True:   #Old one which works -KEEP
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
                    
            else:
                if st.session_state.reinforce_frequency_data is not None:
                    #st.write(st.session_state.reinforce_frequency_data[:])
                    #st.write(Data.freq_data[:])
                    for bell in range(Paras.nbells):
                        #Check individually against the reference
                        if st.session_state.reinforce_frequency_data[4+bell] < Data.freq_data[4+bell]:
                            st.session_state.reinforce_frequency_data[4+bell] = Data.freq_data[4+bell]
                            st.session_state.reinforce_test_frequencies[:] = Data.test_frequencies[:]
                            st.session_state.reinforce_frequency_profile[:,bell] = Data.frequency_profile[:,bell]
                            st.session_state.already_saved = False
                else:
                    st.session_state.reinforce_test_frequencies = Data.test_frequencies
                    st.session_state.reinforce_frequency_profile = Data.frequency_profile
                    st.session_state.reinforce_frequency_data = Data.freq_data
                    st.session_state.already_saved = False
                
        else:
            #st.session_state.reinforce = 0
            st.error("Frequency analysis failed for some reason. If the percentage is reasonably high this is probably due to bad audio or bad ringing. If it is low then check the correct tower/bells are selected.")
            st.stop()
            
        if st.session_state.reinforce_frequency_data is not None:
            #Determine colours:
            toprint = st.session_state.reinforce_frequency_data[2]
            c = find_colour(toprint)
            st.quality_log.write('Best yet frequency match: :%s[%.1f %%]' % (c, 100*toprint))

    return Data

def find_final_strikes(Paras, nested = False):
    
     #Create new data files in turn -- will be more effeicient ways but meh...
     tmin = 0.0
     tmax = tmin + Paras.overall_tcut + Paras.ringing_start*Paras.dt
     allstrikes = []; allcerts = []
     Paras.allcadences = []
     Paras.stop_flag = False
     Paras.local_tmin = Paras.overall_tmin
     Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
     Paras.ringing_finished = False
     
     #st.analysis_sublog.write('Initial rhythm established, finding all strikes')
     st.analysis_sublog.progress(0, text = 'Initial rhythm established, finding all strikes')

     counter = 0
     while not Paras.stop_flag and not Paras.ringing_finished:
         
         if tmax >= Paras.overall_tmax - 1.0:  #Last one
             Paras.stop_flag = True
             
         Paras.local_tmin = tmin + Paras.overall_tmin
         Paras.local_tint = int((tmin+Paras.overall_tmin)/Paras.dt) 

         Data = data(Paras, tmin = tmin, tmax = tmax) #This class contains all the important stuff, with outputs and things
         
         Data.test_frequencies = st.session_state.final_freqs
         Data.frequency_profile = st.session_state.final_freqprobs
             
         Data.strike_probabilities = find_strike_probabilities(Paras, Data, init = False, final = True)
                           
         if len(allstrikes) == 0:  #Look for changes after this time
             Data.handstroke_first = st.session_state.handstroke_first
         else:
             if len(allstrikes)%2 == 0:
                 Data.handstroke_first = st.session_state.handstroke_first
             else:
                 Data.handstroke_first = not(st.session_state.handstroke_first)
             Data.last_change = np.array(allstrikes[-1]) - int(tmin/Paras.dt)
             Data.cadence_ref = Paras.cadence_ref

         Data.strikes, Data.strike_certs = find_strike_times_rounds(Paras, Data, final = True, doplots = 2) #Finds strike times in integer space
                   
         if len(np.shape(Data.strikes)) > 1:
             pass
         else:
             Paras.stop_flag = True
            
         if len(np.shape(Data.strikes)) > 1:
            if len(Data.strikes[:,0]) > 1:
                if len(allstrikes) == 0:   #Check for rounds at the start
                    if np.where(Data.strikes[:,0] == np.max(Data.strikes[:,0]))[0] != Paras.nbells - 1:
                        st.error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                        st.session_state.analysis_status = 0
                        time.sleep(5.0)
                        st.rerun()
            else:
                    st.error('This recording doesn\'t appear to start in rounds. If frequencies are confident check this is the right tower. If it is, then bugger.')
                    st.session_state.analysis_status = 0
                    time.sleep(5.0)
                    st.rerun()

            for row in range(0,len(Data.strikes[0])):
                 allstrikes.append((Data.strikes[:,row] + int(tmin/Paras.dt)).tolist())
                 allcerts.append(Data.strike_certs[:,row].tolist())
                 Paras.allcadences.append((np.max(allstrikes[-1]) - np.min(allstrikes[-1]))/(Paras.nbells-1))
                 
         if counter == 0 and not nested:
             #print('First transform test:')
             diff1s = Data.strikes[:,1::2] - Data.strikes[:,0:-1:2]
             diff2s = Data.strikes[:,2::2] - Data.strikes[:,1:-1:2]
             
             kback = len(diff2s[0])
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
             
             
         tmin = min(allstrikes[-1])*Paras.dt - 5.0
         tmax = min(tmin + Paras.overall_tcut, Paras.overall_tmax)
             
         #Update global class things
         nrows_count = int(min(len(Paras.allcadences), 20))
         Paras.cadence_ref = np.mean(Paras.allcadences[-nrows_count:])
         Paras.allstrikes = np.array(allstrikes)
         
         progress_fraction = (np.max(allstrikes[-1])*Paras.dt - Paras.ringing_start*Paras.dt)/(len(st.session_state.trimmed_signal)/st.session_state.fs - Paras.ringing_start*Paras.dt)
         st.analysis_sublog.progress(progress_fraction, text = 'Complete until time %d seconds, after %d rows' % (np.max(allstrikes[-1])*Paras.dt, len(Paras.allstrikes)))
         #st.analysis_sublog.write('Complete until time %d seconds with %d rows' % (np.max(allstrikes[-1])*Paras.dt, len(Paras.allstrikes)))
         
         st.session_state.allstrikes = np.array(allstrikes).T
         st.session_state.allcerts = np.array(allcerts).T
     
         counter += 1
         
     del allstrikes; del allcerts
     Data = None
     
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



