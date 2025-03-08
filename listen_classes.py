# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 21:53:04 2025

@author: eleph
"""
import streamlit as st

import os
from scipy.io import wavfile
import numpy as np
import wave

class audio_data():
    #Does the initial audio normalisation things
    def __init__(self, raw_file):
        
        upload_success = False
        #Save to temporary file location so it can be converted if necessary
        with open('./tmp/%s' % raw_file.name[:], 'wb') as f: 
            f.write(raw_file.getvalue())        
            #upload_data = raw_file.getvalue()
        
        if raw_file.name[-4:] != '.wav':
            st.write('Uploaded file is not a .wav - attempting to convert it.')
        
        if raw_file.name[-4:] != '.wav':
            new_fname = './tmp/' + raw_file.name[:-4] + '.wav'

            #Convert this to a wav
            os.system('ffmpeg -loglevel quiet -i ./tmp/%s ./tmp/%s.wav' % (raw_file.name, raw_file.name[:-4]))
            if os.path.exists(new_fname):
                st.write('Audio file %s uploaded and converted sucessfully.' % raw_file.name)
                upload_success = True

            else:
                os.system('rm -r ./tmp/' + raw_file.name)
                st.error("This doesn't seem to be an audio file.")
                
        else:
            st.write('File is in a nice format. Lovely.')
            new_fname = './tmp/' + raw_file.name

            st.write('Audio file %s uploaded sucessfully.' % raw_file.name)
            upload_success = True

        if upload_success:
        
            self.fs, self.data = wavfile.read(new_fname)
    
            os.system('rm -r ./tmp/' + raw_file.name)
            os.system('rm -r ./tmp/' + raw_file.name[:-4] + '.wav')
            
            if len(self.data.shape) > 1:  #Is stereo
                import_wave = np.array(self.data)[:,0]
            else:  #Isn't
                import_wave = np.array(self.data)[:]
                
            self.signal = import_wave/(2**(16 - 1))
    
class parameters():
    #Contains information like number of bells, max times etc. 
    #Also all variables that can theoretically be easily changed
    def __init__(self, Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, nreinforces):
                
        self.dt = 0.01
        self.fcut_length = 0.125  #Length of each transform slice (in seconds)
        
        self.transform_smoothing = 0.05 #Transform smoothing for the initial derivatives of the transform (in seconds)
        self.frequency_range = 3    #Range over which to include frequencies in a sweep (as in, 300 will count between 300-range:frequency+range+1 etc.)
        self.derivative_smoothing = 5  #Smoothing for the derivative (in INTEGER time lumps -- could change if necessary...)
        self.smooth_time = 2.0    #Smoothing over which to apply change-long changes (in seconds)
        self.max_change_time = 3.5 #How long could a single change reasonably be
        self.nrounds_min = 8 #How many rounds do you need (8 = 4 whole pulls, seems reasonable...)
        self.nrounds_max = 30 #How many rounds maximum
        self.nreinforce_rows = 4
        
        self.strike_smoothing = 1 #How much to smooth the input probability function
        self.strike_tcut = 1.0 #How many times the average cadence to cut off
        self.strike_alpha = 2  #How much to care about timing
        self.strike_gamma = 1  #How much to care about prominence
        self.strike_gamma_init = 1.5  #How much to care about prominence for the initial rounds
        
        self.freq_tcut = 0.2 #How many times the average cadence to cut off for FREQUENCIES (should be identical strikes really)
        self.freq_smoothing = 2 #How much to smooth the data when looking for frequencies (as an INTEGER)
        self.beta = 1   #How much to care whether strikes are certain when looking at frequencies
        self.freq_filter = 2#How much to filter the frequency profiles (in INT)
        self.n_frequency_picks = 10  #Number of frequencies to look for (per bell)
        
        self.rounds_probs_smooth = 2  
        self.rounds_tcut = 0.5 #How many times the average cadence to cut off find in rounds
        self.rounds_leeway = 1.5 #How far to allow a strike before it is more improbable

        self.rounds_tmax = rounds_tmax
        self.reinforce_tmax = reinforce_tmax
        
        self.overall_tcut = 60.0  #How frequently (seconds) to do update rounds etc.
        self.probs_adjust_factor = 2.0   #Power of the bells-hitting-each-other factor. Less on higher numbers seems favourable.
        
        if overall_tmax > 0.0:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):int(overall_tmax*Audio.fs)]
        else:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):]
            
        self.overall_tmin = overall_tmin
        self.overall_tmax = overall_tmax
        
        self.nbells = len(nominal_freqs)
        
        self.fcut_int = 2*int(self.fcut_length*Audio.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(Audio.signal)/Audio.fs
        
        self.prob_tcut = 0.1   #Time cutoff for all frequency identification
        self.prob_beta = 1.0  #How much to care about prominence looking at STRIKES
        self.near_freqs = 2  #How much to care about frequency peaks being nearby
        
        self.frequency_skew = 2.0   #How much to favour the high frequencies for timing reasons
        
        self.allstrikes = []
        
        if len(nominal_freqs) > 0:
            self.nominals = np.round(nominal_freqs*self.fcut_length).astype('int')
        else:
            self.nominals = []
            
        self.nreinforces = nreinforces
        self.frequency_folder = './tmp/'
        
        st.write('Parameters sorted', self.overall_tmin)
            




