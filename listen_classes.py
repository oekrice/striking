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
import sys

import os
import re
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d

class audio_data():
    #Does the initial audio normalisation things
    def __init__(self, raw_file, doprints = True):
        
        upload_success = False
        
        raw_file.name, ext = os.path.splitext(raw_file.name)
        raw_file.name = re.sub(r'[^\w\-]', '_', raw_file.name)
        raw_file.name = raw_file.name + ext
        raw_file.size = sys.getsizeof(raw_file)

        #Only allow big files if they are wav
        if ext == '.wav':
            limit = 2e8
        else:
            limit = 2e7

        if raw_file.size > limit:
            st.error("Recording is too long... sorry. Hopefully  better server will remove this confounded limitation.")

        #Save to temporary file location so it can be converted if necessary
        with open('./tmp/%s' % raw_file.name[:], 'wb') as f: 
            f.write(raw_file.getvalue())        
        f.close()
    
        if ext != '.wav' and doprints:
            st.write('Uploaded file is not a .wav - attempting to convert it.')
        
        if ext != '.wav':
            new_fname = './tmp/' + raw_file.name[:-4] + '.wav'

            #Convert this to a wav
            os.system('ffmpeg -loglevel quiet -i ./tmp/%s ./tmp/%s.wav' % (raw_file.name, raw_file.name[:-4]))
            if os.path.exists(new_fname):
                upload_success = True

            else:
                os.system('rm -r ./tmp/' + raw_file.name)
                st.error("This doesn't seem to be an audio file.")
                st.stop()
                
        else:
            if doprints:
                st.write('File is in a nice format. Lovely.')
            new_fname = './tmp/' + raw_file.name

            upload_success = True

        if upload_success:
        
            st.session_state.audio_signal = None
            
            self.fs, self.data = wavfile.read(new_fname)  #Hopefully this doesn't use too much...
    
            if os.path.exists('./tmp/' + raw_file.name):
                os.system('rm -r ./tmp/' + raw_file.name)
            if os.path.exists('./tmp/' + raw_file.name[:-4] + '.wav'):
                os.system('rm -r ./tmp/' + raw_file.name[:-4] + '.wav')
            
            if len(self.data.shape) > 1:  #Is stereo
                import_wave = np.array(self.data)[:,0]
            else:  #Isn't
                import_wave = np.array(self.data)[:]
                
            import_wave = import_wave/(2**(16 - 1))
            st.session_state.audio_signal = import_wave
            st.session_state.fs = self.fs
            
            st.session_state.audio_filename = raw_file.name[:-4]
            st.session_state.uploader_key += 1

            del self.fs
            del self.data
            del import_wave
            del raw_file
            del f
            #Only care about the signal itself -- delete all the other file stuff
            
class parameters(object):
    #Contains information like number of bells, max times etc. 
    #Also all variables that can theoretically be easily changed
    
    #Want only one of these to exist at once -- so define as a singleton
    def __new__(cls, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, nreinforces):
        if not hasattr(cls, 'instance'):
            cls.instance = super(parameters, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, nreinforces):
                
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
        
        if True:#not st.session_state.trim_flag:
            if overall_tmax > 0.0:
                st.session_state.trimmed_signal = st.session_state.audio_signal[int(overall_tmin*st.session_state.fs):int(overall_tmax*st.session_state.fs)]
            else:
                st.session_state.trimmed_signal = st.session_state.audio_signal[int(overall_tmin*st.session_state.fs):]
            
        self.overall_tmin = overall_tmin
        self.overall_tmax = overall_tmax
        
        self.nbells = len(nominal_freqs)
        
        self.fcut_int = 2*int(self.fcut_length*st.session_state.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(st.session_state.trimmed_signal)/st.session_state.fs + self.overall_tmin
        
        self.prob_tcut = 0.1   #Time cutoff for all frequency identification
        self.prob_beta = 1.0  #How much to care about prominence looking at STRIKES
        self.near_freqs = 2  #How much to care about frequency peaks being nearby
        
        self.frequency_skew = 2.0   #How much to favour the high frequencies for timing reasons
        
        self.allstrikes = []
        
        if len(nominal_freqs) > 0:
            self.nominals = np.round(nominal_freqs*self.fcut_length).astype('int')
        else:
            self.nominals = []
            
        self.n_reinforces = nreinforces
        self.frequency_folder = './tmp/'
    
class data():
    #As for parameters
    def __new__(cls, Paras, tmin = -1, tmax = -1):
        if not hasattr(cls, 'instance'):
            cls.instance = super(data, cls).__new__(cls)
        return cls.instance

    def __init__(self, Paras, tmin = -1, tmax = -1):
        #This is called at the start -- can make some things like blank arrays for the nominals and the like. Can also do the FTs here etc (just once)
        
        #Chnage the length of the audio as appropriate
        
        if tmin > 0.0:
            cut_min_int = int(tmin*st.session_state.fs)
        else:
            cut_min_int = 0
        if tmax > 0.0:
            cut_max_int = int(tmax*st.session_state.fs)
        else:
            cut_max_int = -1
        
        st.session_state.local_signal = st.session_state.trimmed_signal[cut_min_int:cut_max_int]
            
        self.nominals = Paras.nominals

        self.initial_profile = np.identity(Paras.nbells)     #Initial frequencies for the bells -- these are just the nominals
     
        self.do_fourier_transform(Paras)
     
        self.find_transform_derivatives(Paras)
        
        #print('__________________________________________________________________________________________')
        #print('Calculating transform in range', cut_min_int/st.session_state.fs, 'to', cut_max_int/st.session_state.fs, 'seconds...')
        
        self.test_frequencies = self.nominals    #This is the case initially
        self.frequency_profile = np.identity(Paras.nbells)   #Each bell corresponds to its nominal frequency alone -- this will later be updated.
        

    def do_fourier_transform(self, Paras):
        
        full_transform = []; ts = []
        
        Paras.tmax = len(st.session_state.local_signal)/st.session_state.fs
        
        t = Paras.fcut_length/2   #Initial time (halfway through each transform)
        
        while t < Paras.tmax - Paras.fcut_length/2:
            cut_start  = int(t*st.session_state.fs - Paras.fcut_int/2)
            cut_end    = int(t*st.session_state.fs + Paras.fcut_int/2)
            
            signal_cut = st.session_state.local_signal[cut_start:cut_end]
            
            transform = abs(fft(signal_cut)[:len(signal_cut)//2])
            transform = 0.5*transform*st.session_state.fs/len(signal_cut)
                            
            ts.append(t)        
            full_transform.append(transform)
            
            t = t + Paras.dt
        
        self.ts = np.array(ts)
        self.transform = np.array(full_transform)  
        
        del transform
        del full_transform
        
        Paras.nt = len(ts)
        
        return 
    
    def find_transform_derivatives(self, Paras):
        allfreqs_smooth = gaussian_filter1d(self.transform, int(Paras.transform_smoothing/Paras.dt), axis = 0)
        diffs = np.zeros(allfreqs_smooth.shape)
        diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
        
        diffs[diffs < 0.0] = 0.0
        
        self.transform_derivative = diffs
        del diffs
        del allfreqs_smooth
        
        return 
    



