# -*- coding: utf-8 -*-
"""
Script for taking the dove bells csv and outputting in a sensible way.
As a pandas dataframe. First column tower name, second dove id and the rest (up to 16) as the rest of the columns
"""

'''
Needs to download a group of files...
'''

import pandas as pd
import numpy as np

if False:
    print('Downloading data...')
    data = pd.read_csv('https://dove.cccbr.org.uk/bells.csv?bells=all&ring_type=english')

    data.to_csv('./bell_data/raw_data.csv')  

df = pd.read_csv('./bell_data/raw_data.csv')

#Run through everything just in turn -- only need one loop
#Make a dataframe with name, ids and bell nominals. 
all_bell_types = df["Bell Role"].unique()

#Filter out chimes
all_fullcircles = []
for bell_type in all_bell_types:
    if 'c' not in bell_type:
        all_fullcircles.append(bell_type)

allrows = []; tower_names = []; tower_ids = []
row_data = np.zeros(len(all_fullcircles))
tower_id = -1
fine = True
for bellcount in range(len(df) - 1):

    tower_id = df["Tower ID"][bellcount]
    tower_name = df["Place"][bellcount] + ', ' + df["Dedication"][bellcount]

    #Put in bell data
    belltype = df["Bell Role"][bellcount]
    if belltype in all_fullcircles:
        ind = all_fullcircles.index(belltype)
        if df["Nominal (Hz)"][bellcount] > 0:
            row_data[ind] = df["Nominal (Hz)"][bellcount]
        else:
            fine = False
    
    tower_id = df["Tower ID"][bellcount]

    if df["Tower ID"][bellcount + 1] != tower_id:  #New tower
        if fine:
            allrows.append(row_data)   #Append last one
            tower_ids.append(tower_id)
            tower_names.append(tower_name)
        row_data = np.zeros(len(all_fullcircles))
        #Get new ones
        fine = True

    #Bells are in order of tower so can do it this way. Do as a big array
if fine:
    allrows.append(row_data)   #Append last one
    tower_ids.append(tower_id)
    tower_names.append(tower_name)
    
allrows = np.array(allrows)

newdata = {'Tower ID': tower_ids, 'Tower Name': tower_names}
df_new = pd.DataFrame(newdata)

for ri, bell_type in enumerate(all_fullcircles):
    df_new[bell_type] = allrows[:,ri]
    
df_new.to_csv('./bell_data/nominal_data.csv')  





