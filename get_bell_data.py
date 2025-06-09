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

allrows = []; tower_names = []; tower_ids = []; tower_regions = []
row_data = np.zeros(len(all_fullcircles))
tower_id = -1
fine = True
for bellcount in range(len(df) - 1):

    tower_id = df["Tower ID"][bellcount]
    tower_name = df["Place"][bellcount] + ', ' + df["Dedication"][bellcount]
    tower_region = df["Region"][bellcount]
    #Put in bell data
    belltype = df["Bell Role"][bellcount]
    if belltype in all_fullcircles:
        ind = all_fullcircles.index(belltype)
        if df["Nominal (Hz)"][bellcount] > 0:
            row_data[ind] = df["Nominal (Hz)"][bellcount]
        else:
            fine = False
    
    if df["Tower ID"][bellcount + 1] != tower_id:  #New tower
        if fine:
            allrows.append(row_data)   #Append last one
            tower_ids.append(tower_id)
            tower_names.append(tower_name)
            tower_regions.append(tower_region)
        row_data = np.zeros(len(all_fullcircles))
        #Get new ones
        fine = True

    #Bells are in order of tower so can do it this way. Do as a big array
if fine:
    allrows.append(row_data)   #Append last one
    tower_ids.append(tower_id)
    tower_names.append(tower_name)
    tower_regions.append(tower_region)

def make_unique_names(tower_names, tower_regions):
    #For use when there are two towers with the same name, put the region on as well
    fixed_tower_names = tower_names.copy()
    count = 0
    for ti, tower in enumerate(tower_names):
        if sum(name == tower for name in tower_names) > 1:
            fixed_tower_names[ti] = tower_names[ti] + ' (' + tower_regions[ti] + ')'
            count += 1
    return fixed_tower_names

tower_names = make_unique_names(tower_names, tower_regions)

def tower_alias(df_new, tower_id, alias_id, tower_name):
    #Used on a bespoke basis when the Dove data is incomplete. Will assign the data from 'alias name' to 'tower name'
    alias_data = df_new[df_new["Tower ID"] == alias_id].iloc[0].copy()
    alias_data["Tower ID"] = tower_id
    alias_data["Tower Name"] = tower_name
    df_new = pd.concat([df_new, pd.DataFrame([alias_data])], ignore_index = True)
    return df_new

allrows = np.array(allrows)

newdata = {'Tower ID': tower_ids, 'Tower Name': tower_names}
df_new = pd.DataFrame(newdata)

for ri, bell_type in enumerate(all_fullcircles):
    df_new[bell_type] = allrows[:,ri]
    
#Filter out towers with not enough bells
df_new = df_new[df_new.apply(lambda tower: (tower != 0).sum(), axis=1) >= 6]
#Add tower aliases
df_new = tower_alias(df_new, 15311, 15140, "Woolpit, Blessed Virgin Mary")
df_new = tower_alias(df_new, 15838, 14094, "Sunderland, Bishopwearmouth, Minster Ch of S Michael & All Angels & S Benedict Biscop")

print('Saving to .csv...')
df_new.to_csv('./bell_data/nominal_data.csv')  


