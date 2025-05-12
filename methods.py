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

#This script is for analysing methods. 
#Takes the raw timing csv as an input (or can pass less if necessary) and will do the analysis from that. Hopefully don't need anything else
#Get streamlit to detect commit

import pandas as pd
import numpy as np
import re 
import streamlit as st

#This bit for importing the raw data. 

#STEPS:
#1. Identify the largest portion of not-rounds
#2. Identify the stage
#3. Identify the treble pattern. Options are Plain or Treble Bob up to the stage number. Might have to do that manually.
    #Save as a list somehow -- first and last change of each lead (to allow for spliced?)
#4. If no treble pattern, check for Stedman/Erin
#5. Check for plain hunt. 
#6. Check method database
#7. Deal with touches etc. and print out a composition?

def find_all_rows(raw_data):
    #Hopefully this is in a decent enough format
    nbells = np.max(raw_data)
    nrows = int(len(raw_data)/nbells)
    allrows = np.zeros((nrows, nbells))

    for ri in range(nrows):
        allrows[ri] = raw_data[ri*nbells:(ri+1)*nbells]
    
    return allrows.astype('int')

def find_method_time(all_rows):
    #Finds the longest time away from rounds, outputs start and finish
    nbells = len(all_rows[0])
    current_best = 0
    start = 0
    end  = 0
    isrounds = np.zeros(len(all_rows))

    for ri, row in enumerate(all_rows):
        if (sum(a == b for a, b in zip(row, np.arange(nbells) + 1))/nbells) > 0.9:   #This change is very close to rounds
            isrounds[ri] = 1.0
        else:
            isrounds[ri] = 0.0
    #Remove single blips
    for i in range(1,len(isrounds) - 1):
        if isrounds[i] != isrounds[i+1] and isrounds[i] != isrounds[i-1]:
            isrounds[i] = not(isrounds[i])

    current_best = 0; current_run = 0
    for i in range(1, len(isrounds)):
        if isrounds[i] == 0:
            current_run += 1
        else:
            current_run = 0
        if current_run > current_best:
            start = i - current_run
            end = i
            current_best = current_run
    return start, end + 1

def tostring(place):
    if place < 9:
        return str(place + 1)
    if place == 9:
        return '0'
    if place == 10:
        return 'E'
    if place == 11:
        return 'T'
    if place == 12:
        return 'A'
    if place == 13:
        return 'B'
    if place == 14:
        return 'C'
    if place == 15:
        return 'D'
    if place > 15:
        return 'X'
        
def find_place_notation(lead_rows):
    #Given the rows in the lead, determines the 'place notation' for each one. Do want in the spreadhseet format.
    nbells = len(lead_rows[0])
    def appendstring(notation, places):

        if len(places) < 1:
            notation = notation + '-'
        else:
            if len(notation) > 0:
                if notation[-1] != '-':
                    notation = notation + '.'
            for i in range(len(places)):
                notation = notation + tostring(places[i])   #This needs updating for higher than 10
        return notation

    notation = ''
    #Need a check here for bells which are stationary throughout (i.e. tenor behind)
    nworking = nbells
    for bell in range(nbells-1, -1, -1):
        count = np.sum(lead_rows[:,bell] == bell + 1)/len(lead_rows)
        if count > 0.75:
            nworking -= 1
    lead_rows = lead_rows[:,:nworking]
    for ri in range(len(lead_rows)-1):
        places = np.where(lead_rows[ri] == lead_rows[ri+1])[0]
        notation = appendstring(notation, places)
    return notation
    
#Find method types. Codes: P, T, S, X. Plain hunt, treble bob, stedman, respectively. X for can't figure it out

def find_method_types(trimmed_rows):

    #Attempts to determine treble type. Want to be able to do spliced Plain/Little etc. ideally but that may be tricky.
    def treble_position(trimmed_rows):
        positions = np.zeros(len(trimmed_rows))
        for ri, row in enumerate(trimmed_rows):
            positions[ri] = int(np.where(row == 1)[0][0])
        return positions

    def generate_treble_paths():
        #Create all treble hunt positions from 3 to 16. Both plain and treble bob if even.
        hunt_paths = [None, None]
        bob_paths = [None, None]
        for stage in range(3, 17):
            path = np.concatenate((np.arange(0, stage), np.arange(stage-1, -1, -1)))
            hunt_paths.append(path.tolist())
            if stage%2 == 0:
                pairs = path.reshape(-1, 2)
                repeat = np.repeat(pairs, 2, axis = 0)
                bob_paths.append(repeat.flatten().tolist())
            else:
                bob_paths.append(None)
        return hunt_paths, bob_paths        

    treble_path = treble_position(trimmed_rows)
    hunt_paths, bob_paths = generate_treble_paths()

    def check_for_all_stedman(rows):
        #Checks if the whole thing is Stedman, or if there is just a six of stemdan-like (if more than doubles)
        if len(rows) < 6:
            return False, 0
        else:
            notation = find_place_notation(rows)
            split_notation = notation.split(".")
            max_treble = 0
            for swap in split_notation:
                if len(swap) == 1:
                    max_treble = max(max_treble, tonumber(swap))

            if split_notation.count('1') > 0.3*len(split_notation) and split_notation.count('3') > 0.3*len(split_notation):
                return True, max_treble
            else:
                return False, 0

    def determine_hunt_types(trimmed_rows):
        #Compares the treble positions against the hunt and bob paths and returns the most likely. Hopefully should be obvious...
        start_index = 0
        go = True
        treble_data = []
        is_all_stedman, max_treble = check_for_all_stedman(trimmed_rows)
        if is_all_stedman:
            treble_data = [["S", max_treble - 1]]

        if not is_all_stedman:
            while go:
                bestmatch = 0.
                for pi, path in enumerate(hunt_paths):
                    if path is None:
                        continue
                    if start_index + len(path) > len(treble_path):
                        continue
                    match = sum(a == b for a, b in zip(treble_path[start_index:start_index + len(path)], path))/len(path)
                    if match > bestmatch:
                        bestmatch = match
                        code = 'P'
                        treble_stage = pi
                for pi, path in enumerate(bob_paths):
                    if path is None:
                        continue
                    if start_index + len(path) > len(treble_path):
                        continue
                    match = sum(a == b for a, b in zip(treble_path[start_index:start_index + len(path)], path))/len(path)
                    if match > bestmatch:
                        bestmatch = match
                        code = 'T'
                        treble_stage = pi
                if bestmatch < 0.6:   #This is very conservative... May well allow a lot of rubbish through but oh well
                    #Check for a stedman here? Worth a go. Don't have any spliced to test against though
                    if len(trimmed_rows[start_index:]) < 6 and len(trimmed_rows[start_index:start_index + 7]) > 4:
                        notation = find_place_notation(trimmed_rows[start_index:start_index + 7])
                        split_notation = notation.split(".")
                        max_treble = 0
                        if split_notation.count('1') == 2 and split_notation.count('3') == 2:
                            start_index = start_index + 6
                            for swap in split_notation:
                                if len(swap) == 1:
                                    max_treble = max(max_treble, int(swap))
                            treble_data.append(["S", max_treble - 1])
                    go = False

                else:
                    treble_data.append([code, treble_stage])
                    if code == 'T':
                        start_index = start_index + (treble_stage + 1)*4
                    elif code == 'P':
                        start_index = start_index + (treble_stage + 1)*2
                    else:
                        return 'X'
        return treble_data
    
    hunt_types = determine_hunt_types(trimmed_rows)
    #print('Hunt types', hunt_types)
    return hunt_types

def determine_methods(trimmed_rows, hunt_types, method_data):
    #Now have the number of leads and all the rows.
    #Need to generate place notation and compare against those in the database, for each lead.
    #Can be sped up by assuming it isn't spliced and then checking the rest

    #Returns two bits -- with method IDs and probabilities. Can leave the rest to be sorted out
    nbells = len(trimmed_rows[0])

    def determine_match(string1, string2):
        #Run through strings to see which is best. Bit awkward and horrid but meh. 
        split1 = re.split(r'(-)|\.', string1)
        split1 = [i for i in split1 if i != '.']
        split1 = [i for i in split1 if i not in (None, '')]
        split2 = re.split(r'(-)|\.', string2)
        split2 = [i for i in split2 if i != '.']
        split2 = [i for i in split2 if i not in (None, '')]
        match = sum(a == b for a, b in zip(split1, split2))/len(split1)
        return match

    if len(hunt_types) == 1 and hunt_types[0][0] == "S":
        #This is all stedman, so just do stedman things -- completely different! Determine the pattern of quicks and slows here, but not the composition.
        #The next section *should* catch stedman quicks and slows like they're normal methods, and just insert into the composition if it's Stedman and Bristol or something
        #Assuming nobody is stupid enough to spliced Stedman and Erin, the following should work:
        #Determine time of the first quick six (if any). Could be Erin of course
        if len(trimmed_rows) < 24:   #Fewer than four sixes, stop being silly
            return None, None, None
        
        spliced = []
        bestmatch = 0.
        possible_methods = method_data[method_data['Lead Length'] == 12]
        possible_methods = possible_methods[possible_methods['Type'] == "S"]
        possible_methods = possible_methods[possible_methods['Stage'] == nbells - 1]
        possible_notations = np.array([nots.rsplit(',', 1)[0] + '.' + nots.rsplit(',', 1)[1] for nots in possible_methods['Place Notation']])
        possible_stages = [stage for stage in possible_methods['Stage']] 
        for start in range(12):  #Run through the possible start points of the quick six. Normal Stedman start is 9, Erin is 0
            lead_rows = trimmed_rows[start:start + 13]
            check_notation = find_place_notation(lead_rows)
            for pi, poss in enumerate(possible_notations):
                match = determine_match(check_notation, poss)
                if match > bestmatch:
                    bestmatch = match
                    pbest = pi
                    startbest = start
                    stagebest = possible_stages[pi]
        
        spliced.append([possible_methods.iloc[pbest]['Name'] + ' Start ' + str(startbest), bestmatch, stagebest])
        
        return hunt_types, None, spliced

    else:
        if all(item == hunt_types[0] for item in hunt_types):
            #This could theoretically be a single method, so try to look for one
            sametype = True
        else:
            sametype = False

        all_notations = []
        current_start = 0
        pbest = 0
        for li, type in enumerate(hunt_types[:]):
            #Look for all the same method
            if type[0] == 'P':
                current_end = current_start + (type[1] + 1)*2 + 1
                lead_length =  (type[1] + 1)*2 
            elif type[0] == 'T':
                current_end = current_start + (type[1] + 1)*4 + 1
                lead_length =  (type[1] + 1)*4
            elif type[0] == 'S':
                current_end = current_start + 6 + 1
                lead_length =  6
            bestmatch = 0.
            lead_rows = trimmed_rows[current_start:current_end]
            current_start = current_end - 1 
            all_notations.append(find_place_notation(lead_rows))

        if sametype:
            possible_methods = method_data[method_data['Lead Length'] == lead_length]
            possible_methods = possible_methods[possible_methods['Type'] == type[0]]
            possible_methods = possible_methods[possible_methods['Stage'] <= nbells]
            possible_notations = np.array([nots.rsplit(',', 1)[0] for nots in possible_methods['Interior Notation']])
            
            bestmatch = 0.
            for pi, poss in enumerate(possible_notations):
                match = 0
                for li, type in enumerate(hunt_types[:]):
                    place_notation = all_notations[li]
                    match += determine_match(place_notation, poss)
                match = match/len(hunt_types)
                if match > bestmatch:
                    bestmatch = match
                    pbest = pi

            #print('Overall', li, round(bestmatch*100, 2), '%  match', possible_methods.iloc[pbest]['Name'])
            notspliced = [possible_methods.iloc[pbest]['Name'],  bestmatch, possible_methods.iloc[pbest]['Stage']]
        else:
            notspliced = None
            
        #Look for spliced leads (may as well always do this)
        current_start = 0
        spliced = []
        for li, type in enumerate(hunt_types[:]):
            pbest = 0
            if type[0] == 'P':
                current_end = current_start + (type[1] + 1)*2 + 1
                lead_length =  (type[1] + 1)*2 
            if type[0] == 'T':
                current_end = current_start + (type[1] + 1)*4 + 1
                lead_length =  (type[1] + 1)*4

            findnewdata = False
            if li == 0:
                findnewdata = True
            else:
                if hunt_types[li] != hunt_types[li-1]:
                    findnewdata = True

            if findnewdata:
                possible_methods = method_data[method_data['Lead Length'] == lead_length]
                possible_methods = possible_methods[possible_methods['Type'] == type[0]]
                possible_methods = possible_methods[possible_methods['Stage'] <= nbells]
                possible_notations = np.array([nots.rsplit(',', 1)[0] for nots in possible_methods['Interior Notation']])

            place_notation = all_notations[li]
            bestmatch = 0.
            for pi, poss in enumerate(possible_notations):
                match = determine_match(place_notation, poss)

                if match > bestmatch:
                    bestmatch = match
                    pbest = pi
            spliced.append([possible_methods.iloc[pbest]['Name'], bestmatch, possible_methods.iloc[pbest]['Stage']])

    return hunt_types, notspliced, spliced

def tonumber(string):
    if string.isnumeric():
        if int(string) > 0:
            return int(string)
        else:
            return 10
    elif string == 'E':
        return 11
    elif string == 'T':
        return 12
    elif string == 'A':
        return 13
    elif string == 'B':
        return 14
    elif string == 'C':
        return 15
    elif string == 'D':
        return 16
    else:
        raise Exception('Too many bells')

def tostring_direct(place):
    #Returns the direct string of a place. Used sometimes...
    if place < 10:
        return str(place)
    if place == 10:
        return '0'
    if place == 11:
        return 'E'
    if place == 12:
        return 'T'
    if place == 13:
        return 'A'
    if place == 14:
        return 'B'
    if place == 15:
        return 'C'
    if place == 16:
        return 'D'
    if place > 15:
        return 'X'

def find_stedman_composition(trimmed_rows, hunt_types, methods_notspliced, methods_spliced, method_data):
    #Finds a Stedman composition. Fairly self-explanatory. The methods_spliced shows how many blows until the first complete slow six (3 for normal stedman start)
    nbells = len(trimmed_rows[0])

    def generate_rows(first_change, place_notation):
        #From the change first_change (first one not in rounds, etc., generate all the rows in a lead for comparison with the lead lumps)
        notation_list = re.split(r'(\-|\.)', place_notation)
        notation_list = [i for i in notation_list if i not in ['','.']]
        
        newrows = [first_change]
        for i, swap in enumerate(notation_list):
            nextrow = newrows[-1].copy()
            if swap == '-':  #Swap all places. CAREFUL HERE, is not necessarily every place. Need stage too? Not for now.
                if len(nextrow)%2 == 0:
                    nextrow[::2] = newrows[-1][1::2]
                    nextrow[1::2] = newrows[-1][::2]
                else:   #Minor with a cover etc, theoretically?
                    nextrow[:-1:2] = newrows[-1][1::2]
                    nextrow[1::2] = newrows[-1][:-1:2]
            else:
                place = 0; not_place = 0
                while place < len(first_change) - 1:
                    if not_place < len(swap):
                        if swap[not_place] == tostring_direct(place + 1):   #This one is the same
                            nextrow[place] = newrows[-1][place]
                            place += 1; not_place += 1
                        else:
                            nextrow[place] = newrows[-1].copy()[place + 1]
                            nextrow[place + 1] = newrows[-1].copy()[place]
                            place += 2
                    else:   #This one is not -- change it
                        nextrow[place] = newrows[-1].copy()[place + 1]
                        nextrow[place + 1] = newrows[-1].copy()[place]
                        place += 2
            newrows.append(nextrow)
        return np.array(newrows)

    def find_leadend_options_stedman(previous_change, stage, quick = False):
        #This one is for single-hunt methods where the call affects one change at the lead end itself
        #Outputs options, with PP BB SS near and far (should be 6 or thereabouts)
        nbells = len(previous_change)
        if nbells > 6:
            notation_list = [tostring_direct(nbells - 1) + tostring_direct(nbells), tostring_direct(nbells - 3) + tostring_direct(nbells), tostring_direct(nbells - 3) + tostring_direct(nbells - 2) + tostring_direct(nbells - 1) + tostring_direct(nbells)]
        else:
            if quick:
                notation_list = ['3', '3', '345']
            else:
                notation_list = ['1', '1', '145']
        options = []
        for i, swap in enumerate(notation_list):
            nextrow = previous_change.copy()
            if swap == '-':  #Swap all places
                nextrow[::2] = previous_change[1::2]
                nextrow[1::2] = previous_change[::2]
            else:
                place = 0; not_place = 0
                while place < len(previous_change) - 1:
                    if not_place < len(swap):
                        if swap[not_place] == tostring_direct(place + 1):   #This one is the same
                            nextrow[place] = previous_change[place]
                            place += 1; not_place += 1
                        else:
                            nextrow[place] = previous_change.copy()[place + 1]
                            nextrow[place + 1] = previous_change.copy()[place]
                            place += 2
                    else:   #This one is not -- change it
                        nextrow[place] = previous_change.copy()[place + 1]
                        nextrow[place + 1] = previous_change.copy()[place]
                        place += 2

            options.append(nextrow)
        return np.array(options)

    def compare_set(target_rows, test_rows):
        same_count = np.sum(target_rows[:-1,:] == test_rows[:-1,:])
        return same_count/np.size(target_rows)

    #Establish first few changes, based on the number (allowing for funny starts)
    is_erin = methods_spliced[0][0][:4] == "Erin" #If erin, only slow sixes
    start_offset = int(methods_spliced[0][0].rsplit(' ', 1)[-1])
    is_quick = False #This should alternate between the sixes if necessary
    if not is_erin and start_offset > 0 and start_offset < 7:
        is_quick = True

    if not is_erin:
        quick_title = methods_spliced[0][0].rsplit(' ',2)[0] + " Quick"
        slow_title = methods_spliced[0][0].rsplit(' ',2)[0] + " Slow"

        quick_pn = method_data[method_data['Name'] == quick_title]['Interior Notation'].values[0]
        slow_pn = method_data[method_data['Name'] == slow_title]['Interior Notation'].values[0]
    else:
        slow_title = methods_spliced[0][0].rsplit(' ',2)[0] + " Slow"
        slow_pn = method_data[method_data['Name'] == slow_title]['Interior Notation'].values[0]

    lead_end_options = [np.arange(nbells) + 1] #Assume it starts in rounds (one hopes...)

    nchanges_start = (start_offset)%6

    best_calls_spliced = []; qualities_spliced = []

    if start_offset%6 > 0:
        if is_quick:
            test_rows = generate_rows(lead_end_options[0], quick_pn[-nchanges_start*2:])
        else:
            test_rows = generate_rows(lead_end_options[0], slow_pn[-nchanges_start*2:])
    else: #Starts at the beginning of the six (probably won't happen as this is a backstroke, unless doubles)
        if is_quick:
            test_rows = generate_rows(lead_end_options[0], quick_pn)
        else:
            test_rows = generate_rows(lead_end_options[0], slow_pn)

    allrows_spliced = test_rows
    target_rows = trimmed_rows[:len(test_rows)]

    option_quality = compare_set(target_rows, test_rows)
    qualities_spliced.append(option_quality)

    current_start = len(test_rows) - 1
    current_end = current_start + 7

    while current_end < len(trimmed_rows) + 1:
        if not is_erin:
            is_quick = not is_quick
        #Find rows to match
        target_rows = trimmed_rows[current_start:current_end]

        #Find options for new lead end
        lead_end_options = find_leadend_options_stedman(allrows_spliced[-2], nbells - 1, quick = is_quick)
        option_quality = []
        for i in range(len(lead_end_options)):
            if is_quick:
                test_rows = generate_rows(lead_end_options[i], quick_pn[:])
            else:
                test_rows = generate_rows(lead_end_options[i], slow_pn[:])
            option_quality.append(compare_set(target_rows, test_rows))
        best_call = np.where(option_quality == np.max(option_quality))[0][0]
        best_calls_spliced.append(best_call)
        qualities_spliced.append(np.max(option_quality))
        if is_quick:
            new_rows = generate_rows(lead_end_options[best_call], quick_pn[:])
        else:
            new_rows = generate_rows(lead_end_options[best_call], slow_pn[:])
        allrows_spliced = np.concatenate((allrows_spliced[:-1], new_rows), axis = 0)
        current_start = current_end - 1 
        current_end = current_start + 7

    #Then finish off by seeing if this can come into rounds with a call? Also need to add the extra changes...
    #Could bother with changes here, but we'll see
    #Could be any length really

    lead_end_options = find_leadend_options_stedman(allrows_spliced[-2], nbells - 1)

    option_quality = []
    for i in range(len(lead_end_options)):
        same_count = np.sum(trimmed_rows[current_start] == lead_end_options[i])
        option_quality.append(same_count/np.size(trimmed_rows[current_start]))
    best_call = np.where(option_quality == np.max(option_quality))[0][0]
    best_calls_spliced.append(best_call)
    best_calls_spliced = np.array(best_calls_spliced); qualities_spliced = np.array(qualities_spliced)

    allrows_spliced[-1] = lead_end_options[best_call]

    #Finally, try to make up any last rows if it gets into rounds (if it doesn't work, just leave this out -- things may have fired out etc.)
    if len(trimmed_rows) > len(allrows_spliced):
        if not is_erin:
            is_quick = not is_quick
        nrows_left = len(trimmed_rows) - len(allrows_spliced)
        if is_quick:
            end_rows = generate_rows(allrows_spliced[-1], quick_pn[:])
        else:
            end_rows = generate_rows(allrows_spliced[-1], slow_pn[:])
        if len(end_rows) > nrows_left:
            if (trimmed_rows[-1] == end_rows[nrows_left]).all():
                allrows_spliced = np.concatenate((allrows_spliced[:-1], end_rows[:nrows_left+1]), axis = 0)

    return True, best_calls_spliced, allrows_spliced

def find_composition(trimmed_rows, hunt_types, methods_notspliced, methods_spliced, method_data):
    #Finds the best match composition for trimmed_rows. Will check not spliced and spliced and see which is best
    #Obvioulsy one would expect spliced to be best, but not if the ringing is terrible
    #Will need to add a Stedman flag at some point...
    nbells = len(trimmed_rows[0])
    def generate_rows(first_change, place_notation):
        #From the change first_change (first one not in rounds, etc., generate all the rows in a lead for comparison with the lead lumps)
        notation_list = re.split(r'(\-|\.)', place_notation)
        notation_list = [i for i in notation_list if i not in ['','.']]
        newrows = [first_change]
        for i, swap in enumerate(notation_list):
            nextrow = newrows[-1].copy()
            if swap == '-':  #Swap all places
                nextrow[::2] = newrows[-1][1::2]
                nextrow[1::2] = newrows[-1][::2]
            else:
                place = 0; not_place = 0
                while place < len(first_change) - 1:
                    if not_place < len(swap):
                        if swap[not_place] == tostring_direct(place + 1):   #This one is the same
                            nextrow[place] = newrows[-1][place]
                            place += 1; not_place += 1
                        else:
                            nextrow[place] = newrows[-1].copy()[place + 1]
                            nextrow[place + 1] = newrows[-1].copy()[place]
                            place += 2
                    else:   #This one is not -- change it
                        nextrow[place] = newrows[-1].copy()[place + 1]
                        nextrow[place + 1] = newrows[-1].copy()[place]
                        place += 2
            newrows.append(nextrow)
        return np.array(newrows)

    def find_leadend_options_single(previous_change, stage):
        #This one is for single-hunt methods where the call affects one change at the lead end itself
        #Outputs options, with PP BB SS near and far (should be 6 or thereabouts)
        nbells = len(previous_change)
        if stage == nbells:
            notation_list = ['12', '1' + tostring_direct(nbells), '14', '1' + tostring_direct(nbells - 2), '1234', '1' + tostring_direct(nbells - 2) + tostring_direct(nbells - 1) + tostring_direct(nbells)]
        else:
            notation_list = ['12' + tostring_direct(nbells - 1), '1', '14' + tostring_direct(nbells - 1), '1' + tostring_direct(nbells - 2) + tostring_direct(nbells - 1), '1234' + tostring_direct(nbells - 1), '1' + tostring_direct(nbells - 2) + tostring_direct(nbells - 1) + tostring_direct(nbells)]
        options = []
        for i, swap in enumerate(notation_list):
            nextrow = previous_change.copy()
            if swap == '-':  #Swap all places
                nextrow[::2] = previous_change[1::2]
                nextrow[1::2] = previous_change[::2]
            else:
                place = 0; not_place = 0
                while place < len(previous_change) - 1:
                    if not_place < len(swap):
                        if swap[not_place] == tostring_direct(place + 1):   #This one is the same
                            nextrow[place] = previous_change[place]
                            place += 1; not_place += 1
                        else:
                            nextrow[place] = previous_change.copy()[place + 1]
                            nextrow[place + 1] = previous_change.copy()[place]
                            place += 2
                    else:   #This one is not -- change it
                        nextrow[place] = previous_change.copy()[place + 1]
                        nextrow[place + 1] = previous_change.copy()[place]
                        place += 2

            options.append(nextrow)
        return np.array(options)

    def find_leadend_options_twin(existing_changes, stage, notation):
        #This is for methods like grandsire, where the call comes into effect one earlier
        nbells = len(existing_changes[-1])
        #Keep with six options for ease
        notation_list = [[tostring_direct(stage),'1'], [tostring_direct(stage),'1'], ['3','1'],['3','1'], ['3','123'], ['3','123']]
        options = []; options_previous = []
        for notation_test in notation_list:
            newrows = [existing_changes[-3]]
            for i, swap in enumerate(notation_test):
                nextrow = newrows[-1].copy()
                if swap == '-':  #Swap all places
                    nextrow[::2] = newrows[-1][1::2]
                    nextrow[1::2] = newrows[-1][::2]
                else:
                    place = 0; not_place = 0
                    while place < len(newrows[-1]) - 1:
                        if not_place < len(swap):
                            if swap[not_place] == tostring_direct(place + 1):   #This one is the same
                                nextrow[place] = newrows[-1][place]
                                place += 1; not_place += 1
                            else:
                                nextrow[place] = newrows[-1].copy()[place + 1]
                                nextrow[place + 1] = newrows[-1].copy()[place]
                                place += 2
                        else:   #This one is not -- change it
                            nextrow[place] = newrows[-1].copy()[place + 1]
                            nextrow[place + 1] = newrows[-1].copy()[place]
                            place += 2

                newrows.append(nextrow)
            options.append(newrows[-1]); options_previous.append(newrows[-2])
        return np.array(options), np.array(options_previous)

    def compare_set(target_rows, test_rows):
        same_count = np.sum(target_rows[:-1,:] == test_rows[:-1,:])
        return same_count/np.size(target_rows)
    #Not spliced first
    if methods_notspliced is not None:
        current_start = 0
        lead_end_options = [np.arange(nbells) + 1] #Assume it starts in rounds (one hopes...)
        best_calls_single = []; qualities_single = []
        for li, type in enumerate(hunt_types[:]):
            if type[0] == 'P':
                current_end = current_start + (type[1] + 1)*2 + 1
                lead_length =  (type[1] + 1)*2 
            if type[0] == 'T':
                current_end = current_start + (type[1] + 1)*4 + 1
                lead_length =  (type[1] + 1)*4

            #Find rows to match
            target_rows = trimmed_rows[current_start:current_end]
            notation = method_data[method_data['Name'] == methods_notspliced[0]]['Interior Notation'].values[0]

            if len(target_rows) <= lead_length:
                break
            if li > 0:
                #Find options for new lead end
                if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 1:
                    lead_end_options = find_leadend_options_single(allrows_single[-2], method_data[method_data['Name'] == methods_notspliced[0]]['Stage'].values[0])
                else:
                    lead_end_options, previous_options = find_leadend_options_twin(allrows_single, method_data[method_data['Name'] == methods_notspliced[0]]['Stage'].values[0], notation)
                option_quality = []
                for i in range(len(lead_end_options)):
                    test_rows = generate_rows(lead_end_options[i], notation)
                    option_quality.append(compare_set(target_rows, test_rows))
                best_call = np.where(option_quality == np.max(option_quality))[0][0]
                best_calls_single.append(best_call)
                qualities_single.append(np.max(option_quality))
                new_rows = generate_rows(lead_end_options[best_call], notation)
                if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 2:
                    allrows_single[-2] = previous_options[best_call]
                allrows_single = np.concatenate((allrows_single[:-1], new_rows), axis = 0)
            else:
                #Starting from rounds, one assumes
                test_rows = generate_rows(lead_end_options[0], notation)
                allrows_single = test_rows
                option_quality = compare_set(target_rows, test_rows)
                qualities_single.append(option_quality)
            current_start = current_end - 1 
        
        #Then finish off by seeing if this can come into rounds with a call?
        if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 1:
            lead_end_options = find_leadend_options_single(allrows_single[-2], method_data[method_data['Name'] == methods_notspliced[0]]['Stage'].values[0])
        else:
            lead_end_options, previous_options = find_leadend_options_twin(allrows_single, method_data[method_data['Name'] == methods_notspliced[0]]['Stage'].values[0],notation )

        option_quality = []
        for i in range(len(lead_end_options)):
            same_count = np.sum(trimmed_rows[current_start] == lead_end_options[i])
            option_quality.append(same_count/np.size(trimmed_rows[current_start]))
        best_call = np.where(option_quality == np.max(option_quality))[0][0]
        best_calls_single.append(best_call)

        best_calls_single = np.array(best_calls_single); qualities_single = np.array(qualities_single)
        if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 2:
            allrows_single[-2] = previous_options[best_call]

        allrows_single[-1] = lead_end_options[best_call]

        #Finally, try to make up any last rows if it gets into rounds (if it doesn't work, just leave this out -- things may have fired out etc.)
        if len(trimmed_rows) > len(allrows_single):
            nrows_left = len(trimmed_rows) - len(allrows_single)
            end_notation = method_data[method_data['Name'] == methods_notspliced[0]]['Interior Notation'].values[0]
            end_rows = generate_rows(allrows_single[-1], end_notation)
            if len(end_rows) > nrows_left:
                if (trimmed_rows[-1] == end_rows[nrows_left]).all():
                    allrows_single = np.concatenate((allrows_single[:-1], end_rows[:nrows_left+1]), axis = 0)
                else:  #Just finish off with what actually happened - something fired out
                    allrows_single = np.concatenate((allrows_single[:], trimmed_rows[len(allrows_single):]), axis = 0)
            else:
                allrows_single = np.concatenate((allrows_single[:], trimmed_rows[len(allrows_single):]), axis = 0)

    else:
        best_calls_single = None
        qualities_single = None
        
    #Then spliced
    current_start = 0
    lead_end_options = [np.arange(nbells) + 1] #Assume it starts in rounds (one hopes...)
    best_calls_spliced = []; qualities_spliced = []
    for li, type in enumerate(hunt_types[:]):
        if type[0] == 'P':
            current_end = current_start + (type[1] + 1)*2 + 1
            lead_length =  (type[1] + 1)*2 
        if type[0] == 'T':
            current_end = current_start + (type[1] + 1)*4 + 1
            lead_length =  (type[1] + 1)*4

        #Find rows to match
        target_rows = trimmed_rows[current_start:current_end]
        notation = method_data[method_data['Name'] == methods_spliced[li][0]]['Interior Notation'].values[0]
        if len(target_rows) <= lead_length:
            break

        if li > 0:
            #Find options for new lead end
            if method_data[method_data['Name'] == methods_spliced[li][0]]['Hunt Number'].values[0] == 1:
                lead_end_options = find_leadend_options_single(allrows_spliced[-2], method_data[method_data['Name'] == methods_spliced[li][0]]['Stage'].values[0])
            else:
                lead_end_options, previous_options = find_leadend_options_twin(allrows_spliced, method_data[method_data['Name'] == methods_spliced[li][0]]['Stage'].values[0],notation )

            option_quality = []
            for i in range(len(lead_end_options)):
                test_rows = generate_rows(lead_end_options[i], notation)
                option_quality.append(compare_set(target_rows, test_rows))
            best_call = np.where(option_quality == np.max(option_quality))[0][0]
            best_calls_spliced.append(best_call)
            qualities_spliced.append(np.max(option_quality))
            new_rows = generate_rows(lead_end_options[best_call], notation)
            if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 2:
                allrows_spliced[-2] = previous_options[best_call]
            allrows_spliced = np.concatenate((allrows_spliced[:-1], new_rows), axis = 0)

        else:
            #print(methods_notspliced[0])
            notation = method_data[method_data['Name'] == methods_spliced[li][0]]['Interior Notation'].values[0]
            test_rows = generate_rows(lead_end_options[0], notation)
            option_quality = compare_set(target_rows, test_rows)
            allrows_spliced = test_rows
            qualities_spliced.append(option_quality)

        current_start = current_end - 1 

    #Then finish off by seeing if this can come into rounds with a call?
    if method_data[method_data['Name'] == methods_spliced[-1][0]]['Hunt Number'].values[0] == 1:
        lead_end_options = find_leadend_options_single(allrows_spliced[-2], method_data[method_data['Name'] == methods_spliced[-1][0]]['Stage'].values[0])
    else:
        lead_end_options, previous_options = find_leadend_options_twin(allrows_spliced, method_data[method_data['Name'] == methods_spliced[-1][0]]['Stage'].values[0],notation )

    option_quality = []
    for i in range(len(lead_end_options)):
        same_count = np.sum(trimmed_rows[current_start] == lead_end_options[i])
        option_quality.append(same_count/np.size(trimmed_rows[current_start]))
    best_call = np.where(option_quality == np.max(option_quality))[0][0]
    if method_data[method_data['Name'] == methods_notspliced[0]]['Hunt Number'].values[0] == 2:
        allrows_spliced[-2] = previous_options[best_call]
    allrows_spliced[-1] = lead_end_options[best_call]
    best_calls_spliced.append(best_call)
    best_calls_spliced = np.array(best_calls_spliced); qualities_spliced = np.array(qualities_spliced)
    
    #Finally, try to make up any last rows if it gets into rounds (if it doesn't work, just leave this out -- things may have fired out etc.)
    if len(trimmed_rows) > len(allrows_spliced):
        nrows_left = len(trimmed_rows) - len(allrows_spliced)
        end_notation = method_data[method_data['Name'] == methods_spliced[-1][0]]['Interior Notation'].values[0]
        end_rows = generate_rows(allrows_spliced[-1], end_notation)
        if len(end_rows) > nrows_left:  #This is a good match - do this
            if (trimmed_rows[-1] == end_rows[nrows_left]).all():
                allrows_spliced = np.concatenate((allrows_spliced[:-1], end_rows[:nrows_left+1]), axis = 0)
            else:  #Just finish off with what actually happened - something fired out
                allrows_spliced = np.concatenate((allrows_spliced[:], trimmed_rows[len(allrows_spliced):]), axis = 0)
        else:
            allrows_spliced = np.concatenate((allrows_spliced[:], trimmed_rows[len(allrows_spliced):]), axis = 0)

    spliced_quality = compare_set(trimmed_rows, allrows_spliced)
    if allrows_single is not None:
        single_quality = compare_set(trimmed_rows, allrows_single)

    if single_quality is not None:
        if np.sum(spliced_quality) > np.sum(single_quality):
            return True, best_calls_spliced, allrows_spliced
        else:
            return False, best_calls_single, allrows_single
    else:
        return True, best_calls_spliced, allrows_spliced

def check_lead_ends(methods, calls, nbells, method_data):
    #Checks the plain leads against the method in the previous section, in case of a ballsed-up lead end (common enough to care about I think)
    for mi, method in enumerate(methods):
        if calls[mi] > 1:
            #Is a bob. Just go with it and keep the same. It's probably fine. Or could check against the previous methods? Yeah.
            if mi > 0:
                not_old = method_data[method_data['Name'] == methods[mi-1][0]]['Place Notation'].values[0]
                not_new = method_data[method_data['Name'] == method[0]]['Place Notation'].values[0]
                if not_old == not_new:  #Is the same
                    continue
                not_old = not_old.rsplit(',', 1)[0]
                not_new = not_new.rsplit(',', 1)[0]
                if not_old != not_new:  #Is not -- could have spurious switch
                    continue
                methods[mi][0] = methods[mi-1][0]
        else:

            stage = method_data[method_data['Name'] == method[0]]['Stage'].values[0]

            if stage == nbells:
                notation_list = ['12', '1' + tostring_direct(nbells)]
            else:
                notation_list = ['12' + tostring_direct(nbells - 1), '1']

            nots = method_data[method_data['Name'] == method[0]]['Place Notation'].values[0]
            method_lead_end = nots.rsplit(',', 1)[-1]

            if calls[mi] == 1 and method[0][:9] == 'Plain Bob':
                methods[mi][0] = 'Plain Hunt' + ' ' + method[0].rsplit(' ')[-1]
            #Determine if a swap is needed
            if method_lead_end == notation_list[0]:
                lead_type = 0
            elif method_lead_end == notation_list[1]:
                lead_type = 1
            else:
                lead_type = -1

            if lead_type == 1 and calls[mi] == 0:
                target_notation = nots.rsplit(',', 1)[0] + ',' + notation_list[1]
                if len(method_data[method_data['Place Notation'] == target_notation]['Name'].values) > 0:
                    method = method_data[method_data['Place Notation'] == target_notation]['Name'].values[0]
                    methods[mi][0] = method
            elif calls[mi] == 0 and lead_type == 1:
                target_notation = nots.rsplit(',', 1)[0] + ',' + notation_list[0]
                if len(method_data[method_data['Place Notation'] == target_notation]['Name'].values) > 0:
                    method = method_data[method_data['Place Notation'] == target_notation]['Name'].values[0]
                    methods[mi][0] = method

    return methods

@st.cache_data
def find_method_things(raw_data):
    
    method_data = pd.read_csv('./method_data/clean_methods.csv')

    all_rows = find_all_rows(raw_data)
    nbells = len(all_rows[0])
    start_row, end_row = find_method_time(all_rows)

    if end_row - start_row < 10:  #Is just rounds, probably. That's fine.
        return [], [], [], 0, len(all_rows), all_rows, 1
    
    trimmed_rows = all_rows[start_row:end_row+1]   #Includes all the changes we care about, rounds EITHER END inclusive

    hunt_types = find_method_types(trimmed_rows)

    if len(hunt_types) == 0:
        methods = []
        calls = []
        allrows = all_rows
        count = 1.
    else:
        hunt_types, methods_notspliced, methods_spliced = determine_methods(trimmed_rows, hunt_types, method_data)

        #print(methods_spliced)
        if len(methods_spliced) == 0:
            methods = []
        else:
            if len(hunt_types) == 1 and hunt_types[0][0] == "S":   #Stedman behaves differently so do need a different thing here entirely. Will do later...
                spliced_flag, calls, allrows = find_stedman_composition(trimmed_rows, hunt_types, methods_notspliced, methods_spliced, method_data)
            else:
                spliced_flag, calls, allrows = find_composition(trimmed_rows, hunt_types, methods_notspliced, methods_spliced, method_data)

            if spliced_flag:
                methods = methods_spliced
            else:
                methods = [methods_notspliced]

            if spliced_flag and len(hunt_types) != 1 or methods[0][0][:9] == 'Plain Bob':
                #In spliced, check there aren't unnecessary changes of method name due to bobs happening
                methods = check_lead_ends(methods, calls, nbells, method_data) 
            
            #print('Methods:', methods)
            #print('Calls:', calls)
            #print('Calculated length', len(trimmed_rows), len(allrows))
            count = np.sum(trimmed_rows[:len(allrows)] == allrows)/np.size(allrows)
            #print('Match:', count)

    return methods, hunt_types, calls, start_row, end_row, allrows, count

@st.cache_data
def print_composition(methods, hunt_types, calls, relevant_rows):
    #Should output a markdown of the calling and composition lead-by-lead.
    #All touches output simple PSPPBPPSP etc. 
    #PB (and Grandsire?) do calling positions from the 'tenor' as well, if not a plain course
    #Perhaps figure out nice way of doing stedman and Erin but course lengths are hard
    def produce_call_string(calls, stedman_flag):
        call_string = ''
        for call in calls:
            if not stedman_flag:
                if call in [0,1]:
                    call_string = call_string + 'P'
                elif call in [2,3]:
                    call_string = call_string + 'B'
                elif call in [4,5]:
                    call_string = call_string + 'S'
            else:
                if call == 0:
                    call_string = call_string + 'P'
                elif call == 1:
                    call_string = call_string + 'B'
                elif call == 2:
                    call_string = call_string + 'S'
        return call_string

    def find_lead_ends(hunt_types, relevant_rows, methods):

        if hunt_types[0][0] == 'S':
            #Is stedman, need to start row in a strange place
            start_position = int(methods[0][0].rsplit(" ")[-1])%6 #Number of rows until the next six
            row_number = -start_position
            lead_length = 6
            row_number += lead_length - 1
            lead_ends = [relevant_rows[row_number]]
            while row_number < len(relevant_rows):
                row_number += lead_length
                if row_number < len(relevant_rows):
                    lead_ends.append(relevant_rows[row_number])
        else:
            row_number = 0   #Need to adjust (probably) for stedman as the lead ends are funny
            lead_ends = [relevant_rows[row_number]]
            for li, lead in enumerate(hunt_types):
                if lead[0] == 'T':
                    lead_length = 4*(lead[1] + 1)
                elif lead[0] == 'P':
                    lead_length = 2*(lead[1] + 1)
                row_number += lead_length
                if row_number < len(relevant_rows):
                    lead_ends.append(relevant_rows[row_number])

        return np.array(lead_ends)
    
    def call_position_name(position, stage):
        #Outputs call position as a string. Using what I think is correct but may be wrong...
        if position == 1:
            return "I"
        elif position == 2:
            return "B"
        elif position == 3:
            return "F"
        elif position == stage-1:
            return "H"
        elif position == stage - 2:
            return "W"
        elif position == stage - 3:
            return "M"
        elif position == 4:
            return "V"
        else:
            return str(position+1)
        
    def find_course_ends(lead_ends, hunt_types, methods):
        stage = methods[0][2]
        course_ends = np.zeros(len(lead_ends))
        #Just outputs as the number of leads since the start
        if hunt_types[0][0] == "T" or hunt_types[0][0] == "P":
            #Look for times the tenor returns to home
            for li, lead_end in enumerate(lead_ends):
                if np.where(lead_end == stage)[0] == stage - 1:
                    course_ends[li] = 1
                else:
                    course_ends[li] = 0
            for i in range(len(course_ends)-1):
                if course_ends[i] == 1 and course_ends[i+1] == 1:   #Get rid of repeats, remove the first one
                    course_ends[i] = 0
        elif hunt_types[0][0] == "S":
            if stage == 5:
                for i in range(10,len(lead_ends), 10):
                    course_ends[i] = 1
            else:
                #Is Stedman, so more complicated
                def change_distance(change1, change2):
                    #Gives a 'distance' metric fbetween each change
                    dsum = 0
                    for p1, bell in enumerate(change1):
                        p2 = np.where(change2 == bell)[0][0]
                        dsum += np.abs(p2 - p1)
                    return dsum
                #Course length range
                range_min = min(stage*2 - 10, len(lead_ends))
                range_max = min(stage*2 + range_min - 2, len(lead_ends))
                last_course_end = 0
                go = True
                start_offset = int(methods[0][0].rsplit(" ")[-1])
                if start_offset > 5:
                    starts_slow = 1
                else:
                    starts_slow = 2

                count = 0
                while go:
                    if last_course_end + stage*2 > len(lead_ends):
                        go = False
                    if last_course_end + range_max > len(lead_ends):
                        go = False
                    if count > 100:
                        go = False
                    dists = 100*np.ones(len(lead_ends))
                    for li in range(starts_slow, len(lead_ends), 2):
                        lead_end = lead_ends[li]
                        dists[li] = change_distance(lead_ends[last_course_end], lead_end)
                    lead_end = lead_ends[-1]
                    dists[-1] = change_distance(lead_ends[last_course_end], lead_end)
                    if last_course_end + range_min > len(dists) - 6:
                        break
                    poss = np.where(dists[last_course_end + range_min:last_course_end + range_max] == np.min(dists[last_course_end + range_min:last_course_end + range_max]))[0]
                    if stage*2 - range_min in poss:
                        course_end = stage*2 + last_course_end
                    else:
                        course_end = poss[0] + last_course_end + range_min

                    course_ends[course_end] = 1
                    last_course_end = course_end
        return course_ends


    def find_call_positions(call_string, lead_ends, methods):
        stage = methods[0][2]
        positions = []
        
        for ci in range(len(call_string)):
            if call_string[ci] == 'B':
                position = np.where(lead_ends[ci+1] == stage)[0][0]
                positions.append('-' + call_position_name(position, stage))
            elif call_string[ci] == 'S':
                position = np.where(lead_ends[ci+1] == stage)[0][0]
                positions.append('s' + call_position_name(position, stage))
            else:
                positions.append('  ')
        return positions

    def find_call_positions_stedman(call_string, lead_ends, methods, course_ends):
        stage = methods[0][2]
        course_length = stage*2
        positions = []
        course_ends[0] = 1.
        for ci in range(len(call_string)):
            if ci > 0:
                last_course_end = np.max(np.where(course_ends[:ci+1] == 1)[0])
            else:
                last_course_end = 0
            if call_string[ci] == 'B':
                positions.append('-' + str((ci + 1) - last_course_end))
            elif call_string[ci] == 'S':
                positions.append('s'  + str((ci + 1)- last_course_end))
            else:
                positions.append('  ')
        return positions
    
    def cleanup_methods(methods, calls):
        #Removes duplicate method names
        clean_methods = []
        for i in range(len(calls)):
            if len(methods) > 1:
                method = methods[i]
            else:
                method = methods[0]
            
            if method[0][:4] == "Erin" or method[0][:7] == "Stedman":
                method[0] = method[0].rsplit(" ")[0] + " " + method[0].rsplit(" ")[1]

            if len(clean_methods) == 0:
                clean_methods.append(method[0])
                current_method = method[0]
            else:
                if method[0] != current_method:
                    clean_methods.append(method[0])
                    current_method = method[0]
                else:
                    clean_methods.append(' ')
        return clean_methods
    
    if hunt_types[0][0] == "S":
        is_stedman = True
    else:
        is_stedman = False

 
    lead_ends = find_lead_ends(hunt_types, relevant_rows, methods)

    course_ends = find_course_ends(lead_ends, hunt_types, methods)  #Returns as a string, could be none

    call_string = produce_call_string(calls, is_stedman)

    if len(set(call_string)) == 1 and call_string[0] == 'P':
        plain_course = True
    else:
        plain_course = False

    clean_methods = cleanup_methods(methods, calls)

    if not is_stedman and not plain_course:
        call_positions = find_call_positions(call_string, lead_ends, methods)
    elif not is_stedman:
        call_positions = ' ' * (len(lead_ends) - 1)
    else:
        call_positions = find_call_positions_stedman(call_string, lead_ends, methods, course_ends)
    #Determine markdown widths etc.
    max_method_width = np.max([len(method) for method in clean_methods])
    max_call_width = np.max([len(call) for call in call_positions])
    lead_end_width = len(lead_ends[0])
    pad_width = 4
    total_width = max_method_width + max_call_width + lead_end_width + pad_width*2

    def clean_lead_ends(lead_ends):
        #Change to string
        clean_ends = []
        for lead_end in lead_ends:
            clean = ''
            
            for place in lead_end:
                clean = clean + tostring_direct(place)
            clean_ends.append(clean)
        return clean_ends

    lead_ends = clean_lead_ends(lead_ends)
    lines = []
    if plain_course and len(methods) == 1:
        lines.append('Plain Course <br>')
    else:
        lines.append(' <br>')

    lines.append("<u>" + ' '*(total_width - lead_end_width ) + str(lead_ends[0]) + "</u>"  + "<br>")

    for i in range(0, len(lead_ends) -1):
        if len(clean_methods) > 1:
            pad_method = ' '*(max_method_width - len(clean_methods[i])) + str(clean_methods[i])
        else:
            pad_method = ' '*(max_method_width - len(clean_methods[0])) + str(clean_methods[0])
        pad_call = ' '*(max_call_width - len(call_positions[i])) + str(call_positions[i])

        underline = False
        if (relevant_rows[-1] == np.arange(len(lead_ends[0])) + 1).all() and i == len(lead_ends) - 2:
            underline = True
        if course_ends[i+1] == 1:
            underline = True
        if not underline:
            lines.append(pad_method + ' '*(pad_width) + pad_call + ' '*(pad_width) + str(lead_ends[i + 1]) + "<br>")
        else:
            lines.append("<u>" + pad_method + ' '*(pad_width) + pad_call + ' '*(pad_width) + str(lead_ends[i + 1]) + "</u>" + "<br>")
    lead_end_html = '<pre>' +  ' '.join(lines) + '</pre>'
    return call_string, lead_end_html
