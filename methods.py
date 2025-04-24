#This script is for analysing methods. 
#Takes the raw timing csv as an input (or can pass less if necessary) and will do the analysis from that. Hopefully don't need anything else

import pandas as pd
import numpy as np
import re 
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

#raw_data = pd.read_csv('./striking_data/burley.csv')
#raw_data = pd.read_csv('./striking_data/brancepeth_cambridge.csv')
#raw_data = pd.read_csv('./striking_data/stockton_max.csv')
#raw_data = pd.read_csv('./striking_data/brancepeth_grandsire.csv')
#raw_data = pd.read_csv('./striking_data/PB7_Brancepeth.csv')
#raw_data = pd.read_csv('./striking_data/Little_Bob_Nics.csv')
#raw_data = pd.read_csv('./striking_data/St_Clements_nics.csv')
raw_data = pd.read_csv('./striking_data/Spliced_nics.csv')


method_data = pd.read_csv('./method_data/clean_methods.csv')

nbells = np.max(raw_data["Bell No"])

def find_all_rows(raw_data):
    #Hopefully this is in a decent enough format
    nrows = int(len(raw_data)/nbells)
    allrows = np.zeros((nrows, nbells))

    for ri in range(nrows):
        allrows[ri] = raw_data["Bell No"][ri*nbells:(ri+1)*nbells]
    
    return allrows.astype('int')

def find_method_time(all_rows):
    #Finds the longest time away from rounds, outputs start and finish
    change = False
    rounds_time = 0
    notrounds_time = 0
    current_best = 0; current_start = 0
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
    return start, end + 1

all_rows = find_all_rows(raw_data)
start_row, end_row = find_method_time(all_rows)

trimmed_rows = all_rows[start_row:end_row+1]   #Includes all the changes we care about, rounds EITHER END inclusive

#Find method types. Codes: P, T, S, X. Plain hunt, treble bob, stedman, respectively. X for can't figure it out

def find_method_types(trimmed_rows):
    #Attempts to determine treble type. Want to be able to do spliced Plain/Little etc. ideally but that may be tricky.
    def treble_position(trimmed_rows):
        positions = np.zeros(len(trimmed_rows))
        for ri, row in enumerate(trimmed_rows):
            positions[ri] = int(np.where(row == 1)[0][0])
        return positions
    #print(treble_position(trimmed_rows))

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

    def determine_hunt_types(trimmed_rows):
        #Compares the treble positions against the hunt and bob paths and returns the most likely. Hopefully should be obvious...
        start_index = 0
        go = True
        treble_data = []
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
    
    def determine_methods(trimmed_rows, hunt_types):
        #Now have the number of leads and all the rows.
        #Need to generate place notation and compare against those in the database, for each lead.
        #Can be sped up by assuming it isn't spliced and then checking the rest

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

        if  all(item == hunt_types[0] for item in hunt_types):
            #This could theoretically be a single method, so try and look for one
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
            if type[0] == 'T':
                current_end = current_start + (type[1] + 1)*4 + 1
                lead_length =  (type[1] + 1)*4
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


            print('Overall', li, round(bestmatch*100, 2), '%  match', possible_methods.iloc[pbest]['Name'])
                
            
        #Look for spliced leads (may as well always do this)
        current_start = 0
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
            print('Lead', li, round(bestmatch*100, 2), '%  match', possible_methods.iloc[pbest]['Name'], place_notation)

    hunt_types = determine_hunt_types(trimmed_rows)

    #At this point check for Stedman or Grandsire
    print(len(hunt_types), 'leads found', hunt_types)

    methods = determine_methods(trimmed_rows, hunt_types)

find_method_types(trimmed_rows)