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

#Script to import method data from the xmls and save in a csv format which makes sense to me
from bs4 import BeautifulSoup
import xml.etree.ElementTree as et
import csv
import pandas as pd
import time
tree = et.parse('method_data/CCCBR_methods.xml')

method_data = []
#Namespace thing? Not sure what this does
ns = {'mx': 'http://www.cccbr.org.uk/methods/schemas/2007/05/methods'}

root = tree.getroot()

method_sets = root.findall('.//mx:methodSet', ns)

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

def add_stedmans(method_data):
    #Doubles first. This looks a bit funny but I'm pretty sure is the best way. 
    lead_length = 6
    treble_type = "S"
    nhunts = 0
    stage = 5
    name = "Stedman Doubles Slow"
    place_notation = '3.1.5.3.1,3'
    interior_notation = '3.1.5.3.1.3'
    method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
    name = "Stedman Doubles Quick"
    place_notation =    '1.3.5.1.3,1'
    interior_notation = '1.3.5.1.3.1'
    method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
    name = "Stedman Doubles"
    place_notation = '3.1.5.3.1.3.1.3.5.1.3,1'
    interior_notation = '3.1.5.3.1.3.1.3.5.1.3.1'
    method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length*2, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
    titles = ["Triples", "Caters", "Cinques", "Thirteen", "Fifteen"]
    stages = [7,9,11,13,15]
    for si, stage in enumerate(stages):
        title = titles[si]
        name = "Stedman " + title + " Quick"
        place_notation = '1.3.1.3.1,' + tostring(stage-1)
        interior_notation = '1.3.1.3.1.' + tostring(stage-1)
        method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
        name = "Stedman " + title + " Slow"
        place_notation = '3.1.3.1.3,' + tostring(stage-1)
        interior_notation = '3.1.3.1.3.' + tostring(stage-1)
        method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
        name = "Stedman " + title
        place_notation = '3.1.3.1.3.' + tostring(stage-1) + '.1.3.1.3.1,'+ tostring(stage-1)
        interior_notation = '3.1.3.1.3.' + tostring(stage-1) + '.1.3.1.3.1.'+ tostring(stage-1)
        method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length*2, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
        name = "Erin " + title
        place_notation = '3.1.3.1.3.' + tostring(stage-1) + '.3.1.3.1.3,'+ tostring(stage-1)
        interior_notation = '3.1.3.1.3.' + tostring(stage-1) + '.3.1.3.1.3.'+ tostring(stage-1)
        method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length*2, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
        name = "Erin " + title + " Slow"
        place_notation = '3.1.3.1.3,' + tostring(stage-1)
        interior_notation = '3.1.3.1.3.' + tostring(stage-1)
        method_data.append({'Name': name, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': place_notation, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})
    
    return method_data

def get_interior(place_notation):
    #Gets rid of the lead end and just flips the place notation backwards. All these methods are palindromic so that's fine
    split = place_notation.rsplit(',', 1)
    if len(split[0]) > len(split[-1]):
        #This is single-hunt
        firstbit = place_notation.rsplit(',', 1)[0]
        last_dash = place_notation.rfind('-')
        last_dot = place_notation.rfind('.')
        cut_index = max(last_dash, last_dot) 
        toflip = place_notation[:cut_index+1] 
        flipped = ''
        start = 0
        while start < len(toflip):
            next_dot = toflip[start:].find('.')
            next_dash = toflip[start:].find('-')
            if next_dot < 0:
                next_dot = 10000000
            if next_dash < 0:
                next_dash = 10000000
            if next_dot == 0:
                flipped = '.' + flipped
                start += 1
            elif next_dash == 0:
                flipped = '-' + flipped
                start += 1
            else:
                flipped = toflip[start:start+min(next_dot, next_dash)] + flipped
                start += min(next_dot, next_dash)


        complete = firstbit + flipped
        leadend = place_notation.rsplit(',', 1)[1]

        if complete[-1] == '-':
            return complete + leadend
        else:
            return complete + '.' + leadend
    else:
        #This is double-hunt
        firstbit = place_notation.rsplit(',', 1)[-1]
        last_dash = place_notation.rfind('-')
        last_dot = place_notation.rfind('.')
        cut_index = max(last_dash, last_dot) 
        toflip = firstbit[:cut_index+1] 
        flipped = ''
        start = 0
        while start < toflip.rfind(".") + 1:
            next_dot = toflip[start:].find('.')
            next_dash = toflip[start:].find('-')
            if next_dot < 0:
                next_dot = 10000000
            if next_dash < 0:
                next_dash = 10000000
            if next_dot == 0:
                flipped = '.' + flipped
                start += 1
            elif next_dash == 0:
                flipped = '-' + flipped
                start += 1
            else:
                flipped = toflip[start:start+min(next_dot, next_dash)] + flipped
                start += min(next_dot, next_dash)

        complete = firstbit + flipped
        leadend = place_notation.rsplit(',', 1)[0]

        if complete[0] == '-':
            return leadend + complete
        else:        
            return leadend + '.' + complete

for method_set in method_sets:
    props = method_set.find('mx:properties', ns)
    stage = props.find('mx:stage', ns).text
    classification = props.find('mx:classification', ns)
    if classification is None:
        continue
    attr = classification.attrib
    lead_length =  props.find('mx:lengthOfLead', ns).text

    if not (attr.get('trebleDodging') or attr.get('plain')):
        continue
    if attr.get('trebleDodging'):
        treble_type = 'T'
    elif attr.get('plain'):
        treble_type = 'P'

    nhunts = props.find('mx:numberOfHunts', ns).text
    if not (nhunts == '1' or nhunts == '2'):
        continue

    if not (nhunts == '1' or nhunts == '2'):
        continue


    def check_leadend(notation, stage, nhunts):
        #Return true if the lead end is a normal one
        if nhunts == '1':
            leadend = notation.rsplit(',', 1)[-1]
            if leadend == '12'or leadend == '1' + tostring(int(stage)-1) or leadend == '12' + tostring(int(stage)-1):
                return True
            else:
                return False

        elif nhunts == '2':
            leadend = notation.rsplit(',', 1)[0]
            if leadend == '3':
                return True
            else:
                return False
        else:
            return False

        
    methods = method_set.findall('mx:method', ns)

    for method in methods:
        if method.find('mx:symmetry', ns) is None:
            continue
        if not method.find('mx:symmetry', ns).text == 'palindromic' and not method.find('mx:symmetry', ns).text == 'palindromic double rotational':
            continue
        if method.find('mx:title', ns) is None:
            continue
        if method.find('mx:notation', ns) is None:
            continue
        #Remove methods which don't have a 12 or 1n lead end, as that can be confusing with bobs
        #print(check_leadend(method.find('mx:notation', ns).text, stage, nhunts))
        if not check_leadend(method.find('mx:notation', ns).text, stage, nhunts):
            continue

        
        interior_notation = get_interior(method.find('mx:notation', ns).text)
        method_data.append({'Name': method.find('mx:title', ns).text, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': method.find('mx:notation', ns).text, 'Type': treble_type, 'Interior Notation': interior_notation, 'Hunt Number': nhunts})

#Add on Stedman things here. As bobs/singles happen at two points best to treat each six separately. Have Stedman Slow Triples, Stedman Quick Caters etc.
#Doubles is special but does actually make more sense as you don't need things to bodge at either end
method_data = add_stedmans(method_data)

df = pd.DataFrame(method_data)
df.to_csv('method_data/clean_methods.csv', index=False, mode = 'w')
print('Method data converted and saved')