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

def get_interior(place_notation):
    #Gets rid of the lead end and just flips the place notation backwards. All these methods are palindromic so that's fine
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

    if props.find('mx:numberOfHunts', ns).text != '1':
        continue

    methods = method_set.findall('mx:method', ns)

    for method in methods:
        if method.find('mx:symmetry', ns) is None:
            continue
        if not method.find('mx:symmetry', ns).text == 'palindromic':
            continue
        if method.find('mx:title', ns) is None:
            continue
        if method.find('mx:notation', ns) is None:
            continue

        interior_notation = get_interior(method.find('mx:notation', ns).text)
        method_data.append({'Name': method.find('mx:title', ns).text, 'Stage': stage, 'Lead Length': lead_length, 'Place Notation': method.find('mx:notation', ns).text, 'Type': treble_type, 'Interior Notation': interior_notation})

df = pd.DataFrame(method_data)
df.to_csv('method_data/clean_methods.csv', index=False)