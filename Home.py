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

#Comment so it detects a commit
import streamlit as st
import os
import pandas as pd
import numpy as np
from streamlit.logger import get_logger
from listen_other_functions import find_current_stats
import re


LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Home",
        page_icon="🔔",
    )
    

    st.markdown('''
                ## BReNDA 0.9.8 
                ### Bell REcording with Novel Data Analysis 
                ''')

    st.markdown("This is a new app used to analyse church bell ringing in the English tradition. You can upload a recording of some bells and if all goes well it'll tell you what is being rung and (precisely) how well.")
    st.markdown("In the principle of ringing being 'free at the point of use' I'm not going to charge for Brenda, but servers aren't free and I've put a lot of time into this, so if you feel like donating to the cause it would be greatly appreciated. If so please use [this link](%s) (or alternatively just buy me a pint or three at some point).  Check it works for your tower first though!" % 'https://donate.stripe.com/9B69ATgt4520fex6XZ7ok00') 
    st.markdown("All non-streamlit backend code © 2025 Oliver Rice. All rights reserved.")

    #Add thing here to track number of towers analysed and touches
    ntowers, ntouches, tower_list = find_current_stats()
    st.write('Brenda has currently learnt the bells at **%d** towers and has **%d** touches saved.' % (ntowers, ntouches))
    st.write('#### How to use it')
    st.markdown('''
        There are three 'pages' which can be accessed with the below links or the sidebar on the left (click the right-facing arrow top left if you can't see it):
                ''')
                
    st.page_link("pages/1_Analyse_Recording.py", label = ":blue[Analyse a Recording]")
    st.markdown('''
        Use this page to upload (or record) a recording of bellringing, 'learn' the frequencies of the bells if necessary and determine the strike times. This data can then be viewed on:
                ''')
    st.page_link("pages/2_View_Touches.py", label = ":blue[View Touches]")
    st.markdown('''
        Which will take this data and display things like the blue line, a text 'striking report' and bar charts of errors. 
                ''')
    st.page_link("pages/3_Touch_Library.py", label = ":blue[Touch Library]")
    st.markdown('''
        This page doesn't do any analysis but allow you to save touches in 'collections' so they can be used later and shared easily.
                ''')

    with st.expander("What this app does"):
        st.markdown('''
                        This web app contains tools used to analyse the quality of church bellringing in the English tradition.
            This process is split into two distinct parts: \\
                        1. Analysing a raw recording to determine the times at which the bells strike. \\
                        2. Displaying this data in a way which is meaningful. \\
                    \\
            There already exists software to accomplish this, namely (respectively) HawkEar and Strikeometer. 
                        This new app is designed quite differently from these and is more focused on less technical analysis than the high-quality 12 bell touches people usually focus on with this kind of software. \\
                        Notably, the new method for audio analysis can use (reasonably) noisy ringing chamber recordings, and does not require any calibration beforehand. It achieves using new audio filtering techniques which I've figured out expressly for this purpose.\\
                    \\
                        This is not a replacement for or an update to the existing software -- with the same input HawkEar will generally provide more reliable results (by a few ms) -- but this does open up this kind of analysis to more everday ringing, and the feedback can hopefully be used to help people improve in scenarios other than just the very best 12 bell ringing.
            ''')
    with st.expander("Limitations (not a complete list...)"):
        st.markdown('''
            As you might expect, the recording must be reasonably clear -- if you can't pick out all the bells most of the time then this probably won't manage it. Some towers just work better than others. Anything which works for HawkEar *should* be OK here though, and please let me know if you find something that doesn't (apart from St. Paul's Cathedral, I know it doesn't like it there).\\
            $~$ \\
            The algorithm is designed to look for rhythmic, open handstroke ringing. Analysing bad ringing can be quite amusing but once the rhythm is lost it's difficult for it to figure out and sometimes it'll go terribly wrong. \\
            $~$ \\
            Currently the algorithms for finding the 'ideal' striking times are my own - the 'team model' seems to be reasonably consistent with the current 'contest model' used in Strikeometer, but there are some notable differences. I have added the RWP striking model for direct comparisons if neccessary"
                    
                    ''')
    with st.expander("Recent Updates"):
        st.markdown('''
            **Update (0.9.8):** \\
            Updates to the backend maths, including logarithmic frequency binning and the use of a hamming filter to improve the accuracy of the Fourier Transforms. Now within 2ms of HawkEar's quality for the very best ringing. Also update to streamlit etc. to make things prettier and some changes to the tolerance for bad ringing.
            ''')

        st.markdown('''
            **Update (0.9.7):** \\
            Made shareable links more robust such that they now work on the homepage (eg. "brenda.oekrice.com?collection=example_touches"). Added more features to the touch library to allow for editing the time and tower, and added the quality of the ringing to the list of touches on this page. Fixed duplicate tower name problem (there are 30 pairs of towers with the same name on Dove's guide, incidentally).
            ''')
        st.markdown('''
            **Update (0.9.6):** \\
            Added more checks to ensure the recording analysis is thread-safe (maybe fixed Samsung problem I hope perhaps?), changed some of the syntax to make using method collections easier and introduced shareable links.
            ''')
        st.markdown('''
            **Update (0.9.5):** \\
            Added the 'Touch Library', so touches can be saved and shared within the app. Preserves metadata etc. so things should be able to be organised nicely.
            Also set up a sever on Oracle Cloud (still testing properly) so the limits on upload sizes have been lifted. 
            ''')
        st.markdown('''
                **Update (0.9.4):** \\
                Added several new frontend features:
                * In-browser recording added (a popular request)
                * Added RWP striking model as an option on the View Touches page
                * Added a plain text striking report for each touch, identifying things which can be improved for each bell. Works using confidence intervals so can identify plenty which isn't obvious from the graphs themselves
                * Made some minor changes to the timing identification to try to improve the accuracy. Now usually within 5ms of HawkEar, from what I've tested.
                ''')
        st.markdown('''
                    **Update (0.9.3):** \\
                    Added a new module to retroactively find places where the change becomes ill-defined, such as particularly choppy ringing or firing out and restarting.
                    Also automated more of the frequency reinforcement, and fixed some bugs with the method detection when things fire out.
                    ''')
    with st.expander("Contact"):
        st.markdown('''
            If you have any questions, would like to suggest any new features or find any bugs/any error messages, please let me know at "oliverricesolar(at)gmail.com", where clearly the (at) should be @ if you're not a robot.
            I am not by any means a software developer so I appreciate it's all a little rough and ready.
            ''')
        st.markdown('''
            If you'd like access to the code to make something more of this, or collaborate to achieve something more, then don't hesitate to get in touch. I'll have less time to work on this before long so I'm keen to see it progress with less input from me.
            ''')
        
    with st.expander("View towers Brenda's currently learnt:"):

        st.markdown('\n'.join(f"- {str(tower)}" for tower in tower_list))

    #Allow adding to the cache from the homepage, for completeness
    if 'cached_touch_id' not in st.session_state:
        st.session_state.cached_touch_id = []
    if 'cached_read_id' not in st.session_state:
        st.session_state.cached_read_id = []
    if 'cached_nchanges' not in st.session_state:
        st.session_state.cached_nchanges = []
    if 'cached_methods' not in st.session_state:
        st.session_state.cached_methods = []
    if 'cached_tower' not in st.session_state:
        st.session_state.cached_tower = []
    if 'cached_datetime' not in st.session_state:
        st.session_state.cached_datetime = []
    if 'cached_score' not in st.session_state:
        st.session_state.cached_score = []
    if 'cached_ms' not in st.session_state:
        st.session_state.cached_ms = []
        
    if 'cached_strikes' not in st.session_state:
        st.session_state.cached_strikes = [] 
    if 'cached_certs' not in st.session_state:
        st.session_state.cached_certs = [] 
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = []   
    if 'cached_rawdata' not in st.session_state:
        st.session_state.cached_rawdata = []  

    def determine_collection_from_url(existing_names):
        if 'collection' in st.query_params.keys():
            collection_name = st.query_params["collection"]
            collection_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), collection_name)   
            if collection_name in existing_names:
                return collection_name
            else:
                return None
        else:
            return None
        
    def find_existing_names():
        #Finds a list of existing collection names
        names_raw = os.listdir('./saved_touches/')
        names_lower = []
        for name in names_raw:
            lower_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), name)   
            names_lower.append(lower_name)
        return names_lower

    def add_collection_to_cache(ntouches, saved_index_list):
        if ntouches!= 0 and saved_index_list[0][0] != ' ':
            #Put everything from this collection into the cache (if it's not already there)
            for touch_info in saved_index_list:
                #Check if it's already in there
                if touch_info[0] not in st.session_state.cached_touch_id:
                    #Load in csv
                    raw_data = pd.read_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,touch_info[0]))

                    st.session_state.cached_data.append([touch_info[4], int(len(raw_data)/np.max(raw_data["Bell No"])), touch_info[1]])
                    st.session_state.cached_strikes.append([])
                    st.session_state.cached_certs.append([])
                    st.session_state.cached_rawdata.append(raw_data)

                    st.session_state.cached_touch_id.append(touch_info[0])
                    st.session_state.cached_read_id.append(touch_info[1])
                    st.session_state.cached_nchanges.append(touch_info[2])
                    st.session_state.cached_methods.append(touch_info[3])
                    st.session_state.cached_tower.append(touch_info[4])
                    st.session_state.cached_datetime.append(touch_info[5])
                    st.session_state.cached_score.append(touch_info[6])
                    st.session_state.cached_ms.append(touch_info[7])
                else:
                    cache_index = st.session_state.cached_touch_id.index(touch_info[0])
                    raw_data = pd.read_csv("./saved_touches/%s/%s.csv" % (st.session_state.current_collection_name,touch_info[0]))

                    st.session_state.cached_data[cache_index] = [touch_info[4], int(len(raw_data)/np.max(raw_data["Bell No"])), touch_info[1]]
                    st.session_state.cached_strikes[cache_index] = []
                    st.session_state.cached_certs[cache_index] = []
                    st.session_state.cached_rawdata[cache_index] = raw_data

                    st.session_state.cached_touch_id[cache_index] = touch_info[0]
                    st.session_state.cached_read_id[cache_index] = touch_info[1]
                    st.session_state.cached_nchanges[cache_index] = touch_info[2]
                    st.session_state.cached_methods[cache_index] = touch_info[3]
                    st.session_state.cached_tower[cache_index] = touch_info[4]
                    st.session_state.cached_datetime[cache_index] = touch_info[5]
                    st.session_state.cached_score[cache_index] = touch_info[6]
                    st.session_state.cached_ms[cache_index] = touch_info[7]
        
                  
    existing_names = find_existing_names()
    url_collection = determine_collection_from_url(existing_names)

    def determine_url_params():
        if 'current_collection_name' in st.session_state:
            if st.session_state.current_collection_name is not None:
                st.query_params.from_dict({"collection": [st.session_state.current_collection_name]})
        return
    determine_url_params()

    if url_collection is not None:
        st.session_state.collection_status = 0
        st.session_state.current_collection_name = url_collection

        if os.path.getsize("./saved_touches/%s/index.csv" % st.session_state.current_collection_name) > 0:
            saved_index_list = np.loadtxt("./saved_touches/%s/index.csv" % st.session_state.current_collection_name, delimiter = ';', dtype = str)
        else:
            saved_index_list = np.array([[' ',' ',' ',' ',' ',' ',' ',' ']], dtype = 'str')
        if len(np.shape(saved_index_list)) == 1:
            saved_index_list = np.array([saved_index_list])

        if len(saved_index_list[0]) < 8: #Add extra spaces
            saved_index_list = saved_index_list.tolist()
            nshort = 8 - len(saved_index_list[0])
            for li, item in enumerate(saved_index_list):
                for i in range(nshort):
                    item.append(' ')
                saved_index_list[li] = item
                
        saved_index_list = np.array(saved_index_list)

        ntouches = len(saved_index_list)

        add_collection_to_cache(ntouches, saved_index_list)


if __name__ == "__main__":
    run()
