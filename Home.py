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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Home",
        page_icon="🔔",
    )

    st.write("## BReNDA 0.9.2")
    st.write('### Bell REcording with Novel Data Analysis')

    st.write('### What this is')
    st.markdown('''
                    This web app contains tools used to analyse the quality of church bellringing in the English tradition.
        There are two main parts to this: \\
                    1. Analysing a raw recording to determine the times at which the bells strike. \\
                    2. Displaying this data in a way which is meaningful. \\
                \\
        There already exists software to accomplish this, namely (respectively) HawkEar and Strikeometer. 
                    This new app is designed completely differently from these and is more focused on less technical analysis than the high-quality 12 bell touches people usually focus on with this kind of software. \\
                    Notably, the new method for audio analysis can use (reasonably) noisy ringing chamber recordings, and does not require any calibration beforehand. \\
                \\
                    This is not a replacement for or an update to the existing software -- with the same input HawkEar will probably provide more reliable results -- but this does open up this kind of analysis to more everday ringing, and the feedback can hopefully be used to help people improve more generally than just in fancy places. 
        ''')
    st.write('### How to use it')
    st.markdown('''
        There are two 'pages' which can be accessed with the below links or the sidebar on the left (a bit clunky on a phone, apologies).
                ''')
    st.page_link("pages/1_Analyse_Recording.py", label = ":blue[Analyse Recording]")
    st.markdown('''
        Use this page to upload (or record) audio, 'learn' the frequencies of the bells and determine the strike times. This data can be saved to the device for later or used directly on:
                ''')
    st.page_link("pages/2_Analyse_Striking.py", label = ":blue[Analyse Striking]")
    st.markdown('''
        Which will take this data and display things like the blue line and bar charts of striking errors similarly to the existing Strikeometer software. This can also be used to analyse HawkEar outputs saved as a .csv file.
                ''')
    st.write('### Limitations (more tbc)')
    st.markdown('''
        As you might expect, the recording must be reasonably clear -- if you can't pick out all the bells most of the time then this probably won't manage it. Some towers just work better than others. Anything which works for HawkEar *should* be OK here though, and please let me know if you find something that doesn't.\\
        The algorithm is designed to look for rhythmic, open handstroke ringing. Analysing bad ringing can be quite amusing but once the rhythm is lost that's very difficult for it to figure out. \\
        This site is very limited by how much memory I can use for free. So please just use this for short touches... If someone fancies paying for a better server, do let me know. \\
        Currently the algorithms for finding the 'ideal' striking times are my own - the 'team model' seems to be reasonably consistent with the current 'contest model' used in Strikeometer, but there are some notable differences. I will integrate the old models into this app at some point.
                
                ''')

if __name__ == "__main__":
    run()
