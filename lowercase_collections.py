#Very simple script to rename existing method collections to be lowercase
import os
import re

#Find collection names
for file in os.listdir('.'):
    if os.path.isdir(file):
        new_name = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), file)
        os.rename(file, new_name)

    #os.path.exists("./saved_touches/%s/index.csv" % st.session_state.current_collection_name)