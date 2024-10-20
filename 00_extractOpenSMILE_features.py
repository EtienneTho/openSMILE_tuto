'''
Copyright (c) Etienne Thoret
All rights reserved

'''

import os
import pickle


folder_path = "./wav/"
filenames = []
for filename in os.listdir(folder_path):
    # Check if the path is a file (not a directory)
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        # Print the filename
        print(f"File: {filename}")
        filenames.append(filename)

        # descriptors
        # command = "./opensmile-master/build/progsrc/smilextract/SMILExtract -C ./opensmile-master/config/demo/demo1_breathy.conf -I ./wav/" + filename + " -O ./out_csv/" + filename + ".csv"
        command = "/Users/etienne/Desktop/SleepPaper/openSMILE_tuto/opensmile-master/build/progsrc/smilextract/SMILExtract -C /Users/etienne/Desktop/SleepPaper/openSMILE_tuto/opensmile-master/config/demo/demo1_breathy.conf -I ./wav/" + filename + " -O /Users/etienne/Desktop/SleepPaper/openSMILE_tuto/out_csv/" + filename + ".csv"

        os.system(command)