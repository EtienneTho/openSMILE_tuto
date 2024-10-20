import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
import pickle
from tqdm import tqdm


# Define the path to the subfolder containing CSV files
subfolder_path = "./out_csv/"

nFeatures = 530 # equal to the number of features extracted in the config file
# nTimeFrames = 1496

# Initialize a dictionary to store data from CSV files
csv_data = {}
data2store = np.zeros((nFeatures,)) 
file_names = []
species = []
ids = []
# data2store = []
# List CSV files in the subfolder
csv_files = [os.path.join(subfolder_path, file) for file in os.listdir(subfolder_path) if file.endswith('.csv')]
# print(csv_files)
# Loop through CSV files and read them into Pandas DataFrames
for csv_file in tqdm(csv_files):
    file_name = os.path.splitext(os.path.basename(csv_file))[0]  # Extract the file name without extension
    df = pd.read_csv(csv_file,sep=';')
    csv_data[file_name] = df
    toAdd = np.mean(np.asarray(df.iloc[:,:]),axis=0)
    data2store = np.vstack([np.array(data2store), np.array(toAdd)])
    filename_splited = file_name.split("_")
    file_names.append(file_name)
    species.append(filename_splited[0])
    ids.append(filename_splited[2])

data2store = data2store[1:,:]

pickle.dump({'dataOpenSMILE': np.asarray(data2store), 'species': np.asarray(species), 'colNames': np.asarray(df.columns), 'filenames':np.asarray(file_names), 'ids':ids}, open('./features_OpenSMILE.pkl', 'wb'))