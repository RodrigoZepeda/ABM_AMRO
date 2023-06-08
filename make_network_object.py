import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import re

"""
This file aims to reproduce the `Network` inside `NY_Network_1.mat` file
You must first run the DataTransform.m file in Matlab to obtain the
applicable files in `data_to_py`.

Structure of Network is as follows:

- Days [Table with tables as entries]
    - patients (dataframe 3 columns)
        - mrn
        - ward
        - weight
    - wards (dataframe)
        - ward
        - count
        - size
    - Pfirst (1 dimensional array)
    - wfirst (1 dimensional array)
    - activated (1 dimensional array)
- Nward (double)
- NPat (dounle)
- NDays (double)
- WardSize (string)
- day0 (datetime)
- Daypos (dataframe)
    - day
    - positives
- wardpos (dataframe)
    - day
    - ward
    - positives
"""

# Construct everything from the network except Days
# ---------------------------------------------------------------------------------------

# Read start date (day0)
date_start = pd.read_table(os.path.join("data_to_py", "NY_Network_1_day0.txt"), header=None)[0][0]

# Construct part of the network dictionary
Network = {
    # Network.Days in Matlab
    "Number of days": pd.read_table(os.path.join("data_to_py", "NY_Network_1_NDays.txt"), header=None, dtype=int)[0][0],
    # Network.NPat in Matlab
    "Number of patients": pd.read_table(os.path.join("data_to_py", "NY_Network_1_NPat.txt"), header=None, dtype=int)[0][0],
    # Network.NWard in Matlab
    "Number of wards": pd.read_table(os.path.join("data_to_py", "NY_Network_1_NWard.txt"), header=None, dtype=int)[0][0],
    # Network.WardSize in Matlab
    "Ward size": pd.read_table(os.path.join("data_to_py", "NY_Network_1_WardSize.txt"), header=None)[0][0],
    # No.day0 in Matlab
    "Date before start": datetime.strptime(date_start, '%d-%b-%Y'),
    # DayPos in Matlab
    "Daily positive count": pd.read_csv(os.path.join("data_to_py", "NY_Network_1_daypos.csv"), dtype=int),
    # WardPos in Matlab
    "Ward positive count": pd.read_csv(os.path.join("data_to_py", "NY_Network_1_wardpos.csv"), dtype=int),
}

# Construct the Days field of the network as a
# ---------------------------------------------------------------------------------------

# Get all the files that came form a Matlab Days object via DataTransform
dir_list = os.listdir(path="data_to_py")
words_to_check = ["_patients_", "_wards_", "_Pfirst_", "_Wfirst_", "_activated_"]
matching_files = [file for file in dir_list if any(word in file for word in words_to_check)]

# Get the day range in the generated files
pattern = r'NY_Network_1_.*_(\d+)'
matching_numbers = []
for file in dir_list:
    match = re.match(pattern, file)
    if match:
        matching_numbers.append(int(match.group(1)))

# Loop through all the files
Days = np.empty(0, dtype=object)
for day in range(np.amin(matching_numbers), np.amax(matching_numbers) + 1):

    print("Day: " + str(day))

    # Read the tables
    patients = pd.read_csv(os.path.join("data_to_py", "NY_Network_1_patients_" + str(day) + ".csv"))
    wards = pd.read_csv(os.path.join("data_to_py", "NY_Network_1_wards_" + str(day) + ".csv"))
    pfirst = np.loadtxt(os.path.join("data_to_py", "NY_Network_1_Pfirst_" + str(day) + ".csv"), dtype=int)

    # The wfirst files doesn't exist if array in matlab was empty so we check existance
    wfirst_file = os.path.join("data_to_py", "NY_Network_1_Wfirst_" + str(day) + ".txt")
    if os.path.isfile(wfirst_file):
        wfirst = np.loadtxt(wfirst_file, dtype="int")
    else:
        wfirst = np.empty(0)

    # The activated files doesn't exist if array in matlab was empty so we check existance
    activated_file = os.path.join("data_to_py", "NY_Network_1_activated_" + str(day) + ".csv")
    if os.path.isfile(activated_file):
        activated = np.loadtxt(activated_file, dtype="int")
    else:
        activated = np.empty(0)

    # We now add them to the network object for the day
    Days = np.append(Days, [{
        "Patients": patients,
        "Wards": wards,
        "Pfirst": pfirst,
        "Wfirst": wfirst,
        "Activated": activated,
    }])

# Add days to network
Network["Days"] = Days

# Save everything
with open(os.path.join("data", "NY_Network_1.pickle"), "wb") as file:
    pickle.dump(Network, file)

# Let user know
print("Process finished")
