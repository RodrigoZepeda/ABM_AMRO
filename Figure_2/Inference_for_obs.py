import numpy as np
import os
import pickle
import pandas as pd

# from Inference_iEAKF import Inference_iEAKF

"""
To generate the hyperparameters and associated files for the time being you need
to run the `DataTransform.m` first as there is no reasonable way to read the structures
you are using in Matlab to Python.

I'm using Google's style guide for Python: https://google.github.io/styleguide/pyguide.html
"""

# Set seed
np.random.seed(seed=2487523)

# Hyperparameter setup.
with open(os.path.join("data", "NY_Network_1.pickle"), "rb") as file:
    Network = pickle.load(file)

hyparams = {
    # No.Iter in Matlab
    "Number of iterations": 20,
    # No.ens in Matlab
    "Number of ensemble members": 200,
    # No.tmstep in Matlab
    "Number of days in a timestep": 2 * 7,  # The timestep is tmstep + 1
    # No.IntP in Matlab
    "Probability of colonized upon admission": 0.05,
    # No.Days in Matlab
    "Number of days": Network["Number of days"],
    # No.Pat in Matlab
    "Number of patients": Network["Number of patients"],
    # No.ward in Matlab
    "Number of wards": Network["Number of wards"],
    # No.day0 in Matlab
    "Date before start": Network["Date before start"],
    # Maximum number of inf_res to generate #FIXME: @Tal add more information
    "Trajectory number": 3,
    # Desired variance lower and upper parameters for 1/linspace
    "Std linspace range": [2.0, 20.0],
}

# Positive individuals in network
daily_positive = Network["Daily positive count"]
daily_positive['cumsum'] = daily_positive['positives'].cumsum()
r_pos = np.diff(daily_positive['cumsum'][::hyparams["Number of days in a timestep"]])

# Parameters range [lower, upper]
variable_range = pd.DataFrame({
    'Beta': [0, 0.1],  # Baseline transmission rate, per day
    'Gamma': [0.0001, 0.1],  # Importation rate, per admission
    'Alpha': [1.5 / 365, 1.5 / 365],  # Patient decolonization rate, per day
    'Rho': [0.0160, 0.0160],  # observation rate, per day
})

hyparams["Number of variables"] = variable_range.shape[1]

# Desired standard deviation (Wantstd in Matlab)
wanted_std = [1.0 / np.linspace(start=hyparams["Std linspace range"][0],
                                stop=hyparams["Std linspace range"][1],
                                num=hyparams["Number of iterations"],
                                dtype=float)]
diffs = np.diff(variable_range, axis=0)
wanted_std = np.transpose(wanted_std) * diffs

for traj_num in range(hyparams["Trajectory number"]):
    Dits = Inference_iEAKF(variable_range, hyparams, wanted_std, Network["Days"], r_pos)
    # Save Dits
