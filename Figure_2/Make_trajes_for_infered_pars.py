import numpy as np
import pandas as pd
import scipy.io
import pickle
import os
from progress_patients_uniform import progress_patients_uniform

# Load data
with open(os.path.join("Figure_2","Inf_res_1.pickle"), "rb") as file:
    data = pickle.load(file)

# Retrieve required variables
time_step = data["Number of days in a timestep"]
p_colonization = data["Probability of colonized upon admission"]

mean_beta = np.mean(data['betas'][:, -1])
mean_gamma = np.mean(data['gammas'][:, -1])

num_ens = 10

# Load network
with open(os.path.join("data","NY_Network_1.pickle"), "rb") as file:
    network = pickle.load(file)

days = network['Days']
n_patients = network["Number of patients"]
n_days = network["Number of days"]
initial_day = data["Date before start"]

daily_positive = network["Daily positive count"]
daily_positive['cumsum'] = daily_positive['positives'].cumsum()
r_pos = np.diff(daily_positive['cumsum'][::time_step])

# Parameters values
variable_values = pd.DataFrame({
    'Beta': [mean_beta],
    'Gamma': [mean_gamma],
    'Alpha': [1.5 / 365],
    'Rho': [0.0160]
})

#Timesteps
ts = np.arange(1, n_days, time_step)
num_times = len(ts) - 1

p_status = np.random.binomial(n=1, p=p_colonization, size=(n_patients, num_ens))

positives = np.zeros((num_times, num_ens))

for t in range(num_times):
    _, positives[t], _, _ = progress_patients_uniform(days=days[ts[t]:ts[t + 1]], p_pre=p_status, parameters=variable_values)

trajectories = {
    'Positives': positives,
    'Variables': variable_values
}

# Save trajectory data
with open(os.path.join("Figure_2", "Trajs.pickle"), "wb") as file:
    pickle.dump(trajectories, file)

