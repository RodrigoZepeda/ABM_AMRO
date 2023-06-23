import numpy as np
import pandas as pd
import pickle
import os
import sys
import time
from scipy.special import erfc

#Add parent directory functions to path
#https://stackoverflow.com/a/36623762/5067372
p = os.path.abspath('..')
if p not in sys.path:
    sys.path.append(p)

#Load progress patients function
from ABM_Identifiability_python.progress_patients_uniform import progress_patients_uniform

num_ens = 10
p_colonization = 0.05
days_in_timestep = 14
num_space = 3
alpha = 1.5 / 365
rho = 0.0160

gammas = np.linspace(0.06, 0.03, num_space)
betas = np.linspace(0, 0.03, num_space)

gammas, betas = np.meshgrid(gammas, betas)
gammas = gammas.flatten()
betas = betas.flatten()

vars_bank = pd.DataFrame(
    {
        "Beta": betas,
        "Gamma": gammas,
        "Alpha": alpha,
        "Rho": rho,
    }
)

with open(os.path.join("data", "NY_network_1.pickle"), "rb") as file:
    network = pickle.load(file)

days = network["Days"]
n_patients = network["Number of patients"]
n_days = network["Number of days"]
day0 = network["Date before start"]

daily_positive = network["Daily positive count"]
daily_positive['cumsum'] = daily_positive['positives'].cumsum()
r_pos = np.diff(daily_positive['cumsum'][::days_in_timestep])

ts = range(1, n_days + 1, days_in_timestep)
num_times = len(ts) - 1

scan = pd.DataFrame(columns=['Beta', 'Gamma', 'Alpha', 'Rho', 'positives', 'like'])

for v in range(len(scan), num_space):
    variable_extraction = vars_bank.loc[[v]].reset_index()

    #Mysterious if
    #if abs(variable_extraction['Beta'] + variable_extraction['Gamma'] - 0.058) > 0.005:
    #    num_ens = 100
    #else:
    #    num_ens = 300

    p_status = np.random.binomial(n=1, p=p_colonization, size=(n_patients, num_ens))
    positives = np.empty((num_times, num_ens))

    start_time = time.time()
    for t in range(0, num_times):
        p_status, positives[t, :], _, _ = progress_patients_uniform(days[ts[t]:ts[t + 1]], p_status,
                                                                    variable_extraction)
        elapsed_time = time.time() - start_time
        print(f't = {t} - {elapsed_time / 60: .2f} mins | v =  {v} ... {np.round(99 * (v + 1) / num_space)}%')

    scan.loc[v, 'Beta'] = variable_extraction['Beta'][0]
    scan.loc[v, 'Gamma'] = variable_extraction['Gamma'][0]
    scan.loc[v, 'Alpha'] = variable_extraction['Alpha'][0]
    scan.loc[v, 'Rho'] = variable_extraction['Rho'][0]
    scan.loc[v, 'positives'] = positives

    mu = np.mean(positives, axis=1)
    sig = np.std(positives, axis=1)
    like = (r_pos - mu + 0.5) / sig / np.sqrt(2)
    like = -0.5 * (erfc(like) - erfc(like - 1 / sig / np.sqrt(2)))
    like[like == 0] = np.min(like[like > 0])
    like = np.sum(np.log(like))

    scan.loc[v, 'like'] = like

    with open('Scan_narrow_3.pickle', 'wb') as file:
        pickle.dump(scan, file)

    print(f"scan {v + 1} of {num_space}")
