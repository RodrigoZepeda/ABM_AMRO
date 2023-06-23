import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erfc
import pickle
import os
import sys
import time

#Add parent directory functions to path
#https://stackoverflow.com/a/36623762/5067372
p = os.path.abspath('..')
if p not in sys.path:
    sys.path.append(p)

#Load progress patients function
from ABM_Identifiability_python.progress_patients_uniform import progress_patients_uniform

np.random.seed(73205)

p_colonization = 0.05
days_in_timestep = 14
b = 0.0598
a = -1.03
alpha = 1.5 / 365
rho = 0.0160
num_ens = 10
num_space = 3
betas = np.linspace(0, b + a / 10000, num_space)
gammas = (betas - b) / a


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

    #Simulate patient status
    p_status = np.random.binomial(n=1, p=p_colonization, size=(n_patients, num_ens))
    positives = np.empty((num_times, num_ens))

    start_time = time.time()
    for t in range(0, num_times):
        p_status, positives[t, :], _, _ = progress_patients_uniform(days[ts[t]:ts[t + 1]], p_status,
                                                                    variable_extraction)
        elapsed_time = time.time() - start_time
        print(f't = {t} - {elapsed_time / 60: .2f} mins | v =  {v} ... {np.round(99*(v + 1)/num_space)}%')

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

    with open(os.path.join("Figure_3",'Scan_line_4.pickle'), 'wb') as file:
        pickle.dump(scan, file)

    sns.scatterplot(x='Gamma', y='like', data=scan, color='blue', marker='*')

p0 = sns.scatterplot(x='Gamma', y='like', data=scan, color='blue')
p1 = sns.scatterplot(x='Beta', y='like', data=scan, color = "red")

scan['fit'] = np.nan

for j in range(len(scan)):
    temp = np.transpose(np.tile(r_pos, (scan['positives'][j].shape[1], 1)))
    temp = np.abs((scan['positives'][j] - temp)) / temp
    temp = np.mean(temp, axis=1)
    scan.loc[j, 'fit'] = np.mean(temp)

p2 = sns.scatterplot(x='Gamma', y='fit', data=scan, color = "green")
p2.figure.savefig(os.path.join("Figure_3", "output.pdf"))
plt.close(p2.figure)

scan['like05'] = np.nan
scan['like95'] = np.nan
scan['like_med'] = np.nan
scan['like_prntile'] = np.nan

for j in range(len(scan)):
    positives = scan['positives'][j]
    mu = np.mean(positives, axis=1)
    sig = np.std(positives, axis=1)
    temp = np.ones(positives.shape[1])
    like = (positives - mu[:, None] * temp + 0.5) / (sig[:, None] * temp) / np.sqrt(2)
    like = -0.5 * (erfc(like) - erfc(like - 1 / (sig[:, None] * temp) / np.sqrt(2)))
    like = np.sum(np.log(like), axis = 0)
    scan.loc[j, 'like05'] = np.percentile(like, 5)
    scan.loc[j, 'like95'] = np.percentile(like, 95)
    scan.loc[j, 'like_med'] = np.percentile(like, 50)
    scan.loc[j, 'like_prntile'] = np.sum(like < scan.loc[j, 'like']) / len(like)

plt.figure()
sns.lineplot(x='Gamma', y='like_med', data=scan, color='red')
plt.errorbar(scan['Gamma'], scan['like_med'],
             yerr=(scan['like_med'] - scan['like05'], scan['like95'] - scan['like_med']), color='red')
plt.plot(scan['Gamma'], scan['like'], color='blue')
plt.xlabel(r'$\gamma$')
plt.ylabel('log-likelihood')
plt.title('log-likelihood along ridge')
plt.savefig(os.path.join("Figure_3", "fig_3_scan_across_line_1.pdf"))
plt.close()

plt.figure()
sns.lineplot(x='Gamma', y='like_prntile', data=scan, color='blue')
plt.xlabel(r'$\gamma$')
plt.ylabel('likelihood percentile')
plt.title('log-likelihood along ridge')

# Adjust layout and save figure
plt.savefig(os.path.join("Figure_3", "fig_3_scan_across_line_2.pdf"))
plt.close()

