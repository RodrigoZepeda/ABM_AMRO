import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os
import datetime

# Clear all existing figures
plt.close("all")

# Load data
with open(os.path.join("Figure_2", "Inf_res_1.pickle"), "rb") as file:
    data = pickle.load(file)

# Load data
with open(os.path.join("Figure_2", "Trajs.pickle"), "rb") as file:
    trajectories = pickle.load(file)

colors = sns.color_palette(n_colors=6)

# Retrieve required variables
n_iterations = data["Number of iterations"]
number_of_times = data["Number of times"]
variable_range = data["variable_range"]
initial_day = data["Date before start"]
time_step = data["Number of days in a timestep"]

# Create trajectories variables
poss = trajectories["Positives"]
p_95 = np.percentile(trajectories["Positives"], 95, axis=1)
p_05 = np.percentile(trajectories["Positives"], 5, axis=1)
real_trajectory = data["realTraj"]

t = [initial_day + datetime.timedelta(days=i * time_step) for i in range(1, number_of_times + 1)]
#t2 = [initial_day + datetime.timedelta(days=(i + 1) * time_step) for i in range(number_of_times)]
#t2 += [initial_day + datetime.timedelta(days=(i + 1) * time_step) for i in range(number_of_times - 1, -1, -1)]
t2 = [initial_day + datetime.timedelta(days=(i + 1) * time_step) for i in range(number_of_times)]

# Plotting
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)
ax_beta = fig.add_subplot(gs[0, 0])
ax_gamma = fig.add_subplot(gs[0, 1])
ax_traj = fig.add_subplot(gs[1, :])

sns.set(style="whitegrid", font_scale=1.3)

for j in range(1, 4):  # Run over repeats of the inference

    with open(os.path.join("Figure_2", "Inf_res_" + str(j) + ".pickle"), "rb") as file:
        temp = pickle.load(file)

    gammas = temp["gammas"]
    betas = temp["betas"]

    # Get mean of prior
    gammas_mean = np.zeros(n_iterations + 1)
    gammas_mean[0] = np.mean(gammas[:, 0])
    betas_mean = np.zeros(n_iterations + 1)
    betas_mean[0] = np.mean(betas[:, 0])

    # Get 90% CI of prior
    gammas_q95 = np.zeros(n_iterations + 1)
    gammas_q95[0] = np.percentile(gammas[:, 0], 95)
    betas_q95 = np.zeros(n_iterations + 1)
    betas_q95[0] = np.percentile(betas[:, 0], 95)
    gammas_q05 = np.zeros(n_iterations + 1)
    gammas_q05[0] = np.percentile(gammas[:, 0], 5)
    betas_q05 = np.zeros(n_iterations + 1)
    betas_q05[0] = np.percentile(betas[:, 0], 5)

    # Get mean and 90% CI of posterior for each iteration
    for n in range(1, n_iterations + 1):
        x = (n * (number_of_times + 1) + 1)-1
        x2 = np.arange((n - 1) * (number_of_times + 1) + 2, n * (number_of_times + 1))
        gammas_mean[n] = np.mean(gammas[:, x-1])
        betas_mean[n] = np.mean(betas[:, x-1])
        gammas_q95[n] = np.percentile(gammas[:, x2], 95)
        gammas_q05[n] = np.percentile(gammas[:, x2], 5)
        betas_q95[n] = np.percentile(betas[:, x2], 95)
        betas_q05[n] = np.percentile(betas[:, x2], 5)

    # Collect into a single structure
    parameters_df = pd.DataFrame(
        {
            "betas_mean": betas_mean,
            "betas_q95": betas_q95,
            "betas_q05": betas_q05,
            "gammas_mean": gammas_mean,
            "gammas_q95": gammas_q95,
            "gammas_q05": gammas_q05,
        }
    )

    # Plot Beta
    ax_beta.fill_between(
        range(n_iterations + 1),
        parameters_df["betas_q95"],
        parameters_df["betas_q05"],
        alpha=0.2,
        color=colors[j-1]
    )
    ax_beta.plot(range(n_iterations + 1), parameters_df["betas_mean"], linewidth=1, color=colors[j-1])
    ax_beta.plot(range(n_iterations + 1), parameters_df["betas_mean"], "*", color=colors[j-1])
    ax_beta.axhline(parameters_df["betas_mean"][len(parameters_df["betas_mean"]) - 1], ls='--',
                    color=colors[j-1])
    ax_beta.set_xlabel("Iteration")
    ax_beta.set_ylabel(r"$\beta$")
    ax_beta.text(-4, 0.105, "A.", fontsize=17, fontname="Times New Roman")
    ax_beta.set_xticks(range(0,n_iterations + 1,5))
    ax_beta.tick_params(labelsize=12 )

    # Plot Gamma
    ax_gamma.fill_between(
        range(n_iterations + 1),
        parameters_df["gammas_q95"],
        parameters_df["gammas_q05"],
        alpha=0.2,
        color = colors[j-1]
    )
    ax_gamma.plot(range(n_iterations + 1), parameters_df["gammas_mean"], linewidth=1, color=colors[j-1])
    ax_gamma.plot(range(n_iterations + 1), parameters_df["gammas_mean"], "*", color=colors[j-1])
    ax_gamma.axhline(parameters_df["gammas_mean"][len(parameters_df["gammas_mean"]) - 1], ls='--', color=colors[j-1])
    ax_gamma.set_xlabel("Iteration")
    ax_gamma.set_ylabel(r"$\gamma$")
    ax_gamma.text(-4, 0.105, "B.", fontsize=17, fontname="Times New Roman")
    ax_gamma.set_xticks(range(0, n_iterations + 1, 5))
    ax_gamma.tick_params(labelsize=12)

# Plot Trajectories
ax_traj.fill_between(
    t2,
    p_95, p_05,
    alpha=0.2,
    color=colors[3]
)
ax_traj.plot(t, real_trajectory, "k", linewidth=3)
ax_traj.plot(t, poss[:, 0], linewidth=1.5, color=colors[4])
ax_traj.plot(t, poss[:, 4], linewidth=1.5, color=colors[5])
ax_traj.set_xlabel("Time")
ax_traj.set_ylabel("Position")
ax_traj.text(initial_day - datetime.timedelta(days=150), 42, "C.", fontsize=17, fontname="Times New Roman")
ax_traj.tick_params(labelsize=12)
ax_traj.set_xticklabels(labels=[(initial_day + datetime.timedelta(days=date_lag)).strftime("%b-%y") for date_lag in
                                    range(0, number_of_times * time_step, 6 * 30)], rotation=45, ha='right')

# Adjust layout and save figure
plt.tight_layout()
plt.savefig(os.path.join("Figure_2", "fig_2.pdf"))
plt.close()
