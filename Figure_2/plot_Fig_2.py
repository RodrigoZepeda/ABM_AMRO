import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os


# Clear all existing figures
plt.close("all")

# Load data
with open(os.path.join("Figure_2","Inf_res_1.pickle"), "rb") as file:
    data = pickle.load(file)

#FIXME: Use a better color palette
colors = ["#0072BD", "#77AC30", "#A2142F", "#7E2F8E", "#D95319"]  # Blue, Green, Red, Purple, Orange
colors2 = [(49, 196, 88), (220, 149, 27), (26, 110, 220), (158, 93, 145)]

# Retrieve required variables
n_iterations = data["Number of iterations"]
number_of_times = data["Number of times"]
variable_range = data["variable_range"]
initial_day = data["Date before start"]
time_step = data["Number of days in a timestep"]

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
    for n in range(1, n_iterations):
        x = n * (number_of_times + 1) + 1
        x2 = np.arange((n - 1) * (number_of_times + 1) + 2, n * (number_of_times + 1))
        gammas_mean[n] = np.mean(gammas[:, x])
        betas_mean[n] = np.mean(betas[:, x])
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

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    sns.set(style="whitegrid", font_scale=1.3)

    # Plot Beta
    axs[0, 0].fill_between(
        range(n_iterations + 1),
        parameters_df["betas_q95"],
        parameters_df["betas_q05"],
        alpha=0.2
    )
    axs[0, 0].plot(range(n_iterations + 1), parameters_df["betas_mean"], linewidth=1)
    axs[0, 0].plot(range(n_iterations + 1), parameters_df["betas_mean"], "*")
    axs[0, 0].plot([0, n_iterations], [parameters_df["betas_mean"], parameters_df["betas_mean"]], "r.", linewidth=1.5)
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel(r"$\beta$")
    axs[0, 0].text(-4, 0.105, "A.", fontsize=17, fontname="Times New Roman")
    axs[0, 0].tick_params(labelsize=12)

    # Plot Gamma
    axs[0, 1].fill_between(
        range(n_iterations + 1),
        parameters_df["gammas_q95"],
        parameters_df["gammas_q05"],
        alpha=0.2
    )
    axs[0, 1].plot(range(n_iterations + 1), parameters_df["gammas_mean"], linewidth=1)
    axs[0, 1].plot(range(n_iterations + 1), parameters_df["gammas_mean"], "*")
    axs[0, 1].plot([0, n_iterations], [parameters_df["gammas_mean"], parameters_df["gammas_mean"]], "r.", linewidth=1.5)
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel(r"$\gamma$")
    axs[0, 1].text(-4, 0.105, "B.", fontsize=17, fontname="Times New Roman")
    axs[0, 1].tick_params(labelsize=12)

    # Plot Trajectories
    axs[1, 0].fill_between(
        t2,
        np.concatenate([P95, P05[::-1]]),
        color=colors2[3],
        alpha=0.6
    )
    axs[1, 0].plot(t, realTraj, "k", linewidth=3)
    axs[1, 0].plot(t, Poss[:, 0], color=colors[0], linewidth=1.5)
    axs[1, 0].plot(t, Poss[:, 4], color=colors[3], linewidth=1.5)
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Position")
    axs[1, 0].text(day0 - 100, 42, "C.", fontsize=17, fontname="Times New Roman")
    axs[1, 0].tick_params(labelsize=12)
    axs[1, 0].set_xlim([tmstep, tmstep * No_times + day0])

    # Remove empty subplot
    fig.delaxes(axs[1, 1])

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"figure_{j}.png")
    plt.close()

# Set the figure size for the final plot
fig, axs = plt.subplots(figsize=(10, 7))
sns.set(style="whitegrid", font_scale=1.3)

# Plot Trajectories
axs.fill_between(
    t2,
    np.concatenate([P95, P05[::-1]]),
    color=colors2[3],
    alpha=0.6
)
axs.plot(t, realTraj, "k", linewidth=3)
axs.plot(t, Poss[:, 0], color=colors[0], linewidth=1.5)
axs.plot(t, Poss[:, 4], color=colors[3], linewidth=1.5)
axs.set_xlabel("Time")
axs.set_ylabel("Position")
axs.text(day0 - 100, 42, "C.", fontsize=17, fontname="Times New Roman")
axs.tick_params(labelsize=12)
axs.set_xlim([tmstep, tmstep * No_times + day0])

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("figure.png")
plt.show()
