import numpy as np


def iEAKF(obs, obstruth, samples_from_parameters, variable_range, tolerance=1e-3):
    var_names = samples_from_parameters.columns
    var_num = len(var_names)

    obs_var = 1 + np.square(0.2 * obstruth)
    prior_var = np.var(obs)
    post_var = prior_var * obs_var / (prior_var + obs_var)
    prior_var = max(prior_var, tolerance)  # making sure prior_var > 0
    prior_mean = np.mean(obs)
    post_mean = post_var * (prior_mean / prior_var + obstruth / obs_var)
    alpha = np.sqrt(obs_var / (obs_var + prior_var))
    dy = post_mean + alpha * (obs - prior_mean) - obs
    rr = np.zeros(var_num)

    for j in range(var_num):
        A = np.cov(samples_from_parameters[var_names[j]], obs)
        rr[j] = A[1, 0] / prior_var

    dx = np.dot(dy.reshape(-1, 1), rr.reshape(1, -1))

    for j in range(var_num):
        temp = var_names[j]
        samples_from_parameters[temp] = samples_from_parameters[temp] + dx[:, j]

        # check lower bound
        samples_from_parameters[temp][samples_from_parameters[temp] < variable_range[temp][0]] = variable_range[temp][0]

        # check upper bound
        samples_from_parameters[temp][samples_from_parameters[temp] > variable_range[temp][1]] = variable_range[temp][1]

    return samples_from_parameters
