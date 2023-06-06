import numpy as np
def Inference_iEAKF(variable_range, hyparams, wanted_std, Days, RPos):
    # No fields:
    # 1. tmstep - number of days in a time step
    # 2. Days - number of days in the data set
    # 3. vars - number of variables
    # 4. Iter - number of iterations
    # 5. Ens - number of ensemble members
    # 6. IntP - probability to be colonized on day 0

    # Timesteps:
    ts = np.arange(1, hyparams["Number of days"], hyparams["Number of days in a timestep"])
    hyparams["Number of times"] = len(ts) - 1

    # Initialization:
    theta = np.zeros((hyparams["Number of variables"], hyparams["Number of iterations"] + 1))
    theta[:, 0] = np.mean(list(variable_range.values()), axis=1)

    # initialize parameters from uniform distribution over the parameter range:
    samples_from_parameters = {}
    for variable, range_values in variable_range.items():
        lower_bound, upper_bound = range_values
        samples_from_parameters[variable] = np.random.uniform(lower_bound, upper_bound, hyparams["Number of iterations"])

    betas = samples_from_parameters['Beta']
    gammas = samples_from_parameters["Gamma"]
    poss = samples_from_parameters["Beta"]

    # Iterations: =============================================================
    for n in range(hyparams["Number of iterations"]):

        xpost = np.zeros((hyparams["Number of variables"], hyparams['Number of ensemble members'], hyparams["Number of times"]))

        # Initialize patient status
        P_status = np.random.binomial(n=1, p=hyparams["Probability of colonized upon admission"],
                                      size=(hyparams["Number of patients"], hyparams['Number of ensemble members']))

        for t in range(hyparams["Number of times"]):

            #TODO: Days being a dictionary is pretty useless and should be changed to Panel
            days_subset = dict((k, Days[k]) for k in list(range(ts[t],ts[t + 1])))
            P_status, Pos = progress_patients_nec(days=days_subset, P_status, vars=samples_from_parameters)
            #FIXME: I'm here
            Poss = np.concatenate([Poss, Pos.T])

            # Kalman filter: ==============================================
            Obs = Pos
            Obs[np.isnan(Obs)] = 0
            Obstruth = RPos[t]

            So = iEAKF(Obs, Obstruth, So, Vrange)
            # Note: The iEAKF function is not provided, please define it separately.
            # =============================================================

            # Jittering between time steps
            for j in range(2):
                if np.std(So.iloc[:, j]) < Wantstd[n, j]:
                    sig = np.sqrt(Wantstd[n, j] ** 2 - np.std(So.iloc[:, j]) ** 2)
                    So.iloc[:, j] = So.iloc[:, j] + np.random.normal(0, sig, (hyparams['Number of ensemble members'],))
                    So.iloc[So.iloc[:, j] < Vrange.iloc[0, j], j] = Vrange.iloc[0, j]  # check lower bound
                    So.iloc[So.iloc[:, j] > Vrange.iloc[1, j], j] = Vrange.iloc[1, j]  # check upper bound

            xpost[:, :, t] = So.values.T
