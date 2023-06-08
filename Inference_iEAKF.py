import numpy as np
from progress_patients_nec import progress_patients_nec
from iEAKF import iEAKF
import pandas as pd

def Inference_iEAKF(variable_range, hyparams, wanted_std, days, r_pos):
    # No fields:
    # 1. tmstep - number of days in a time step
    # 2. Days - number of days in the data set
    # 3. vars - number of variables
    # 4. Iter - number of iterations
    # 5. Ens - number of ensemble members
    # 6. IntP - probability to be colonized on day 0

    # Timesteps:
    ts = np.arange(start=1, stop=hyparams["Number of days"], step=hyparams["Number of days in a timestep"])
    hyparams["Number of times"] = len(ts) - 1

    # Initialization:
    theta = np.empty((hyparams["Number of variables"], hyparams["Number of iterations"] + 1))
    theta[:, 0] = np.mean(variable_range, axis=0)

    # initialize parameters from uniform distribution over the parameter range:
    samples_from_parameters = {}
    for variable, range_values in variable_range.items():
        lower_bound, upper_bound = range_values
        samples_from_parameters[variable] = np.random.uniform(lower_bound, upper_bound,
                                                              hyparams["Number of ensemble members"])
    samples_from_parameters = pd.DataFrame(samples_from_parameters)

    #Add initial samples
    betas = samples_from_parameters["Beta"].to_numpy()
    gammas = samples_from_parameters["Gamma"].to_numpy()
    poss = samples_from_parameters["Beta"].to_numpy()

    # Iterations: =============================================================
    for n in range(hyparams["Number of iterations"]):

        xpost = np.zeros(
            (hyparams["Number of variables"], hyparams['Number of ensemble members'], hyparams["Number of times"]))

        # Initialize patient status
        p_status = np.random.binomial(n=1, p=hyparams["Probability of colonized upon admission"],
                                      size=(hyparams["Number of patients"], hyparams['Number of ensemble members']))

        for t in range(hyparams["Number of times"]):

            p_status, pos, _, _ = progress_patients_nec(days=days[ts[t]:ts[t + 1]], p_pre=p_status,
                                                        samples_from_parameters=samples_from_parameters)
            poss = np.concatenate([poss, pos.T])

            # Kalman filter: ==============================================
            obetas = pos
            obetas[np.isnan(obetas)] = 0
            obetastruth = r_pos[t]

            samples_from_parameters = iEAKF(obetas, obetastruth, samples_from_parameters, variable_range)
            for j in range(2):
                if np.std(samples_from_parameters.iloc[:, j]) < wanted_std[n, j]:
                    sig = np.sqrt(np.square(wanted_std[n, j]) - np.square(np.std(samples_from_parameters.iloc[:, j])))
                    samples_from_parameters.iloc[:, j] = samples_from_parameters.iloc[:, j] + np.random.normal(0, sig, (hyparams['Number of ensemble members'],))
                    samples_from_parameters.iloc[samples_from_parameters.iloc[:, j] < variable_range.iloc[0, j], j] = variable_range.iloc[0, j]  # check lower bound
                    samples_from_parameters.iloc[samples_from_parameters.iloc[:, j] > variable_range.iloc[1, j], j] = variable_range.iloc[1, j]  # check upper bound

            xpost[:, :, t] = samples_from_parameters.values.T

        #Exit time loop
        theta[:, n + 1] = np.mean(xpost, axis=(1, 2)).squeeze()

        #TODO: Aquí vamos
        betas = np.concatenate((betas, xpost[0, :, :].squeeze()), axis = 1)
        gammas = np.concatenate((gammas, xpost[1, :, :].squeeze()))

        samples_from_parameters = np.mean(xpost, axis=2).T
        betas.append(samples_from_parameters[:, 0])
        gammas.append(samples_from_parameters[:, 1])

        VarNames = [f"Var_{i}" for i in range(samples_from_parameters.shape[1])]
        samples_from_parameters = pd.DataFrame(samples_from_parameters, columns=VarNames)

        print(f'iteration {n} - {toc / 60} mins')

    Dits = {}
    Dits['realTraj'] = RPos
    Dits['gammas'] = gammas
    Dits['betas'] = betas
    Dits['Poss'] = Poss
    Dits['Vrange'] = Vrange
    Dits['tmstep'] = No_tmstep
    Dits['IntP'] = No_IntP
    Dits['day0'] = No_day0
    Dits['num_times'] = No_times
    Dits['num_ens'] = No_ens
    Dits['num_iter'] = No_Iter
