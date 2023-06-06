import numpy as np
#FIXME Nothing works
def progress_patients_nec(days, p_pre, vars):
    # Progressing the status of E ensembles by one day
    # No environmental contamination
    # Returns the positives, the total new patients, and the importations

    pos = 0
    new = 0
    imp = 0
    active_P = days[min(days.keys())]["Patients"]
    active_W = days[min(days.keys())]["Wards"]

    for d in range(1, len(days)):
        P_status = p_pre * (1 - np.tile(vars["Alpha"], (p_pre.shape[0], 1)))

        for k in range(active_W.shape[0]):
            temp = active_P["ward"] == active_W["ward"][k]
            X = np.sum(
                p_pre[active_P["MRN"][temp], :] * np.tile(active_P["weight"][temp], (1, p_pre.shape[1])),
                axis=0,
            )
            X = X / active_W["Size"][k]  # X = sum(Ck) / (nrk)

            temp2 = active_P["MRN"][temp]
            temp3 = vars["Beta"] * X  # Beta * X + eps
            temp3 = np.tile(temp3, (len(temp2), 1)) * (1 - p_pre[temp2, :])
            P_status[temp2, :] = P_status[temp2, :] + temp3

        active_P = days[d]["patients"]
        active_W = days[d]["wards"]
        new_comer = days[d]["Pfirst"]
        P_status[new_comer, :] = np.tile(vars["Gamma"], (len(new_comer), 1))

        temp = np.where(P_status == 0)
        P_status[temp] = np.random.rand(len(temp[0])) <= P_status[temp]

        new += np.sum(P_status > p_pre)
        imp += np.sum(P_status[new_comer, :])

        p_pre = P_status

        temp = P_status[active_P["MRN"], :]
        test = np.tile(vars["Rho"], (temp.shape[0], 1))
        test = test > np.random.rand(*test.shape)
        pos += np.sum(temp * test, axis=0)

    return P_status, pos, new, imp
