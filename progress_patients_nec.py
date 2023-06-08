import numpy as np


def progress_patients_nec(days, p_pre, samples_from_parameters):
    # Progressing the status of E ensembles by one day
    # No environmental contamination
    # Returns the positives, the total new patients, and the importations

    pos = 0
    new = 0
    imp = 0
    p_status = p_pre
    active_p = days[0]["Patients"]
    active_w = days[0]["Wards"]

    for d in range(1, len(days)):

        p_status = p_pre * (1 - np.tile(samples_from_parameters["Alpha"], (p_pre.shape[0], 1)))

        for k in range(active_w.shape[0]):
            temp = active_p["ward"] == active_w["ward"][k]

            prod = p_pre[active_p["MRN"][temp], :] * np.transpose(
                np.tile(active_p["weight"][temp], (p_pre.shape[1], 1)))
            X = np.sum(prod, axis=0)
            X = X / active_w["Size"][k]

            temp2 = active_p["MRN"][temp]
            temp3 = samples_from_parameters["Beta"] * X  # Beta * X + eps
            temp3 = np.tile(temp3, (len(temp2), 1)) * (1 - p_pre[temp2, :])
            p_status[temp2, :] = p_status[temp2, :] + temp3

        active_p = days[d]["Patients"]
        active_w = days[d]["Wards"]
        new_comer = days[d]["Pfirst"]

        p_status[new_comer, :] = np.tile(samples_from_parameters["Gamma"], (len(new_comer), 1))

        temp = np.where(p_status != 0)
        p_status[temp] = np.random.rand(len(temp[0])) <= p_status[temp]
        new += np.sum(p_status > p_pre)
        imp += np.sum(p_status[new_comer, :])

        p_pre = p_status

        temp = p_status[active_p["MRN"], :]
        test = np.tile(samples_from_parameters["Rho"], (temp.shape[0], 1))
        test = test > np.random.rand(*test.shape)
        pos += np.sum(temp * test, axis=0)

    return p_status, pos, new, imp
