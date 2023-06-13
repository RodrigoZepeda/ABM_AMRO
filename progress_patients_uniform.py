import numpy as np

def progress_patients_uniform(days, p_pre, parameters):
    # Progressing the status of E ensembles by one day
    # No environmental contamination
    # Returns the positives, the total new patients, and the importations

    pos = 0
    nos = 0
    imp = 0
    p_status = p_pre
    active_p = days[0]["Patients"]
    active_w = days[0]["Wards"]

    for d in range(1, len(days)):

        p_status = (1 - parameters["Alpha"][0])*p_pre

        for k in range(active_w.shape[0]):
            temp = active_p["ward"] == active_w["ward"][k]

            prod = p_pre[active_p["MRN"][temp], :] * np.transpose(
                np.tile(active_p["weight"][temp], (p_pre.shape[1], 1)))
            x = np.sum(prod, axis=0)
            x = x / active_w["Size"][k]

            temp2 = active_p["MRN"][temp]
            temp3 = parameters["Beta"][0] * x  # Beta * X + eps
            temp3 = np.tile(temp3, (len(temp2), 1)) * (1 - p_pre[temp2, :])
            temp3 = temp3 * np.transpose(np.tile(active_p["weight"][temp], (p_pre.shape[1], 1)))
            p_status[temp2, :] = p_status[temp2, :] + temp3

        new_comer = days[d]["Pfirst"]
        p_status[new_comer, :] = parameters["Gamma"]

        temp = np.where(p_status != 0)
        p_status[temp] = np.random.rand(len(temp[0])) <= p_status[temp]

        nos += np.sum(p_status[temp] > p_pre[temp])
        imp += np.sum(p_status[new_comer, :])

        active_p = days[d]["Patients"]
        active_w = days[d]["Wards"]

        p_pre = p_status

        temp = p_status[active_p["MRN"], :]
        test = np.tile(parameters["Rho"], (temp.shape[0], 1))
        test = test > np.random.rand(*test.shape)
        pos += np.sum(temp * test, axis=0)

    return p_status, pos, nos, imp
