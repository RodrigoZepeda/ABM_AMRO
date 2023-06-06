def progress_patients_uniform(days, p_pre, vars):
    # Progressing the status of an ensemble where all Vars are the same
    pos = 0
    nos = 0
    imp = 0
    active_P = days[min(days.keys())]["Patients"]
    active_W = days[min(days.keys())]["Wards"]

    for d in range(1, len(days)):
        P_status = p_pre * (1 - vars["Alpha"])

        for k in range(active_W.shape[0]):
            temp = active_P["ward"] == active_W["ward"][k]
            temp2 = active_P["MRN"][temp]
            x = np.sum(
                p_pre[temp2, :] * np.tile(active_P["weight"][temp], (1, p_pre.shape[1])),
                axis=0,
            )
            x = x / active_W["Size"][k]  # X = sum(Ck) / (nrk)

            temp3 = vars["Beta"] * x  # Beta * F/N
            temp3 = np.tile(temp3, (len(temp2), 1)) * (1 - p_pre[temp2, :])
            temp3 = temp3 * np.tile(active_P["weight"][temp], (1, p_pre.shape[1]))  # Multiply by 1/k
            P_status[temp2, :] = P_status[temp2, :] + temp3

        new_comer = days[d]["Pfirst"]
        P_status[new_comer, :] = np.tile(vars["Gamma"], (len(new_comer), 1))

        temp = np.where(P_status == 0)
        P_status[temp] = np.random.rand(len(temp[0])) <= P_status[temp]

        temp = active_P["MRN"]
        nos += np.sum(P_status[temp, :] > p_pre[temp, :])
        imp += np.sum(P_status[new_comer, :])

        active_P = days[d]["patients"]
        active_W = days[d]["wards"]

        p_pre = P_status

        temp = P_status[active_P["MRN"], :]
        test = np.tile(vars["Rho"], (temp.shape[0], 1))
        test = test > np.random.rand(*test.shape)
        pos += np.sum(temp * test, axis=0)

    return P_status, pos, nos, imp
