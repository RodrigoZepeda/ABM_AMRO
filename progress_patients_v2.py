import numpy as np
import numba
import pandas as pd


# from numba.pycc import CC
# Single trajectory 2 minutes
# Get values from Tal
# Using Google's style https://google.github.io/styleguide/pyguide.html
@numba.njit(cache=True)
def progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters):
    """
        Examples:
        ```
        total_patients = 20
        ward_matrix = np.array([
                            [52, 53, 17, 200, 87, 99], #MRN
                            [1, 1, 0, 0, 1, 0], #New_arrival
                            [1,1,1,1,1,1], #Weight
                            [0,0,1,0,0,0], #Colonized status under model 1
                            [0,0,1,0,0,1], #Colonized status under model 2
                            [0,0,0,0,0,0], #Colonized status under model 3
                            ]).transpose()

        parameters = np.array([
                        np.linspace(0.1, 0.9, 3), #Alpha
                        np.linspace(0.2, 0.5, 3), #Beta
                        np.linspace(0.5, 0.7, 3), #Gamma
                        ])
        %timeit -n1 -r100 next_colonized = progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters)

        ```

    """

    # Create matrix of weights = C/k
    weighted_colonized = np.multiply((1 / ward_matrix[:, 2]).reshape(-1, 1), ward_matrix[:, 3:ward_matrix.shape[1]])

    # Calculate the ward's force of infection
    # force_of_infection = beta/N*sum(C_[:,d-1]/k)
    force_of_infection = np.multiply(parameters[1, :] / total_patients, np.sum(weighted_colonized, axis=0))

    # Calculate colonizations attributable to ward:
    # ward_attributable = (1- alpha)*C[i,d-1]/k + (1 - C[i, d - 1]/k)*transmission_rate
    ward_attributable = np.multiply(weighted_colonized, 1 - parameters[0, :]) + np.multiply(1 - weighted_colonized,
                                                                                            force_of_infection)

    # Compute the new colonized and return
    # new_colonized = ward_attributable + gamma*h[i,d]
    colonized_probability = np.multiply(ward_matrix[:, 1], parameters[2, :].reshape(-1, 1)).transpose() + \
                            np.multiply((1 - ward_matrix[:, 1]).reshape(-1, 1), ward_attributable)

    # Calculate the number of colonized individuals from the binomial model
    # https://stackoverflow.com/questions/66468953/numba-compatibility-with-numpy-random-binomial
    new_colonized = np.empty(shape=(colonized_probability.shape[0], colonized_probability.shape[1] + 3))
    new_colonized[:, 0:3] = ward_matrix[:, 0:3]
    new_colonized[:, 3:new_colonized.shape[1]] = np.random.uniform(low=0, high=1,
                                                                   size=colonized_probability.shape) < colonized_probability

    return new_colonized


@numba.njit(cache=True, parallel=True)
def progress_patients_1_timestep(ward_matrix, total_patients, parameters, ward_progression_function):
    """
        Examples:
        ```
        total_patients = np.array([
                                    [1,3], #Ward
                                    [10,20], #Size
                                    ]).transpose()
        ward_matrix = np.array([
                            [1, 3, 1, 3, 3, 3],         #Ward
                            [52, 53, 17, 200, 87, 99], #MRN
                            [1, 1, 0, 0, 1, 0], #New_arrival
                            [1,1,1,1,1,1], #Weight
                            [0,0,1,0,0,0], #Colonized status under model 1
                            [0,0,1,0,0,1], #Colonized status under model 2
                            [0,0,0,0,0,0], #Colonized status under model 3
                            ]).transpose()

        parameters = np.array([
                        np.linspace(0.1, 0.9, 3), #Alpha
                        np.linspace(0.2, 0.5, 3), #Beta
                        np.linspace(0.5, 0.7, 3), #Gamma
                        ])
        %timeit -n1 -r100 next_colonized = progress_patients_1_timestep(ward_matrix, total_patients, parameters, progress_patients_probability_ward_1_timestep)
        ```
    """

    # Get unique ward values for the day
    active_wards = np.unique(ward_matrix[:, 0])
    next_step_colonized = np.empty(shape=ward_matrix.shape)

    # Loop through the wards that have cases during the period
    for w in numba.prange(len(active_wards)):
        # Select the specific ward
        ward = ward_matrix[ward_matrix[:, 0] == active_wards[w], 1:ward_matrix.shape[1]]
        total_patients_in_ward = total_patients[total_patients[:, 0] == active_wards[w], 1]

        # Advance the specific ward
        next_step_colonized[ward_matrix[:, 0] == active_wards[w], 0] = active_wards[w]
        next_step_colonized[ward_matrix[:, 0] == active_wards[w],
        1:next_step_colonized.shape[1]] = ward_progression_function(ward, total_patients_in_ward[0], parameters)

    return next_step_colonized


@numba.njit(fastmath=True)
def isin(value, A):
    for a in range(len(A)):
        if A[a] == value:
            return True
    return False


@numba.njit(cache=True, fastmath=True, parallel=True)
def where_array_A_in_B(A, B):
    """
    Checks the places where array A is in B and viceversa
    :param A: One dimensional array
    :param B: One dimensional array
    :return: in_A, in_B the indexes of where the intersection elements that are in A and B
    match

    Example
    --------
    A = np.array([1,2,3,4,5,6,7,8,9,10])
    B = np.array([8,1, 16, 17, 18, 9, 3])
    in_A, in_B = where_array_A_in_B(A,B)
    print(f'Elements in A:{in_A}')
    print(f'Elements in B:{in_B}')
    """
    in_A = np.empty(A.shape[0], dtype=numba.boolean)
    in_B = np.empty(B.shape[0], dtype=numba.boolean)

    for a in numba.prange(A.shape[0]):
        in_A[a] = isin(A[a], B)

    for b in numba.prange(B.shape[0]):
        in_B[b] = isin(B[b], A)

    return in_A, in_B


@numba.njit(cache=True)
def simulate_discrete_model_internal_one(colonized_day_0, ward_matrix, total_patients_per_ward, parameters,
                                         ward_progression_function, day_progression_function):
    """
    Examples:
    ```
    total_patients_per_ward = np.array([
                                        [0,0,1,1,1,2,2,2], #Day
                                        [1,3,1,2,3,1,2,3], #Ward
                                        [10,20,11,12,19,11,13,18], #Size
                                        ]).transpose()
    ward_matrix = np.array([
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], #Day
                        [1, 3, 1, 3, 3, 3, 2, 1, 2, 3, 3, 1, 2, 3, 1], #Ward
                        [52, 53, 17, 200, 87, 99, 52, 53, 44, 86, 100, 87, 87, 44, 52], #MRN
                        [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], #New_arrival
                        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], #Weight
                        ]).transpose()
   colonized_day_0 = np.array([
                    [0,0,1,0,0,0], #Colonized probability under model 1
                    [0.9,0.1,0.5,0.2,0.1,0.2], #Colonized probability under model 2
                    np.repeat(1/2, 6), #Colonized probability under model 3
                    ]).transpose()
    parameters = np.array([
                    np.linspace(0.1, 0.9, 3), #Alpha
                    np.linspace(0.2, 0.5, 3), #Beta
                    np.linspace(0.5, 0.7, 3), #Gamma
                    ])
    %timeit -n1 -r100 model_colonized = simulate_discrete_model_internal_one(colonized_day_0, ward_matrix, total_patients_per_ward, parameters, progress_patients_probability_ward_1_timestep, progress_patients_1_timestep)
    ```
    """

    # Get the maximum amount of days to run the model for
    total_days = max(total_patients_per_ward[:, 0]) + 1

    # Start the process with the initial state simulation
    col_init_colonized = ward_matrix.shape[1]
    col_end_colonized = col_init_colonized + colonized_day_0.shape[1] + 1
    model_colonized = np.zeros(shape=(ward_matrix.shape[0], col_end_colonized - 1))
    model_colonized[ward_matrix[:, 0] == 0, col_init_colonized:col_end_colonized] = \
        np.random.uniform(low=0, high=1, size=colonized_day_0.shape) < colonized_day_0

    # Fill the other values of the initial state
    model_colonized[:, 0:ward_matrix.shape[1]] = ward_matrix

    # Loop through each of the days
    for day in range(total_days):

        # Get the total amount of patients in that day
        total_patients_per_day = total_patients_per_ward[total_patients_per_ward[:, 0] == day,
                                 1:total_patients_per_ward.shape[1]]

        # Get the ward info for the day
        ward_per_day = model_colonized[model_colonized[:, 0] == day, 1:model_colonized.shape[0]]

        # Run the step
        colonized_status_day = day_progression_function(ward_matrix=ward_per_day,
                                                        total_patients=total_patients_per_day,
                                                        parameters=parameters,
                                                        ward_progression_function=ward_progression_function)

        # Add to the wards_per_day
        in_data, in_model = where_array_A_in_B(model_colonized[model_colonized[:, 0] == (day + 1), 2], colonized_status_day[:, 1])

        #Get the index of stuff
        model_colonized[np.where(model_colonized[:, 0] == day + 1)[0][in_data], col_init_colonized:col_end_colonized] = \
            colonized_status_day[in_model,4:colonized_status_day.shape[1]]

    return model_colonized


@numba.njit(parallel=True, cache=True)
def simulate_discrete_model_internal_multiple(nsim, colonized_day_0, ward_matrix, total_patients_per_ward, parameters,
                                         ward_progression_function, day_progression_function):
    """
    Example
    ```
    total_patients_per_ward = np.array([
                                        [0,0,1,1,1,2,2,2], #Day
                                        [1,3,1,2,3,1,2,3], #Ward
                                        [10,20,11,12,19,11,13,18], #Size
                                        ]).transpose()
    ward_matrix = np.array([
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2], #Day
                        [1, 3, 1, 3, 3, 3, 2, 1, 2, 3, 3, 1, 2, 3, 1], #Ward
                        [52, 53, 17, 200, 87, 99, 52, 53, 44, 86, 100, 87, 87, 44, 52], #MRN
                        [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], #New_arrival
                        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], #Weight
                        ]).transpose()
   colonized_day_0 = np.array([
                    [0,0,1,0,0,0], #Colonized probability under model 1
                    [0.9,0.1,0.5,0.2,0.1,0.2], #Colonized probability under model 2
                    np.repeat(1/2, 6), #Colonized probability under model 3
                    ]).transpose()
    parameters = np.array([
                    np.linspace(0.1, 0.9, 3), #Alpha
                    np.linspace(0.2, 0.5, 3), #Beta
                    np.linspace(0.5, 0.7, 3), #Gamma
                    ])
    nsim = 100
    simulations = simulate_discrete_model_internal_multiple(nsim, colonized_day_0, ward_matrix, total_patients_per_ward,
        parameters, progress_patients_probability_ward_1_timestep, progress_patients_1_timestep)
    ```
    """
    # Loop through simulations
    simulations = np.empty(shape=(nsim, ward_matrix.shape[0], ward_matrix.shape[1] + colonized_day_0.shape[1]))
    for n in numba.prange(nsim):
        simulations[n, :, :] = simulate_discrete_model_internal_one(colonized_day_0, ward_matrix,
                                                                    total_patients_per_ward, parameters,
                                                                    ward_progression_function, day_progression_function)

    return simulations



