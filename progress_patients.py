import numpy as np
import numba
from numba import njit
import pandas as pd

# Using Google's style https://google.github.io/styleguide/pyguide.html
@njit
def progress_patients_probability_ward_1_timestep(colonized, new_arrivals, total_patients, parameters, weights):
    """
        progress_patients_probability_ward_1_timestep progresses all patients in a ward from 1 timestep to the next
        The function is defined to progress each patient `i` of ward `w` by computing their colonized C
        status on day `d` as:
            p[i,d] = [(1- alpha)*C[i,d-1]/k + beta/N_[d]*(1-C[i,d-1]/k)*sum(C_[:,d-1]/k)]*(1 - h[i,d]) +
                gamma*h[i,d]
        and
            C[i,d] ~ Bernoulli(p[i,d])
        where:
        - gamma is the importation rate,
        - alpha is the clearance rate
        - beta is the transmision rate
        - N[d] is the sum of all individuals at the ward during day d
        - k weights

        Keyword arguments:
        :param colonized: array of colonized individuals by day d - 1. Colonized = 1 else = 0.
            Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
        :param new_arrivals: array of new arrival individuals by day d - 1. New arrival = 1 else = 0.
            Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
        :param total_patients: (int) total number of patients in ward.
        :param parameters: list of arrays with parameters of the model. For present model parameters
            must contain `gamma`, `beta` and `alpha` entries.
        :param weights: Parameter describing the percent of day spent by individual in ward.

        :return: `colonized` array of colonized individuals by day d. Individuals are rows.


        Returns multiple columns if you are evaluating several parameters at once.

        Examples:
        ```
        #Advance the following ward one day testing one set of parameters
        np.random.seed(3245)
        colonized = np.random.binomial(n=1,p=0.4,size=7) == 1
        new_arrivals = np.random.binomial(n=1,p=0.05,size=7) == 1
        total_patients = 200
        parameters = numba.typed.Dict()
        parameters["alpha"] = 0.15
        parameters["beta"] = 0.25
        parameters["gamma"] = 0.07
        weights = np.tile(1, colonized.shape)
        new_colonized = progress_patients_probability_ward_1_timestep(colonized,new_arrivals,total_patients,parameters,weights)

        #Advance the following ward one day testing multiple parameters
        np.random.seed(3245)
        num_parameters = 10
        colonized = np.random.binomial(n=1,p=0.4,size=7) == 1
        new_arrivals = np.random.binomial(n=1,p=0.05,size=7) == 1
        colonized = np.tile(colonized, (num_parameters,1)).transpose()
        new_arrivals = np.tile(new_arrivals, (num_parameters,1)).transpose()
        total_patients = 200
        weights = np.tile(1, colonized.shape)
        parameters = numba.typed.Dict()
        parameters["alpha"] = np.linspace(0.1, 0.2, num_parameters)
        parameters["beta"] = np.linspace(0.2, 0.3, num_parameters)
        parameters["gamma"] = np.linspace(0.05, 0.1, num_parameters)
        new_colonized = progress_patients_probability_ward_1_timestep(colonized,new_arrivals,total_patients,parameters,weights)
        ```

    """

    # Get the weighted version of colonized patients
    # weighted_colonized = C/k
    weighted_colonized = np.multiply(colonized, 1 / weights)

    # Calculate the ward's force of infection
    # force_of_infection = beta/N*sum(C_[:,d-1]/k)
    force_of_infection = np.multiply(parameters["beta"] / total_patients, np.sum(weighted_colonized, axis=0))

    # Calculate colonizations attributable to ward:
    # ward_attributable = (1- alpha)*C[i,d-1]/k + (1 - C[i, d - 1]/k)*transmission_rate
    ward_attributable = np.multiply(weighted_colonized, 1 - parameters["alpha"])
    ward_attributable += np.multiply(1 - weighted_colonized, force_of_infection)

    # Calculate the ones attributable to new arrivals
    # arrival_attributable = gamma*h[i,d]
    arrival_attributable = np.multiply(new_arrivals, parameters["gamma"])

    # Compute the new colonized and return
    # new_colonized = ward_attributable + arrival_attributable
    colonized_probability = np.multiply(ward_attributable, 1 - new_arrivals) + arrival_attributable

    # Calculate the number of colonized individuals from the binomial model
    # https://stackoverflow.com/questions/66468953/numba-compatibility-with-numpy-random-binomial
    random_probability = np.random.uniform(low=0, high=1, size=colonized_probability.shape)
    new_colonized = random_probability < colonized_probability

    return new_colonized


@njit
def progress_patients_1_timestep(colonized, wards, new_arrivals, weights, total_patients_per_ward, parameters,
                                 ward_progression_function):
    """
        progress_patients_1_timestep progresses all patients in all wards from 1 timestep to the next
        The function is defined to progress each patient `i` of each ward `w` by computing their colonized C
        status on day `d` as specified by the `ward_progression_function`

        Keyword arguments:
        :param colonized: array of colonized individuals by day d - 1. Colonized = 1 else = 0.
            Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
        :param wards: array of the wards each individual corresponds to by day d - 1. Individuals are rows.
        :param new_arrivals: array of new arrival individuals by day d - 1. New arrival = 1 else = 0.
            Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
        :param weights: Parameter describing the percent of day spent by individual in ward.
        :param total_patients_per_ward: Array of (ints) total number of patients at day `d`
        indexed by ward.
            - 1st column: `N` total number of patients
            - 2nd column: `ward` The ward for which the number of patients was computed
        :param parameters: Dict of arrays with parameters of the `ward_progression_function` model.
        :param method: Method for propagating uncertainty either probability propagation or cases. 
        :param ward_progression_function: Function describing how to progress each ward
        :return: array of colonized individuals by day d. Individuals are rows. Returns multiple
            columns if you are evaluating several parameters at once.

        Examples:
        ```
        ### RUN ONE SET OF PARAMETERS
        np.random.seed(24798)

        #Generate the colonized_ward_arrival dataset
        colonized = np.random.binomial(n=1,p=0.4,size=(20,1)) == 1
        wards = np.random.randint(1, 4, size = 20)
        new_arrivals = np.random.binomial(n=1,p=0.1,size=(20,1)) == 1
        weights = np.tile(1, (20,1))

        #Generate the total patients per ward
        wardnames = np.unique(wards) #Total number of wards
        minward = sum(colonized)
        maxward = minward + 25
        total_patients_per_ward = np.random.randint(minward, maxward, size = len(wardnames))
        total_patients_per_ward = np.column_stack((total_patients_per_ward, range(1, len(wardnames) + 1)))

        #Create parameters
        parameters = numba.typed.Dict() # dict()
        parameters["alpha"] = 0.15
        parameters["beta"] = 0.25
        parameters["gamma"] = 0.07

        #Colonize nextstep
        next_step = progress_patients_1_timestep(colonized, wards, new_arrivals, weights, total_patients_per_ward,
                parameters, ward_progression_function=progress_patients_probability_ward_1_timestep)


        #RUN MULTIPLE PARAMETERS
        np.random.seed(3245)
        num_parameters = 10

        #Generate the data
        colonized = np.random.binomial(n=1,p=0.4,size=7)
        new_arrivals = np.random.binomial(n=1,p=0.05,size=7)
        wards = np.random.randint(1, 4, size = 7)
        colonized = np.tile(colonized, (num_parameters,1)).transpose()
        new_arrivals = np.tile(new_arrivals, (num_parameters,1)).transpose()
        weights = np.tile(1, colonized.shape)

        #Generate the total patients per ward
        wardnames = np.unique(wards) #Total number of wards
        minward = max(sum(colonized))
        maxward = minward + 25
        total_patients_per_ward = np.random.randint(minward, maxward, size = len(wardnames))
        total_patients_per_ward = np.column_stack((total_patients_per_ward, range(1, len(wardnames) + 1)))

        #Create the parameters
        parameters = numba.typed.Dict() # dict()
        parameters["alpha"] = np.linspace(0.1, 0.2, num_parameters)
        parameters["beta"] = np.linspace(0.2, 0.3, num_parameters)
        parameters["gamma"] = np.linspace(0.2, 0.3, num_parameters)

        #Colonize nextstep
        next_step = progress_patients_1_timestep(colonized, wards, new_arrivals, weights, total_patients_per_ward,
                parameters, ward_progression_function=progress_patients_probability_ward_1_timestep)
        ```
    """

    # Get unique ward values for the day
    active_wards = np.unique(wards)

    # Create empty array to concatenate results into
    next_step_colonized = np.empty(shape=(0, colonized.shape[1]))
    # next_step_wards = np.empty(shape=0)

    # Loop through the wards that have cases during the period
    for ward in active_wards:
        # Select the specific ward
        ward_index = (wards == ward)

        # Get the ward's size
        total_patients = total_patients_per_ward[total_patients_per_ward[:, 1] == ward, 0]

        # Advance the specific ward
        colonized_ward = ward_progression_function(colonized=colonized[ward_index],
                                                   new_arrivals=new_arrivals[ward_index],
                                                   total_patients=total_patients,
                                                   parameters=parameters,
                                                   weights=weights[ward_index], )

        # Add to next steps
        next_step_colonized = np.append(next_step_colonized, colonized_ward, axis=0)
        # next_step_wards = np.append(next_step_wards, np.array(ward).repeat(colonized_ward.shape[0]), axis=0)

    return next_step_colonized


@njit
def simulate_discrete_model_internal_one(initial_colonized, wards, new_arrivals, weights, total_patients_per_ward,
                                         parameters, day_progression_function, ward_progression_function):
    """
    `simulate_discrete_model_internal_one` progresses all patients in all wards from 1 timestep to the next
    across all days in a range.  The function is defined to progress each patient `i` of each ward `w` by
    computing their colonized C status on day `d` as specified by the `day_progression_function`. This
    function does not do any checks. They are done in `simulate_discrete_model`.

    Keyword arguments:
    :param initial_colonized: array of colonized individuals in day 0 (1st column). Colonized = 1 else = 0.
        Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
    :param wards: array of the wards each of the colonized individuals corresponds to. Indexed by day
        (1st column). Individuals are rows.
    :param new_arrivals: array of new arrival individuals indexed day (1st column). New arrival = 1 else = 0.
        Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
    :param weights: array describing the percent of day spent by individual in ward. Indexed by day (1st column)
    :param total_patients_per_ward: Array of (ints) total number of patients at day `d`
    indexed by ward.
        - 0th column: `d` day
        - 1st column: `N` total number of patients
        - 2nd column: `ward` The ward for which the number of patients was computed
    :param parameters: Dict of arrays with parameters of the `ward_progression_function` model.
    :param day_progression_function: Function describing how to progress each day
    :param ward_progression_function: Function describing how to progress within ward.

    :return: array of colonized individuals across al days. Individuals are rows. Returns multiple
        columns if you are evaluating several parameters at once.

    Examples:
    ```
    #RUN ONE SET OF PARAMETERS
    np.random.seed(3245)
    num_obs = 7   #Number of patients
    num_days = 5  #Run model by 5 days
    num_wards = 4 #Total number of wards

    initial_colonized = np.random.binomial(n=1,p=0.4,size=(num_obs,1)) == 1
    new_arrivals = np.empty((num_days, num_obs, 1))
    weights = np.tile(1, new_arrivals.shape)
    wards = np.empty((num_days, num_obs))
    total_patients_per_ward = np.empty((0, 3), dtype = "int")

    for day in range(num_days):

        #Generate the data
        new_arrivals_temp = np.random.binomial(n=1,p=0.05,size=(7,1))
        new_arrivals[day,:,:] = new_arrivals_temp

        wards[day,:] = np.random.randint(1, 4, size = 7)

        minward = 20
        maxward = minward + 25
        total_patients_per_ward_temp = np.random.randint(minward, maxward, size = num_wards)
        total_patients_per_ward_temp = np.column_stack((np.repeat(day, num_wards),
                total_patients_per_ward_temp, np.arange(1, num_wards + 1)))
        total_patients_per_ward = np.append(total_patients_per_ward, total_patients_per_ward_temp, axis = 0)

    #Create the parameters
    parameters = numba.typed.Dict() # dict()
    parameters["alpha"] = 0.15
    parameters["beta"] = 0.24
    parameters["gamma"] = 0.2

    #Colonize nextstep
    model_simulation = simulate_discrete_model_internal_one(initial_colonized, wards, new_arrivals, weights,
                                                        total_patients_per_ward, parameters,
                                                        progress_patients_1_timestep,
                                                        progress_patients_probability_ward_1_timestep)

    #RUN MULTIPLE PARAMETERS
    np.random.seed(3245)
    num_obs = 7   #Number of patients
    num_days = 5  #Run model by 5 days
    num_wards = 4 #Total number of wards
    num_parameters = 10 #Total number of parameters to run at same time

    initial_colonized = np.random.binomial(n=1,p=0.4,size=num_obs) == 1
    initial_colonized = np.tile(initial_colonized, (num_parameters,1)).transpose()

    new_arrivals = np.empty((num_days, num_obs, num_parameters))

    wards = np.empty((num_days, num_obs))
    total_patients_per_ward = np.empty((0, 3), dtype = "int")

    for day in range(num_days):

        #Generate the data
        new_arrivals_temp = np.random.binomial(n=1,p=0.05,size=7)
        new_arrivals_temp = np.tile(new_arrivals_temp, (num_parameters,1)).transpose()
        new_arrivals[day,:,:] = new_arrivals_temp

        wards[day,:] = np.random.randint(1, 4, size = 7)

        minward = 20
        maxward = minward + 25
        total_patients_per_ward_temp = np.random.randint(minward, maxward, size = num_wards)
        total_patients_per_ward_temp = np.column_stack((np.repeat(day, num_wards),
                total_patients_per_ward_temp, np.arange(1, num_wards + 1)))
        total_patients_per_ward = np.append(total_patients_per_ward, total_patients_per_ward_temp, axis = 0)

    #Create weights
    weights = np.tile(1, new_arrivals.shape)

    #Create the parameters
    parameters = numba.typed.Dict() # dict()
    parameters["alpha"] = np.linspace(0.1, 0.2, num_parameters)
    parameters["beta"] = np.linspace(0.2, 0.3, num_parameters)
    parameters["gamma"] = np.linspace(0.2, 0.3, num_parameters)


    #Colonize nextstep
    model_simulation = simulate_discrete_model_internal_one(initial_colonized, wards, new_arrivals, weights, total_patients_per_ward,
                                                parameters, progress_patients_1_timestep,
                                                progress_patients_probability_ward_1_timestep)
    ```
    """

    total_days = new_arrivals.shape[0]

    model_colonized = np.empty(shape=((total_days,) + initial_colonized.shape))
    model_colonized[0, :] = initial_colonized

    # Loop through each of the days
    for day in range(1, total_days):
        # Total number of patients
        total_patients = total_patients_per_ward[total_patients_per_ward[:, 0] == day][:, 1:3]

        # Run a day of the function
        model_colonized[day, :, :] = day_progression_function(colonized=model_colonized[day - 1, :, :],
                                                              wards=wards[day, :],
                                                              new_arrivals=new_arrivals[day, :, :],
                                                              weights=weights[day, :, :],
                                                              total_patients_per_ward=total_patients,
                                                              parameters=parameters,
                                                              ward_progression_function=ward_progression_function)

    return model_colonized


@njit(parallel=True)
def simulate_discrete_model_internal_multiple(initial_colonized, wards, new_arrivals, weights, total_patients_per_ward,
                                              parameters, day_progression_function, ward_progression_function,
                                              nsim=100):
    """
    `simulate_discrete_model_internal_one` progresses all patients in all wards from 1 timestep to the next
    across all days in a range.  The function is defined to progress each patient `i` of each ward `w` by
    computing their colonized C status on day `d` as specified by the `day_progression_function`. This
    function does not do any checks. They are done in `simulate_discrete_model`.

    Keyword arguments:
    :param nsim: number of simulations
    :param initial_colonized: array of colonized individuals in day 0 (1st column). Colonized = 1 else = 0.
        Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
    :param wards: array of the wards each of the colonized individuals corresponds to. Indexed by day
        (1st column). Individuals are rows.
    :param new_arrivals: array of new arrival individuals indexed day (1st column). New arrival = 1 else = 0.
        Individuals are rows. Use multiple columns if you are evaluating several parameters at once.
    :param weights: array describing the percent of day spent by individual in ward. Indexed by day (1st column)
    :param total_patients_per_ward: Array of (ints) total number of patients at day `d`
    indexed by ward.
        - 0th column: `d` day
        - 1st column: `N` total number of patients
        - 2nd column: `ward` The ward for which the number of patients was computed
    :param parameters: Dict of arrays with parameters of the `ward_progression_function` model.
    :param day_progression_function: Function describing how to progress each day
    :param ward_progression_function: Function describing how to progress within ward.

    :return: array of colonized individuals across al days. Individuals are rows. Returns multiple
        columns if you are evaluating several parameters at once.

    Examples:
    ```
    #RUN ONE SET OF PARAMETERS
    np.random.seed(3245)
    num_obs = 7   #Number of patients
    num_days = 5  #Run model by 5 days
    num_wards = 4 #Total number of wards

    initial_colonized = np.random.binomial(n=1,p=0.4,size=(num_obs,1)) == 1
    new_arrivals = np.empty((num_days, num_obs, 1))
    weights = np.tile(1, new_arrivals.shape)
    wards = np.empty((num_days, num_obs))
    total_patients_per_ward = np.empty((0, 3), dtype = "int")

    for day in range(num_days):

        #Generate the data
        new_arrivals_temp = np.random.binomial(n=1,p=0.05,size=(7,1))
        new_arrivals[day,:,:] = new_arrivals_temp

        wards[day,:] = np.random.randint(1, 4, size = 7)

        minward = 20
        maxward = minward + 25
        total_patients_per_ward_temp = np.random.randint(minward, maxward, size = num_wards)
        total_patients_per_ward_temp = np.column_stack((np.repeat(day, num_wards),
                total_patients_per_ward_temp, np.arange(1, num_wards + 1)))
        total_patients_per_ward = np.append(total_patients_per_ward, total_patients_per_ward_temp, axis = 0)

    #Create the parameters
    parameters = numba.typed.Dict() # dict()
    parameters["alpha"] = 0.15
    parameters["beta"] = 0.24
    parameters["gamma"] = 0.2

    #Colonize nextstep
    model_simulation = simulate_discrete_model_internal_multiple(initial_colonized, wards, new_arrivals, weights,
                                                        total_patients_per_ward, parameters,
                                                        progress_patients_1_timestep,
                                                        progress_patients_probability_ward_1_timestep)

    #RUN MULTIPLE PARAMETERS
    np.random.seed(3245)
    num_obs = 7   #Number of patients
    num_days = 5  #Run model by 5 days
    num_wards = 4 #Total number of wards
    num_parameters = 10 #Total number of parameters to run at same time

    initial_colonized = np.random.binomial(n=1,p=0.4,size=num_obs) == 1
    initial_colonized = np.tile(initial_colonized, (num_parameters,1)).transpose()

    new_arrivals = np.empty((num_days, num_obs, num_parameters))

    wards = np.empty((num_days, num_obs))
    total_patients_per_ward = np.empty((0, 3), dtype = "int")

    for day in range(num_days):

        #Generate the data
        new_arrivals_temp = np.random.binomial(n=1,p=0.05,size=7)
        new_arrivals_temp = np.tile(new_arrivals_temp, (num_parameters,1)).transpose()
        new_arrivals[day,:,:] = new_arrivals_temp

        wards[day,:] = np.random.randint(1, 4, size = 7)

        minward = 20
        maxward = minward + 25
        total_patients_per_ward_temp = np.random.randint(minward, maxward, size = num_wards)
        total_patients_per_ward_temp = np.column_stack((np.repeat(day, num_wards),
                total_patients_per_ward_temp, np.arange(1, num_wards + 1)))
        total_patients_per_ward = np.append(total_patients_per_ward, total_patients_per_ward_temp, axis = 0)

    #Create weights
    weights = np.tile(1, new_arrivals.shape)

    #Create the parameters
    parameters = numba.typed.Dict() # dict()
    parameters["alpha"] = np.linspace(0.1, 0.2, num_parameters)
    parameters["beta"] = np.linspace(0.2, 0.3, num_parameters)
    parameters["gamma"] = np.linspace(0.2, 0.3, num_parameters)


    #Colonize nextstep
    model_simulation = simulate_discrete_model_internal_multiple(initial_colonized, wards, new_arrivals, weights, total_patients_per_ward,
                                                parameters, progress_patients_1_timestep,
                                                progress_patients_probability_ward_1_timestep)
    ```
    """
    # Loop through simulations
    simulations = np.empty(shape=(nsim,) + new_arrivals.shape)
    for n in numba.prange(nsim):
        simulations[n, :, :, :] = simulate_discrete_model_internal_one(initial_colonized, wards, new_arrivals,
                                                                       weights, total_patients_per_ward, parameters,
                                                                       day_progression_function,
                                                                       ward_progression_function)

    return simulations


# This one has no decorator of njit as it is a wrapper for `simulate_discrete_model_internal_multiple`
def simulate_discrete_model(initial_colonized, wards, total_patients_per_ward, parameters, nsim=100, new_arrivals=None,
                            weights=None, day_progression_function=None, ward_progression_function=None):
    """
    `simulate_discrete_model` progresses all patients in all wards from 1 timestep to the next
    across all days in a range.  The function is defined to progress each patient `i` of each ward `w` by
    computing their colonized C status on day `d` as specified by the `day_progression_function`. This
    function also checks for input errors.

    Keyword arguments:
    :param nsim: number of simulations to do of the model for each pair of parameters. As an example if
    you are setting 10 parameter-pairs and nsim = 3 then you will get 30 simulations: 3 simulations for each
    of the 10 parameter-pairs.
    :param initial_colonized: one dimensional array of colonized individuals in day 0 (1st column).
    Colonized = 1 else = 0. Individuals are rows.
    :param wards: array of the wards each of the colonized individuals corresponds to. Indexed by day
        (1st column). Individuals are rows.
    :param new_arrivals: array of new arrival individuals indexed day (1st column). New arrival = 1 else = 0.
        Individuals are rows. Use multiple columns if you are evaluating several parameters at once. By default, there
        are no new_arrivals unless user specifies them.
    :param weights: array describing the percent of day spent by individual in ward. Indexed by day (1st column).
        By default, all individuals have weight 1.
    :param total_patients_per_ward: Array of (ints) total number of patients at day `d` indexed by ward.
        - 0th column: `d` day
        - 1st column: `N` total number of patients
        - 2nd column: `ward` The ward for which the number of patients was computed
    :param parameters: Dict of arrays with parameters of the `ward_progression_function` model.
    :param day_progression_function: Function describing how to progress each day. By default, uses
            `progress_patients_1_timestep` with a ward function of `progress_patients_probability_ward_1_timestep`
    :param ward_progression_function: How are values inside the ward transmitted each day.

    :return: array of colonized individuals across al days. Individuals are rows. Returns multiple
        columns if you are evaluating several parameters at once.

    Examples:
    ```
    #RUN MULTIPLE PARAMETERS
    np.random.seed(3245)
    nsim = 100
    num_obs = 7   #Number of patients
    num_days = 5  #Run model by 5 days
    num_wards = 4 #Total number of wards
    num_parameters = 10 #Total number of parameters to run at same time

    initial_colonized = np.random.binomial(n=1,p=0.4,size=num_obs) == 1

    new_arrivals = np.empty((num_days, num_obs))

    wards = np.empty((num_days, num_obs))
    total_patients_per_ward = np.empty((0, 3), dtype = "int")

    for day in range(num_days):

        #Generate the data
        new_arrivals_temp = np.random.binomial(n=1,p=0.05,size=7)
        new_arrivals[day,:] = new_arrivals_temp

        wards[day,:] = np.random.randint(1, 4, size = 7)

        minward = 20
        maxward = minward + 25
        total_patients_per_ward_temp = np.random.randint(minward, maxward, size = num_wards)
        total_patients_per_ward_temp = np.column_stack((np.repeat(day, num_wards),
                total_patients_per_ward_temp, np.arange(1, num_wards + 1)))
        total_patients_per_ward = np.append(total_patients_per_ward, total_patients_per_ward_temp, axis = 0)

    #Create weights
    weights = np.tile(1, new_arrivals.shape)

    #Create the parameters
    parameters = numba.typed.Dict() # dict()
    parameters["alpha"] = np.linspace(0.1, 0.2, num_parameters)
    parameters["beta"] = np.linspace(0.2, 0.3, num_parameters)
    parameters["gamma"] = np.linspace(0.2, 0.3, num_parameters)


    #Colonize nextstep
    model_simulation = simulate_discrete_model(initial_colonized, wards, total_patients_per_ward, parameters, nsim,
                                                new_arrivals, weights, progress_patients_1_timestep,
                                                progress_patients_probability_ward_1_timestep)
    ```
    """

    # Check parameters have same dimension
    parameter_values = list(parameters.items())
    num_parameters = len(parameter_values[0][1])
    incorrect_params = [key for key in list(parameters) if len(parameters[key]) != num_parameters]
    if len(incorrect_params) > 0:
        raise ValueError(f"The following parameters have different dimensions: " +
                         f"'{parameter_values[0][0]}' and {incorrect_params}.")

    # Check that the ward and patient days are correctly specified
    total_days = wards.shape[0]
    day_patients = len(np.unique(total_patients_per_ward[:, 0]))
    if len(incorrect_params) > 0:
        raise ValueError(f"`wards` have: {total_days} days (1st column) and `total_patients_per_ward` have " +
                         f"{day_patients} days (1st column).")

    # Check that we are giving at least two individuals:
    num_obs = initial_colonized.shape[0]
    if num_obs < 2:
        raise ValueError(f"Model has only {num_obs} individuals. You need at least `2`.")

    # Create weights if no weights are inputed
    if weights is None:
        weights = np.ones(shape=(total_days, num_obs, 1))

    if weights.shape[0] != total_days or weights.shape[1] != num_obs:
        raise ValueError(f"Invalid dimension for `weights` should be {(total_days, num_obs)}")

    # Create new_arrivals if no new_arrivals are given
    if new_arrivals is None:
        new_arrivals = np.zeros(shape=(total_days, num_obs, 1))

    if new_arrivals.shape[0] != total_days or new_arrivals.shape[1] != num_obs:
        raise ValueError(f"Invalid dimension for `new_arrivals` should be {(total_days, num_obs)}")

    # Check that the progress functions are specified
    if day_progression_function is None:
        day_progression_function = progress_patients_1_timestep

    if ward_progression_function is None:
        ward_progression_function = progress_patients_probability_ward_1_timestep

    # Fill the new arrivals and initial colonized to fit the length required by model which is 3D, 3D and 2D arrays
    if len(new_arrivals.shape) == 2:
        new_arrivals = new_arrivals[:, :, np.newaxis]

    if len(weights.shape) == 2:
        weights = weights[:, :, np.newaxis]

    if len(initial_colonized.shape) == 1:
        initial_colonized = initial_colonized[:, np.newaxis]

    # Stack the arrivals and weights
    if weights.shape[2] == 1:
        weights = np.repeat(weights, num_parameters, axis=2)

    if new_arrivals.shape[2] == 1:
        new_arrivals = np.repeat(new_arrivals, num_parameters, axis=2)

    if initial_colonized.shape[1] == 1:
        initial_colonized = np.repeat(initial_colonized, num_parameters, axis=1)

    # Check that the number of individuals in new arrivals and in colonized are the same
    if initial_colonized.shape[0] != new_arrivals.shape[1]:
        raise ValueError(f"`new_arrivals` has {new_arrivals.shape[1]} individuals while" +
                         f" `initial_colonized` only has {initial_colonized.shape[0]}.")

    return simulate_discrete_model_internal_multiple(initial_colonized, wards, new_arrivals, weights,
                                                     total_patients_per_ward, parameters, day_progression_function,
                                                     ward_progression_function, nsim)

# This one has no decorator of njit as it is a wrapper for `simulate_discrete_model_internal_multiple`
def get_totals(model_simulation, parameters=None, nround=3):
    """
    :param model_simulation: An array from `simulate_discrete_model`
    return A pandas DataFrame of the totals by parameter and simulation group
    """

    total_positives = np.sum(model_simulation, axis = 2)

    df = pd.concat([pd.DataFrame(arr) for arr in total_positives], keys=np.arange(len(total_positives)))
    df = df.add_prefix('parameters_')
    df = df.reset_index()
    df = df.rename(columns={"level_0": "simulation"})
    df = df.rename(columns={"level_1": "day"})
    df = pd.wide_to_long(df, i=["simulation","day"],  stubnames="parameters_", j = "pars").reset_index()
    df = df.rename(columns={"parameters_": "colonized"})

    if parameters is not None:

        # Paste the parameter values to the column for ease of plotting
        df_params = pd.DataFrame(dict(parameters))
        cols = df_params.columns.values
        for val in cols:
            df_params[val] = val + ": " +  np.round(df_params[val], nround).astype(str)

        df_params = df_params.reset_index().rename(columns={"index": "pars"})

        df_params['parameters'] = df_params[cols].apply(lambda row: '|'.join(row.values), axis=1)
        df_params = df_params[["pars","parameters"]]
        df = df.join(df_params.set_index('pars'), on = "pars")

    return df
