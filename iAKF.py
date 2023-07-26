#Implements the iterated adjusted Kalman Filter
import numpy as np
from numba import njit

@njit
def iterated_ensemble_adjusted_kalman_filter():
    """
        `iterated_ensemble_adjusted_kalman_filter` implements the iterated adjusted Kalman filter
        as defined in: 'Evensen, Geir. Data assimilation: the ensemble Kalman filter.
        Vol. 2. Berlin: springer, 2009.'

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
