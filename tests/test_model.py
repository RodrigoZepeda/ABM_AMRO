import numpy
import amro


def test_gamma_equals_1_all_arrivals_infected():
    # Gamma = 1 should infect all new arrivals
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 0, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 0, 0, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [0, 0, 0, 0, 0, 0],  # Colonized status under model 1
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [0.2],  # Beta
        [1.0],  # Gamma
    ]).transpose()
    model_run = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters,
                                                          10)
    assert numpy.all(model_run[:, 3] == model_run[:, 6])
