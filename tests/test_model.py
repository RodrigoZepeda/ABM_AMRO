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
        [-1, -1, -1, -1, -1, -1],  # Next day
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [0.2],  # Beta
        [1.0],  # Gamma
    ]).transpose()
    model_run = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters,10)
    assert numpy.all(model_run[:, 3] == model_run[:, 6])

def test_sum_of_positives():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 0, 0, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1], #Next day position
        [0, 0, 0, 1, 0, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0],  # Colonized status under model 3
        [1, 0, 1, 1, 0, 0],  # Colonized status under model 4
    ]).transpose()

    assert numpy.all(amro.total_positive(ward_matrix) == numpy.array([1, 3, 4, 3]))

def test_summary_mean():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1], #Next day position
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 3
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 4
    ]).transpose()

    pos = amro.total_positive(ward_matrix)
    results = amro.summary_of_total_positive(ward_matrix, numpy.array([0.2, 0.9]))

    assert numpy.all(numpy.mean(pos, axis=1) == results[:,1])

def test_summary_sd():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1], #Next day position
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 3
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 4
    ]).transpose()

    pos = amro.total_positive(ward_matrix)
    results = amro.summary_of_total_positive(ward_matrix, numpy.array([0.2, 0.9]))

    assert numpy.all(numpy.std(pos, axis=1) == results[:,2])

def test_seed_equal():
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 1, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [0.2],  # Beta
        [0.5],  # Gamma
    ]).transpose()
    model_1 = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters, 10)
    model_2 = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters, 10)
    assert numpy.all(model_1 == model_2)

def test_seed_different():
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 1, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [1.0],  # Beta
        [0.5],  # Gamma
    ]).transpose()
    model_1 = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters, 1)
    model_2 = amro.simulate_discrete_model_internal_one(initial_colonized, ward_matrix, total_patients, parameters, 10)
    assert ~numpy.all(model_1 == model_2)

def test_seed_different_across_threads():
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 1, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [1.0],  # Beta
        [0.5],  # Gamma
    ]).transpose()
    model_1 = amro.simulate_discrete_model_internal(initial_colonized, ward_matrix, total_patients, parameters, 10, 1, 4)
    model_2 = amro.simulate_discrete_model_internal(initial_colonized, ward_matrix, total_patients, parameters, 10, 2, 4)
    assert ~numpy.all(model_1 == model_2)

""" def test_seed_equal_across_threads():
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 1, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
    ]).transpose()
    parameters = numpy.array([
        [0.1],  # Alpha
        [1.0],  # Beta
        [0.5],  # Gamma
    ]).transpose()
    model_1 = amro.simulate_discrete_model_internal(initial_colonized, ward_matrix, total_patients, parameters, 2, 2, 4)
    model_2 = amro.simulate_discrete_model_internal(initial_colonized, ward_matrix, total_patients, parameters, 10, 2, 4)
    assert numpy.all(model_1 == model_2)   """