import numpy
import amro
#There should be a test that if we detect all promptly then colonized should equal deected
def test_progress_patients_clearance_rate_100():
    """
    Tests the run of just one simulation when clearance rate is 100% to check no one remains colonized
    """
    nf = numpy.inf
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 20, 87, 99, 12, 43, 59, 98],  # MRN
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Next day
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1],  # Colonized
        [nf, 4, nf, nf, 1, nf, nf, 0, 0, nf],  # Detected
    ]).transpose()
    total_patients = 100
    parameters = numpy.array([
        [1.0],  # Alpha
        [0.0],  # Beta
        [0.0],  # Gamma
        [0.12],  # Rho
        [1.0],  # New alpha
        [0.3],  # Rho in hospital
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all(pstep[:, 6] == 0) & numpy.all(pstep[:, 7] == numpy.inf)

def test_progress_patients_clearance_rate_0():
    """
    Tests the run of just one simulation when clearance rate is 0% to check all remain colonized
    """
    nf = numpy.inf
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 20, 87, 99, 12, 43, 59, 98],  # MRN
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # Next day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Colonized
        [nf, 4, nf, nf, 1, nf, nf, 0, 0, nf],  # Detected
    ]).transpose()
    total_patients = 100
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [0.0],  # Gamma
        [0.12],  # Rho
        [0.0],  # New alpha
        [0.1],  # Rho in hospital
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all(pstep[:, 6] == 1)

def test_progress_patients_clearance_rho_hospital_100():
    """
    Tests the run of just one simulation when detection rate is 100% to check all colonized are detected
    for hospitalized cases
    """
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [0, 1, 1, 0, 0, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day
        [0, 0, 0, 1, 1, 1],  # Colonized
        [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf],  # Detected
    ]).transpose()
    total_patients = 6
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [0.0],  # Gamma
        [1.0],  # Rho
        [0.0],  # New alpha
        [0.0],  # Rho for imported
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all(pstep[3:,7] == 0)

def test_progress_patients_clearance_rho_hospital_0():
    """
    Tests the run of just one simulation when detection rate is 0% to check all colonized are not detected
    for hospitalized cases
    """
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [0, 1, 1, 0, 0, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day
        [0, 1, 1, 0, 1, 1],  # Colonized
        [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf],  # Detected
    ]).transpose()
    total_patients = 6
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [0.0],  # Gamma
        [0.0],  # Rho
        [0.0],  # New alpha
        [0.0],  # Rho for imported
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all(pstep[:, 7] == numpy.inf)

def test_progress_patients_clearance_rho_imported_100():
    """
    Tests the run of just one simulation when detection rate is 100% to check all colonized are detected
    for hospitalized cases
    """
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 0, 0, 0, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day
        [0, 0, 0, 0, 0, 0],  # Colonized
        [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf],  # Detected
    ]).transpose()
    total_patients = 6
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [1.0],  # Gamma
        [0.0],  # Rho
        [0.0],  # New alpha
        [1.0],  # Rho for imported
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all((pstep[:, 6] == 1) == (pstep[:,7] <= 0)) and numpy.all(pstep[:, 6] == pstep[:, 3])

def test_progress_patients_clearance_rho_imported_0():
    """
    Tests the run of just one simulation when detection rate is 0% to check all colonized are not detected
    for new arrivals
    """
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 1, 1, 0, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day
        [0, 0, 0, 0, 0, 0],  # Colonized
        [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf],  # Detected
    ]).transpose()
    total_patients = 6
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [1.0],  # Gamma
        [0.0],  # Rho
        [0.0],  # New alpha
        [0.0],  # Rho for imported
    ]).transpose()
    n_sims = 1
    pstep = amro.progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters, n_sims, 0, 1, 1)
    assert numpy.all(pstep[:, 7] == numpy.inf) and numpy.all(pstep[:, 6] == pstep[:, 3]) #All imported colonized but none detected

def test_gamma_equals_1_all_arrivals_infected():
    """
    Tests that all arrivals are infected if gamma = 100%
    """
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[0, 0, 0, 0, 0, 0]]).transpose()
    initial_detected = numpy.array([[0, 0, 0, 0, 0, 0]]).transpose()
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
        [0.3],  # Rho for cases in hospital
        [0.2],  # New alpha
        [0.3],  # Rho for new cases
    ]).transpose()
    model_run = amro.simulate_discrete_model(initial_colonized, initial_detected, ward_matrix, total_patients, parameters, 10)
    assert numpy.all(model_run[:, 3] == model_run[:, 6])

def test_detected_persistance():
    """
    Tests that if patient was detected as colonized they remain detected until they leave the hospital or
    get clearance
    """
    total_patients = numpy.array([
        [0,1,2], #Day
        [1,1,1], #Ward
        [20,10,12], #Next day
    ]).transpose()
    initial_colonized = numpy.array([[1, 0, 1, 0, 0, 1]]).transpose()
    initial_detected = numpy.array([[1, 0, 0, 0, 0, 0]]).transpose()
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 200, 99, 52, 53, 17, 200, 99],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, 9, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1],  # Next day
    ]).transpose()
    parameters = numpy.array([
        [0.0],  # Alpha
        [0.0],  # Beta
        [0.0],  # Gamma
        [1.0],  # Rho
        [0.0],  # New alpha
        [0.3],  # Rho for new cases
    ]).transpose()
    model_run = amro.simulate_discrete_model(initial_colonized, initial_detected, ward_matrix, total_patients, parameters, 0, 1, 1)
    assert numpy.all((model_run[:, 6] == 1) == (model_run[:, 7] <= 0))

def test_initial_detected_equals_1_all_colonized_detected():
    """
    Tests that if rho = 1 then all colonized individuals show up as infected
    """
    total_patients = numpy.array([
        [0],
        [1],
        [20],
    ]).transpose()
    initial_colonized = numpy.array([[1, 0, 1, 0, 0, 1]]).transpose()
    initial_detected = numpy.array([[1, 1, 1, 1, 1, 1]]).transpose()
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
        [0.3],  # Gamma
        [1.0],  # Rho
        [0.2],  # New alpha
        [1.0],  # Rho for new cases
    ]).transpose()
    model_run = amro.simulate_discrete_model(initial_colonized, initial_detected, ward_matrix, total_patients, parameters, 0, 1, 1)
    assert numpy.all((model_run[:, 7]  <= 0)== (model_run[:, 6] == 1))

def test_sum_of_positives_colonized():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 0, 0, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day position
        [0, 0, 0, 1, 0, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0],  # Colonized status under model 3
        [1, 0, 1, 1, 0, 0],  # Colonized status under model 4
    ]).transpose()

    assert numpy.all(amro.total_positive_colonized(ward_matrix) == numpy.array([1, 3]))

def test_sum_of_positives_detected():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0],  # Day
        [1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99],  # MRN
        [1, 1, 0, 0, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1],  # Weight
        [-1, -1, -1, -1, -1, -1],  # Next day position
        [numpy.inf, numpy.inf, numpy.inf, -1, numpy.inf, numpy.inf],  # Colonized status under model 1
        [-3, numpy.inf, -2, -1, numpy.inf, numpy.inf],  # Colonized status under model 2
        [-1, 0, 0, -1, numpy.inf, numpy.inf],  # Colonized status under model 3
        [-1, numpy.inf, -1, -4, numpy.inf, numpy.inf],  # Colonized status under model 4
    ]).transpose()

    assert numpy.all(amro.total_positive_detected(ward_matrix) == numpy.array([4,3]))

def test_summary_mean():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 3
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 4
    ]).transpose()

    pos = amro.total_positive_colonized(ward_matrix, 4)
    results = amro.summary_of_total_positive(pos, numpy.array([0.2, 0.9]))

    assert numpy.all(numpy.mean(pos, axis=1) == results[:, 1])

def test_summary_sd():
    ward_matrix = numpy.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Day
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Ward
        [52, 53, 17, 200, 87, 99, 52, 53, 17, 20],  # MRN
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # New_arrival
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Weight
        [6, 7, 8, -1, -1, -1, -1, -1, -1, 1],  # Next day position
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 1
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 2
        [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 3
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # Colonized status under model 4
    ]).transpose()

    pos = amro.total_positive_colonized(ward_matrix, 4)
    results = amro.summary_of_total_positive(pos, numpy.array([0.2, 0.9]))

    assert numpy.all(numpy.std(pos, axis=1) == results[:, 2])

def test_seed_equal():
    """
    Tests that same seeds generate same results
    """
    total_patients = numpy.array([
        [0,1],
        [1,1],
        [20,12],
    ]).transpose()
    initial_colonized = numpy.array([[0, 1, 0, 0, 0, 0]]).transpose()
    initial_detected = numpy.array([[0, 0.5, 0, 0, 0, 0]]).transpose()
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
        [0.8],  # Rho
        [0.2],  # New alpha
        [0.3],  # Rho for new cases
    ]).transpose()
    model_1 = amro.simulate_discrete_model(initial_colonized, initial_detected, ward_matrix, total_patients, parameters, 1, 1, 1, 34279)
    model_2 = amro.simulate_discrete_model(initial_colonized, initial_detected, ward_matrix, total_patients, parameters, 1, 1, 1, 34279)
    assert numpy.all(model_1 == model_2)
