#include <carma>
#include <armadillo>
#include <cstdio>
#include <progressbar.hpp>
#include <pybind11/iostream.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_num_threads() 0
    inline void omp_set_num_threads(int num_threads) { static_cast<void>(num_threads); }
    #define omp_get_thread_num() 0
#endif

/**
@title Progress patients across one ward

@description Progresses the patients that belong to the same ward one timestep. Consider
a group of individuals in the same ward. Let `C_i(t)` denote whether individual `i` is colonized
`C_i(t) = 1` or not colonized `C_i(t) = 0` at time `t`. The probability of an individual being colonized
at time `t + 1` is given by:

     [(1 - detected)*(1 - alpha) + detected*(1 - alpha2)]*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

Once an individual is colonized there is a certain probability that they will be detected. If the
individual is an imported colonized case they will be detected with probability `rho_imported` and
if they are a colonized case that was already hospitalized they will be detected with probability
`rho_hospital`.

Furthermore  if they are detected then alpha can be substituted by alpha2 with alpha2 being a different
clearance probability.

@param ward_matrix  A `ward_matrix` corresponding to the state of the ward at time $t$. The matrix should
include the following columns:

Column 0: For this function it can be anything; however see `progress_patients_1_timestep`.
Column 1: For this function it can be anything; however see `progress_patients_1_timestep`.
Column 2: Indicator for new arrivals (= 1) and individuals that were already here (= 0)
Column 3: Weight w_i of the individual in the ward.
Column 4: For this function it can be anything; however see `simulate_discrete_model_internal_one`.
Column 5 to `n_sims + 5`: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases and 0's correspond to not colonized individuals.
Column `n_sims + 6` to last column: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases THAT WERE DETECTED and 0's correspond to UNDETECTED individuals
(either not colonized or colonized but not detected)

@param total_patients The total number of patients in the current ward at time `t` (current time).

@param parameters A matrix with columns corresponding to the model's parameters.

Column 0: `alpha` the clearance probability for the undetected.
Column 1: `beta` the force of infection.
Column 2: `gamma` probability of an imported case being colonized.
Column 3: `rho_hospital` the probability of being detected for a hospitalized case.
Column 4: `alpha_2` the clearance probability for the detected.
Column 5: `rho_imported` the probability of being detected for an imported case.

@param n_sims Number of simulations currently involved in the process.

@param time_to_detect Time (integer) it takes for a detected colonized to receive their results

@param detect_today_hospitalized (boolean) Whether individuals that were hospitalized in the previous step
are tested in the current timestep or not.

@param detect_today_arrival (boolean) Whether individuals that were imported in the step
are tested or not.

@return A `ward_matrix` corresponding to the state of the ward at time `t + 1`
*/
arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix,
                                                        const double& total_patients,
                                                        const arma::mat& parameters,
                                                        const arma::uword n_sims,
                                                        const int& time_to_detect,
                                                        const bool& detect_today_hospitalized,
                                                        const bool& detect_today_arrival) {

  //Location of parameters in the parameter matrix.
  const arma::uword alpha_col      = 0; //Probability of clearance
  const arma::uword beta_col       = 1; //Infection rate for the hospitalized cases
  const arma::uword gamma_col      = 2; //Probability of colonized in imported cases
  const arma::uword rho_col_hosp   = 3; //Probability of detection for hospitalized cases
  const arma::uword alpha_col_2    = 4; //New probability of clearance if detected
  const arma::uword rho_col_import = 5; //Probability of detection for the imported cases

  //Information location in the ward matrix:
  const arma::uword arrivals_col = 3;                                       //Column in ward_matrix marking if new arrival = 1 or was already here = 0
  const arma::uword weights_col = 4;                                        //Column in ward_matrix with observation weights (between 0 and 1)
  const arma::uword colonized_col_init = 6;                                 //Column where the colonized individuals start. From here on every col is a different simulation
  const arma::uword colonized_col_end  = n_sims + (colonized_col_init - 1); //Column where the colonized individuals end

  for (arma::uword col_index = colonized_col_init; col_index <= colonized_col_end; ++col_index) {

    // Weight the colonized part of the matrix W = weights*C
    ward_matrix.col(col_index) %= ward_matrix.col(weights_col);

    // Get the force of infection, F = beta/N * sum(W)
    double force_of_infection = (parameters(col_index - colonized_col_init, beta_col) / total_patients) *
        arma::sum(ward_matrix.col(col_index));

    // Get the coeficient of multiplication coef = (1 - alpha)*(1 - detected) + (1 - alpha2)*detected
    arma::vec detected = arma::conv_to<arma::vec>::from(ward_matrix.col(col_index + n_sims) <= 0);
    arma::vec coef =
        (1 - parameters(col_index - colonized_col_init, alpha_col)) * (1 - detected) +
        (1 - parameters(col_index - colonized_col_init, alpha_col_2)) * detected;

    // Get the cases inside the ward: ward_attributable =  coef*W + (1 - W)*F
    ward_matrix.col(col_index) = coef % ward_matrix.col(col_index) +
        force_of_infection * (1 - ward_matrix.col(col_index));

    // Compute colonized probability: colonized_probability = ward_attributable + gamma*h[i,d]
    ward_matrix.col(col_index) += parameters(col_index - colonized_col_init, gamma_col) * ward_matrix.col(arrivals_col);

    // Simulate the probability of colonization and input it into the ward matrix
    for (arma::uword row_index = 0; row_index < ward_matrix.n_rows; ++row_index) {
      if (arma::randu() < ward_matrix(row_index, col_index)) {
        ward_matrix(row_index, col_index) = 1;
      } else {
        ward_matrix(row_index, col_index) = 0;
      }

      // Simulate the probability of detection in those colonized if they are colonized but haven't been detected
      // The idea is that tests are performed at this step but there is a time to detect (i.e. test results take time)
      // which is how much time it takes for a detected case to actually appear in the dataset. So the model
      // either assigns time_to_detect if it is going to be detected, infinity if it is never going to be detected
      // or a negative number if it is already detected.

      // * Colonized and not imported:
      if ((ward_matrix(row_index, col_index) == 1) && (ward_matrix(row_index, arrivals_col) == 0)){
        // Check whether it has already been tested / detected but results haven't come yet.
        if (ward_matrix(row_index, col_index + n_sims) <= time_to_detect){
            ward_matrix(row_index, col_index + n_sims) -= 1;
        // Else we simulate a detection with probability rho_col_hosp if today is a day for testing
        } else if (detect_today_hospitalized && (arma::randu() <= parameters(col_index - colonized_col_init, rho_col_hosp))) {
            ward_matrix(row_index, col_index + n_sims) = time_to_detect;
        // Else the individual has not been detected today
        } else {
            ward_matrix(row_index, col_index + n_sims) = arma::datum::inf;
        }
      // * Colonized and imported cases:
      } else if ((ward_matrix(row_index, col_index) == 1) && (ward_matrix(row_index, arrivals_col) == 1)){
        // Check whether it has already been tested but results haven't come yet.
        if (ward_matrix(row_index, col_index + n_sims) <= time_to_detect){
            ward_matrix(row_index, col_index + n_sims) -= 1;
        // Else we simulate a detection with probability rho_col_import if today is a day for detecting
        } else if (detect_today_arrival && (arma::randu() <= parameters(col_index - colonized_col_init, rho_col_import))) {
            ward_matrix(row_index, col_index + n_sims) = time_to_detect;
        // Else the individual was not detected today
        } else {
            ward_matrix(row_index, col_index + n_sims) = arma::datum::inf;
        }
      // * Not colonized anymore thus we remove detection:
      } else {
            ward_matrix(row_index, col_index + n_sims) = arma::datum::inf;
      }
    }
  }
  return ward_matrix;
}

/**
@title Progress patients across all wards

@description Progresses all the patients across all wards one timestep. Consider
a group of individuals in the same ward. Let `C_i(t)` denote whether individual `i` is colonized
`C_i(t) = 1` or not colonized `C_i(t) = 0` at time `t`. The probability of an individual being colonized
at time `t + 1` is given by:

    [(1 - detected)*(1 - alpha) + detected*(1 - alpha2)]*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

Once an individual is colonized there is a certain probability that they will be detected. If the
individual is an imported colonized case they will be detected with probability `rho_imported` and
if they are a colonized case that was already hospitalized they will be detected with probability
`rho_hospital`.

Furthermore  if they are detected then alpha can be substituted by alpha2 with alpha2 being a different
clearance probability.

@param ward_matrix  A `ward_matrix` corresponding to the state of the ward at time $t$. The matrix should
include the following columns:

Column 1: Unique identifier for the ward.
Column 2: Key that indicates where in the total_patients_per_ward matrix the ward identified in column 1 is.
Column 3: Indicator for new arrivals (= 1) and individuals that were already here (= 0)
Column 4: Weight w_i of the individual in the ward.
Column 5: For this function it can be anything; however see `simulate_discrete_model_internal_one`.
Column 6 to `n_sims + 5`: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases and 0's correspond to not colonized individuals.
Column `n_sims + 6` to last column: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases THAT WERE DETECTED and 0's correspond to UNDETECTED individuals
(either not colonized or colonized but not detected)

@param total_patients_per_ward A matrix with two columns:

Column 1: The number of the ward (unique identifier).
Column 2: The number of patients in that ward.

@param parameters A matrix with columns corresponding to the model's parameters.

Column 0: `alpha` the clearance probability for the undetected.
Column 1: `beta` the force of infection.
Column 2: `gamma` probability of an imported case being colonized.
Column 3: `rho_hospital` the probability of being detected for a hospitalized case.
Column 4: `alpha_2` the clearance probability for the detected.
Column 5: `rho_imported` the probability of being detected for an imported case.

@param n_sims Number of simulations currently involved in the process.

@param time_to_detect Time (integer) it takes for a detected colonized to receive their results

@param detect_today_hospitalized (boolean) Whether individuals that were hospitalized in the previous step
are tested in the current timestep or not.

@param detect_today_arrival (boolean) Whether individuals that were imported in the step
are tested or not.

@return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`
*/
arma::mat progress_patients_1_timestep(arma::mat& ward_matrix,
                                       const arma::mat& total_patients_per_ward,
                                       const arma::mat& parameters,
                                       const arma::uword n_sims,
                                       const int& time_to_detect,
                                       const bool& detect_today_hospitalized,
                                       const bool& detect_today_arrival) {

  //In ward_matrix the ward is in `ward_col` column
  const arma::uword ward_col_wards    = 1;
  const arma::uword ward_col_patients = 1;

  //In `total_patients_per_ward` matrix the size of the ward is in `patient_size_col` column
  const arma::uword patient_size_col = 2;

  //Get unique ward values for the day
  arma::vec active_wards = arma::unique(ward_matrix.col(ward_col_wards));

  // Loop through the wards that have cases during the period
  for (arma::uword w = 0; w < active_wards.n_elem; ++w) {

    //Get the index of the ward
    arma::uvec ward_index    = arma::find(ward_matrix.col(ward_col_wards) == active_wards[w]);
    arma::uvec patient_index = arma::find(total_patients_per_ward.col(ward_col_patients) == active_wards[w]);

    //Get the patients and ward submatrix
    arma::mat total_patients = total_patients_per_ward.rows(patient_index);
    arma::mat ward_temp      = ward_matrix.rows(ward_index);

    //Apply the function
    progress_patients_probability_ward_1_timestep(ward_temp, total_patients(0, patient_size_col),
        parameters, n_sims, time_to_detect, detect_today_hospitalized, detect_today_arrival);

    //Update results
    ward_matrix.rows(ward_index) = ward_temp;

  }
  return ward_matrix;
}

/**
@title Simulate discrete model

@description Simulates all the patients across all wards across all timesteps. Consider
a group of individuals in the same ward. Let `C_i(t)` denote whether individual `i` is colonized
`C_i(t) = 1` or not colonized `C_i(t) = 0` at time `t`. The probability of an individual being colonized
at time `t + 1` is given by:

     [(1 - detected)*(1 - alpha) + detected*(1 - alpha2)]*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

Once an individual is colonized there is a certain probability that they will be detected. If the
individual is an imported colonized case they will be detected with probability `rho_imported` and
if they are a colonized case that was already hospitalized they will be detected with probability
`rho_hospital`.

Furthermore  if they are detected then alpha can be substituted by alpha2 with alpha2 being a different
clearance probability.

@param initial_colonized_probability A matrix describing the initial state of the model. All simulations are run with the same 
`initial_colonized_probability`.               

@param ward_matrix  A `ward_matrix` corresponding to the state of the ward at time $t$. The matrix should
include the following columns:

Column 1: Unique identifier for the ward.
Column 2: Key that indicates where in the total_patients_per_ward matrix the ward identified in column 1 is. 
Column 3: Indicator for new arrivals (= 1) and individuals that were already here (= 0)
Column 4: Weight w_i of the individual in the ward.
Column 5: Location (row) where the same patient appears in the dataset the following day. 
Column 6 to Last column: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases and 0's correspond to not colonized individuals. You can add as many columns as your memory allows. 

@param total_patients_per_ward A matrix with columns:

Column 0: The day of the observation.
Column 1: The number of the ward (unique identifier).
Column 2: The number of patients in that ward. 

@param parameters A matrix with columns corresponding to the model's parameters.

Column 0: `alpha` the clearance probability for the undetected.
Column 1: `beta` the force of infection.
Column 2: `gamma` probability of an imported case being colonized.
Column 3: `rho_hospital` the probability of being detected for a hospitalized case.
Column 4: `alpha_2` the clearance probability for the detected.
Column 5: `rho_imported` the probability of being detected for an imported case.

@param n_sims Number of simulations currently involved in the process.

@param time_to_detect Time (integer) it takes for a detected colonized to receive their results

@param testing_schedule_hospitalized (integer) Time it takes to test. Hospitalized individuals are only tested
every `testing_schedule` days. For example if  `testing_schedule = 7` then individuals are tested every week
(7 days). To test individuals daily set `testing_schedule = 1`.

@param testing_schedule_arrivals (integer) Time it takes to test. Newly arrived individuals are only tested
every `testing_schedule` days. For example if  `testing_schedule = 7` then individuals are tested every week
(7 days). To test individuals daily set `testing_schedule = 1`.

@param seed Random seed for `arma_rng`. 

@return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`
*/
arma::mat simulate_discrete_model_internal_one(const arma::mat& initial_colonized_probability,
                                               const arma::mat& initial_detected_probability,
                                               arma::mat& ward_matrix,
                                               const arma::mat& total_patients_per_ward,
                                               const arma::mat& parameters,
                                               const int& time_to_detect,
                                               const int& testing_schedule_hospitalized,
                                               const int& testing_schedule_arrivals,
                                               const int seed){

  //Set seed for the simulation
  arma::arma_rng::set_seed(seed);

  const arma::uword day_col_wards = 0;       //Column for days in ward_matrix
  const arma::uword day_col_patients = 0;    //Column for days in total_patients_per_ward
  const arma::uword next_day_col = 5;        //Column for next day registry of patient in ward matrix
  const arma::uword colonized_col_init = 6;  //Column indicating where the initial colonized start

  //Get the total number of simulations from the number of columns in initial colonized
  const arma::uword n_sims = initial_colonized_probability.n_cols;

  //Create an indicator vector of which are the columns that keep information on colonization
  arma::uvec colonized_cols(n_sims);
  for (arma::uword col = 0; col < n_sims; ++col){
    colonized_cols(col) = colonized_col_init + col;
  }

  //Create an indicator vector of which are the columns that keep information on detection
  arma::uvec detected_cols(n_sims);
  for (arma::uword col = 0; col < n_sims; ++col){
    detected_cols(col) = colonized_col_init + n_sims + col;
  }

  //Get the total amount of days
  arma::uword total_days = arma::max(total_patients_per_ward.col(day_col_patients));

  //Create initial matrix for results and initialize column with oo value
  arma::mat model_colonized(ward_matrix.n_rows, ward_matrix.n_cols + 2*n_sims);
  model_colonized.cols(detected_cols).fill(arma::datum::inf);

  //Get initial probability of being colonized
  arma::mat uniform_mat_colonized(initial_colonized_probability.n_rows, n_sims, arma::fill::randu);               //Simulate random numbers
  arma::mat is_colonized = arma::conv_to<arma::mat>::from(uniform_mat_colonized < initial_colonized_probability); //Keep random numbers with probability initial_colonized
  model_colonized.submat(0, colonized_col_init, initial_colonized_probability.n_rows - 1,                         //Substitute those colonized into initial matrix
                         colonized_col_init + n_sims - 1) = is_colonized;

  //Get initial probability of being detected given colonized
  arma::mat uniform_mat_detected(initial_detected_probability.n_rows, n_sims, arma::fill::randu);                 //Simulate random numbers
  arma::mat is_detected = arma::conv_to<arma::mat>::from(uniform_mat_detected < initial_detected_probability); //Keep random numbers with probability initial_detected
  model_colonized.submat(0, colonized_col_init + n_sims, initial_colonized_probability.n_rows - 1,
                         model_colonized.n_cols - 1) = is_detected % is_colonized; //Return only those colonized as detected

  //Get the other values
  model_colonized.submat(0, 0, model_colonized.n_rows - 1, colonized_col_init - 1) = ward_matrix;

  //Initialize colonization schedule
  bool test_hospitalized;
  bool test_arrivals;

  for(arma::uword day = 0; day <= total_days; ++day) {

    //Get the total amount of patients in that day
    arma::mat total_patients_per_day = total_patients_per_ward.rows(arma::find(total_patients_per_ward.col(day_col_patients) == day));

    //Get the ward info for the day
    arma::uvec model_colonized_idx = arma::find(model_colonized.col(day_col_wards) == day);
    arma::mat ward_per_day = model_colonized.rows(model_colonized_idx);

    //Calculate whether we are testing or not
    test_hospitalized = (day % testing_schedule_hospitalized) == 0;
    test_arrivals = (day % testing_schedule_arrivals) == 0;

    //Get the next timestep of the model
    progress_patients_1_timestep(ward_per_day, total_patients_per_day, parameters, n_sims,
            time_to_detect, test_hospitalized, test_arrivals);

    //Update current colonized values
    model_colonized.rows(model_colonized_idx) = ward_per_day;

    //Fill the values of the next step (temporarily)
    arma::mat active_wards_next = ward_per_day.rows(arma::find(ward_per_day.col(next_day_col) > 0));
    arma::uvec active_columns   = arma::conv_to<arma::uvec>::from(active_wards_next.col(next_day_col));

    //Persistance of colonized and detected in next snapshot
    model_colonized.submat(active_columns, colonized_cols) = active_wards_next.cols(colonized_cols);
    model_colonized.submat(active_columns, detected_cols)  = active_wards_next.cols(detected_cols);


  }

  return model_colonized;

}

/*
@title Obtain daily number of 1's

@description Given a simulation created with `simulate_discrete` this function obtains the total number of
1s between `col_init` and `col_end`.

@details To use internally in total_positive_colonized and total_positive_detected

@param model_colonized A matrix obtained from a `simulate_discrete_model` object

@param col_init First column with 1's

@param col_end Last column with 1's

@param mode Whether we check <= 0 (mode = 0) or we check == 1 (mode = 1)

@return A matrix with days being each rows and columns being the number of simulations. Each
matrix entry corresponds to the total number of 1's in that day/simulation.
*/
arma::mat total_positive(const arma::mat& model_colonized, const arma::uword col_init, const arma::uword col_end,
    const int& mode){

  //Get the total number of days
  const arma::uword day_column = 0;
  arma::uword total_days = arma::max(model_colonized.col(day_column)) + 1;

  arma::mat model_positive(total_days, col_end - col_init + 1);

  //Loop through days and sum
  for(arma::uword day = 0; day < total_days; ++day) {
    //Get the day
    arma::mat subcol = model_colonized.rows(arma::find(model_colonized.col(day_column) == day));

    // Perform equality check and convert to numeric matrix
    arma::mat binary_matrix = arma::conv_to<arma::mat>::from(subcol.cols(col_init, col_end));
    if (mode == 0){
        binary_matrix = arma::conv_to<arma::mat>::from(subcol.cols(col_init, col_end) <= 0);
    } else if (mode == 1){
        binary_matrix = arma::conv_to<arma::mat>::from(subcol.cols(col_init, col_end) == 1);
    }

    // Sum the binary matrix
    model_positive.row(day) = arma::sum(binary_matrix, 0);

  }

  return model_positive;

}

/*
@title Obtain daily number of colonized

@description Given a simulation created with `simulate_discrete` this function obtains the total number of
colonized cases per simulation and per day. 

@param model_colonized A matrix obtained from a `simulate_discrete_model` object

@return A matrix with days being each rows and columns being the number of simulations. Each
matrix entry corresponds to the total number of colonized cases in that day/simulation. 
*/
arma::mat total_positive_colonized(const arma::mat& model_colonized, const arma::uword n_sims){

  //Column for days in ward_matrix
  const arma::uword colonized_col_init = 6;
  const arma::uword colonized_col_end = colonized_col_init + n_sims - 1;

  return total_positive(model_colonized, colonized_col_init, colonized_col_end, 1);

}

/*
@title Obtain daily number of detected

@description Given a simulation created with `simulate_discrete` this function obtains the total number of
detected colonized cases per simulation and per day.

@param model_colonized A matrix obtained from a `simulate_discrete_model` object

@return A matrix with days being each rows and columns being the number of simulations. Each
matrix entry corresponds to the total number of detect colonized cases in that day/simulation.
*/

arma::mat total_positive_detected(const arma::mat& model_colonized, const arma::uword n_sims){

  //Column for days in ward_matrix
  const arma::uword detected_col_init = 6 + n_sims;
  const arma::uword detected_col_end = model_colonized.n_cols - 1;

  //Get the total number of positive
  return total_positive(model_colonized, detected_col_init, detected_col_end, 0);

}

/*
@title Summarize daily number of colonized across all simulations

@description Given a simulation created with `simulate_discrete` this function obtains the quantiles
specified in `quantiles`

@param model_colonized A matrix obtained from a `simulate_discrete_model` object
@param quantiles A vector of the quantiles to calculate for the summary. 

@return A matrix with the following columns:
Column 0: The day (each row is a day and this is the day number). 
Column 1: The mean number of cases. 
Column 2: The standard deviation on the number of cases. 
Column 3 to Last: Any quantiles specified in the quantiles parameter.  
*/
arma::mat summary_of_total_positive(const arma::mat& model_positive, const arma::vec quantiles){

  //Create matrix of quantiles, mean and sd
  arma::mat positive_summary(model_positive.n_rows, quantiles.n_elem + 3);

  for(arma::uword day = 0; day < model_positive.n_rows; ++day) {
    positive_summary(day, 0) = day;
  }
  positive_summary.col(1) = arma::mean(model_positive, 1);
  positive_summary.col(2) = arma::stddev(model_positive, 1, 1);
  positive_summary.submat(0, 3, model_positive.n_rows - 1,
                          positive_summary.n_cols - 1) = arma::quantile(model_positive, quantiles, 1);

  return positive_summary;
}
