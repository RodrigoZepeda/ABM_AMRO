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

                (1 - alpha)*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

@param ward_matrix  A `ward_matrix` corresponding to the state of the ward at time $t$. The matrix should
include the following columns:

Column 1: For this function it can be anything; however see `progress_patients_1_timestep`.
Column 2: For this function it can be anything; however see `progress_patients_1_timestep`.
Column 3: Indicator for new arrivals (= 1) and individuals that were already here (= 0)
Column 4: Weight w_i of the individual in the ward.
Column 5: For this function it can be anything; however see `simulate_discrete_model_internal_one`.
Column 6 to Last column: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases and 0's correspond to not colonized individuals. You can add as many columns as your memory allows. 

@param total_patients The total number of patients in the current ward at time $t$ (current time).

@param parameters A matrix with three columns corresponding to the model's three parameters. The
first column corresponds to `alpha`, the second to `beta` and the third to `gamma`.

@return A `ward_matrix` corresponding to the state of the ward at time `t + 1`
*/
arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix,
                                                        const double& total_patients,
                                                        const arma::mat& parameters) {

  //Location of parameters in the parameter matrix.
  const arma::uword alpha_col = 0;
  const arma::uword beta_col  = 1;
  const arma::uword gamma_col = 2;

  //Information location in the ward matrix:
  const arma::uword arrivals_col = 3;
  const arma::uword weights_col = 4;
  const arma::uword colonized_col_init = 6;
  const arma::uword colonized_col_end = ward_matrix.n_cols;

  for (arma::uword col_index = colonized_col_init; col_index < colonized_col_end; ++col_index) {

    // Weight the colonized part of the matrix W = weights*C
    ward_matrix.col(col_index) %= ward_matrix.col(weights_col);

    // Get the force of infection, F = beta/N * sum(W)
    double force_of_infection = (parameters(col_index - colonized_col_init, beta_col) / total_patients) * arma::sum(ward_matrix.col(col_index));

    // Get the cases inside the ward: ward_attributable = (1 - alpha)*W + (1 - W)*F
    ward_matrix.col(col_index) = (1 - parameters(col_index - colonized_col_init, alpha_col)) * ward_matrix.col(col_index) + force_of_infection * (1 - ward_matrix.col(col_index));

    // Compute colonized probability: colonized_probability = ward_attributable + gamma*h[i,d]
    ward_matrix.col(col_index) += parameters(col_index - colonized_col_init, gamma_col) * ward_matrix.col(arrivals_col);

    // Simulate the probability of colonization and input it into the ward matrix
    for (arma::uword row_index = 0; row_index < ward_matrix.n_rows; ++row_index) {
      if (arma::randu() < ward_matrix(row_index, col_index)) {
        ward_matrix(row_index, col_index) = 1;
      } else {
        ward_matrix(row_index, col_index) = 0;
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

                (1 - alpha)*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

@param ward_matrix  A `ward_matrix` corresponding to the state of the ward at time $t$. The matrix should
include the following columns:

Column 1: Unique identifier for the ward.
Column 2: Key that indicates where in the total_patients_per_ward matrix the ward identified in column 1 is. 
Column 3: Indicator for new arrivals (= 1) and individuals that were already here (= 0)
Column 4: Weight w_i of the individual in the ward.
Column 5: For this function it can be anything; however see `simulate_discrete_model_internal_one`.
Column 6 to Last column: Each of these columns correspond to a different simulation with 1's and 0's. Values
equal to 1 correspond to colonized cases and 0's correspond to not colonized individuals. You can add as many columns as your memory allows. 

@param total_patients_per_ward A matrix with two columns: 

Column 1: The number of the ward (unique identifier).
Column 2: The number of patients in that ward. 

@param parameters A matrix with three columns corresponding to the model's three parameters. The
first column corresponds to `alpha`, the second to `beta` and the third to `gamma`.

@return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`
*/
arma::mat progress_patients_1_timestep(arma::mat& ward_matrix,
                                       const arma::mat& total_patients_per_ward,
                                       const arma::mat& parameters) {

  //In ward_matrix the ward is in `ward_col` column
  const arma::uword ward_col_wards = 1;
  const arma::uword ward_col_patients = 1;

  //In `total_patients_per_ward` matrix the size of the ward is in `patient_size_col` column
  const arma::uword patient_size_col = 2;

  //Get unique ward values for the day
  arma::vec active_wards = arma::unique(ward_matrix.col(ward_col_wards));

  // Loop through the wards that have cases during the period
  for (arma::uword w = 0; w < active_wards.n_elem; ++w) {

    //Get the index of the ward
    arma::uvec ward_index = arma::find(ward_matrix.col(ward_col_wards) == active_wards[w]);
    arma::uvec patient_index = arma::find(total_patients_per_ward.col(ward_col_patients) == active_wards[w]);

    //Get the patients and ward submatrix
    arma::mat total_patients = total_patients_per_ward.rows(patient_index);
    arma::mat ward_temp = ward_matrix.rows(ward_index);

    //Apply the function
    progress_patients_probability_ward_1_timestep(ward_temp, total_patients(0, patient_size_col), parameters);

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

                (1 - alpha)*w_i + (1 - w_i)(beta / N)*sum(w_i * C_i) + gamma*h_i

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

@param total_patients_per_ward A matrix with two columns: 

Column 1: The number of the ward (unique identifier).
Column 2: The number of patients in that ward. 

@param parameters A matrix with three columns corresponding to the model's three parameters. The
first column corresponds to `alpha`, the second to `beta` and the third to `gamma`.

@param seed Random seed for `arma_rng`. 

@return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`
*/
arma::mat simulate_discrete_model_internal_one(const arma::mat& initial_colonized_probability,
                                               arma::mat& ward_matrix,
                                               const arma::mat& total_patients_per_ward,
                                               const arma::mat& parameters,
                                               const int seed){

  //FIXME: In R its not needed. Idk Python
  //Set seed for the simulation
  arma::arma_rng::set_seed(seed);

  const arma::uword day_col_wards = 0; //Column for days in ward_matrix
  const arma::uword day_col_patients = 0; //Column for days in total_patients_per_ward
  const arma::uword colonized_col_init = 6;
  const arma::uword next_day_col = 5;

  arma::uvec colonized_cols(initial_colonized_probability.n_cols);
  for (arma::uword col = 0; col < initial_colonized_probability.n_cols; ++col){
    colonized_cols(col) = colonized_col_init + col;
  }

  //Get the total amount of days
  arma::uword total_days = arma::max(total_patients_per_ward.col(day_col_patients)) + 1;

  //Create initial matrix for results
  arma::mat model_colonized(ward_matrix.n_rows, ward_matrix.n_cols + initial_colonized_probability.n_cols);

  //Get initial probability
  arma::mat uniform_mat(initial_colonized_probability.n_rows, initial_colonized_probability.n_cols, arma::fill::randu);
  model_colonized.submat(0, colonized_col_init, initial_colonized_probability.n_rows - 1,
                         model_colonized.n_cols - 1) = arma::conv_to<arma::mat>::from(
                           uniform_mat < initial_colonized_probability);

  //Get the other values
  model_colonized.submat(0, 0, model_colonized.n_rows - 1, colonized_col_init - 1) = ward_matrix;

  for(arma::uword day = 0; day < total_days; ++day) {

    //Get the total amount of patients in that day
    arma::mat total_patients_per_day = total_patients_per_ward.rows(arma::find(total_patients_per_ward.col(day_col_patients) == day));

    //Get the ward info for the day
    arma::uvec model_colonized_idx = arma::find(model_colonized.col(day_col_wards) == day);
    arma::mat ward_per_day = model_colonized.rows(model_colonized_idx);

    //Get the next timestep of the model
    progress_patients_1_timestep(ward_per_day, total_patients_per_day, parameters);

    //Update current colonized values
    model_colonized.rows(model_colonized_idx) = ward_per_day;

    //Fill the values of the next step (temporarily)
    arma::mat active_wards_next = ward_per_day.rows(arma::find(ward_per_day.col(next_day_col) > 0));
    arma::uvec active_columns = arma::conv_to<arma::uvec>::from(active_wards_next.col(next_day_col));
    model_colonized.submat(active_columns, colonized_cols) = active_wards_next.cols(colonized_cols);


  }

  return model_colonized;

}

/** 
arma::cube simulate_discrete_model_internal(const arma::mat& initial_colonized_probability,
                                            arma::mat& ward_matrix,
                                            const arma::mat& total_patients_per_ward,
                                            const arma::mat& parameters,
                                            const arma::uword n,
                                            unsigned int arma_seed,
                                            const unsigned int num_threads) {


  omp_set_num_threads(num_threads);

  py::print("Using `num_threads` = ", omp_get_max_threads(),
            "threads to run `n` = ", static_cast<int>(n),
            "independent simulations,\n each one being a block of `nrow(parameters)` = ",
            static_cast<int>(initial_colonized_probability.n_cols),
            "different parameters\n"
            "for a total of `nrow(parameters)*n` = ",
            static_cast<int>(initial_colonized_probability.n_cols * n),
            "distinct trajectories.\n");

  //progressbar bar(static_cast<int>(n));

  arma::cube simulation_results(ward_matrix.n_rows,
                                ward_matrix.n_cols + initial_colonized_probability.n_cols,
                                n, arma::fill::zeros);

  // Parallelize the loop using OpenMP and set a different seed for each thread
  #pragma omp parallel for schedule(static) private(arma_seed)
  for (arma::uword sim = 0; sim < n; ++sim) {
    // Generate a different seed for each thread using the thread index and the global seed
    unsigned int thread_seed = arma_seed + sim;


    //printf("Starting simulation in thread %d", omp_get_num_threads());
    // Simulate one discrete model and store the result in the cube
    simulation_results.slice(sim) = simulate_discrete_model_internal_one(
      initial_colonized_probability, ward_matrix, total_patients_per_ward, parameters, thread_seed);

    //#pragma omp critical
    //    bar.update();

  }

  return simulation_results;
}
*/

/*
@title Obtain daily number of colonized

@description Given a simulation created with `simulate_discrete` this function obtains the total number of
colonized cases per simulation and per day. 

@param model_colonized A matrix obtained from a `simulate_discrete_model` object

@return A matrix with days being each rows and columns being the number of simulations. Each
matrix entry corresponds to the total number of colonized cases in that day/simulation. 
*/
arma::mat total_positive(const arma::mat& model_colonized){

  //Column for days in ward_matrix
  const arma::uword day_column = 0;
  const arma::uword colonized_col_init = 6;
  const arma::uword colonized_col_end = model_colonized.n_cols - 1;

  //Get the total number of days
  arma::uword total_days = arma::max(model_colonized.col(day_column)) + 1;

  arma::mat model_positive(total_days, colonized_col_end - colonized_col_init + 1);

  //Loop through days and sum
  for(arma::uword day = 0; day < total_days; ++day) {
    arma::mat subcol = model_colonized.rows(arma::find(model_colonized.col(day_column) == day));
    model_positive.row(day) = arma::sum(subcol.cols(colonized_col_init, colonized_col_end), 0);

  }

  return model_positive;

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
arma::mat summary_of_total_positive(const arma::mat& model_colonized,
                                      const arma::vec quantiles){

  //Get the total values of colonized
  arma::mat model_positive = total_positive(model_colonized);

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
