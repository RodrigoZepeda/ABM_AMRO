#include <carma>
#include <armadillo>
#include <cstdio>
#include <progressbar.hpp>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_num_threads() 0
    inline void omp_set_num_threads(int num_threads) { static_cast<void>(num_threads); }
    #define omp_get_thread_num() 0
#endif

// Example
// --------------
// library(Matrix)
// total_patients <- 20
// ward_matrix <- c(rep(0,6), #Day
//                  rep(1,6), #Ward
//                  52, 53, 17, 200, 87, 99, #MRN
//                  1, 1, 0, 0, 1, 0, #New_arrival
//                  1,0.5,0.5,1,1,0.3, #Weight
//                  rep(-1, 6), #Next day position
//                  0,0,0,1,0,0, #Colonized status under model 1
//                  0,0,1,0,0,1, #Colonized status under model 2
//                  0,0,0,0,0,0) |>  #Colonized status under model 3
//   matrix(ncol = 9)
// parameters <- c(0.1 , 0.5 , 0.9, #alpha
//                   0.2 , 0.35, 0.5, #beta
//                   0.5 , 0.6 , 0.7) |> matrix(ncol = 3) #gamma
// colnames(parameters) <- c("alpha","beta","gamma")
// progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters)
// [[Rcpp::export]]
arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix,
                                                        const double& total_patients,
                                                        const arma::mat& parameters) {

  //In parameter matrix
  const arma::uword alpha_col = 0;
  const arma::uword beta_col = 1;
  const arma::uword gamma_col = 2;

  //In ward matrix
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

// Example
// --------------
// library(Matrix)
// total_patients = matrix(
//   c(0,0, #Day
//     1,3, #Ward
//     10,20 #Size
//   ), ncol = 3)
//
// ward_matrix <- matrix(
//     c(rep(0, 6), #Day
//       1, 3, 1, 3, 3, 3,#Ward
//      52, 53, 17, 200, 87, 99, #MRN
//      1, 1, 0, 0, 1, 0, #New_arrival
//      1,1,1,1,1,1, #Weight
//      rep(-1, 6), #Next day position
//      0,0,1,0,0,0, #Colonized status under model 1
//      0,0,1,0,0,1, #Colonized status under model 2
//      0,0,0,0,0,0 #Colonized status under model 3
//     ), ncol = 9)
//
// parameters <- c(0.1 , 0.5 , 0.9, #alpha
//                 0.2 , 0.35, 0.5, #beta
//                 0.5 , 0.6 , 0.7) |> matrix(ncol = 3) #gamma
// colnames(parameters) <- c("alpha","beta","gamma")
//
// progress_patients_1_timestep(ward_matrix, total_patients, parameters)
// [[Rcpp::export]]
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

// [[Rcpp::export]]
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

// Function to run 'n' simulations in parallel
// [[Rcpp::export]]
arma::cube simulate_discrete_model_internal(const arma::mat& initial_colonized_probability,
                                            arma::mat& ward_matrix,
                                            const arma::mat& total_patients_per_ward,
                                            const arma::mat& parameters,
                                            const arma::uword n,
                                            unsigned int arma_seed,
                                            const unsigned int num_threads) {


  omp_set_num_threads(num_threads);

  printf("Using `num_threads` = %d threads to run `n` = %d independent simulations,\n"
         "each one being a block of `nrow(parameters)` %d different parameters\n"
         "for a total of `nrow(parameters)*n` = %d distinct trajectories.\n",
          omp_get_max_threads(),
          static_cast<int>(n),
          static_cast<int>(initial_colonized_probability.n_cols),
          static_cast<int>(initial_colonized_probability.n_cols * n));

  progressbar bar(static_cast<int>(n));

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

    #pragma omp critical
        bar.update();

  }

  return simulation_results;
}


// [[Rcpp::export]]
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

// [[Rcpp::export]]
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
