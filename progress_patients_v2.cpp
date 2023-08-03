// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

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
arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix, 
                                                        const double& total_patients,
                                                        const arma::mat& parameters) {
  
  //Get the column positions for the variables. Doing this is good practice to avoid coding errors.
  //In parameter matrix
  const arma::uword alpha_col = 0;
  const arma::uword beta_col = 1;
  const arma::uword gamma_col = 2;
  
  //In ward matrix
  const arma::uword arrivals_col = 3;
  const arma::uword weights_col = 4;
  const arma::uword colonized_col_init = 6;
  const arma::uword colonized_col_end = ward_matrix.n_cols;
  
  //Create the force of infection
  for(arma::uword col_index = colonized_col_init; col_index < colonized_col_end; ++col_index){
    
    //Weight the colonized part of the matrix W = weights*C
    ward_matrix.col(col_index) %= ward_matrix.col(weights_col);
    
    //Get the force of infection, F = beta/N * sum(W)
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
/Colonize the initial values in day 0
  arma::mat uniform_mat(initial_colonized_probability.n_rows, initial_colonized_probability.n_cols, arma::fill::randu);
  model_colonized.submat(0, colonized_col_init, 
                         initial_colonized_probability.n_rows - 1, 
                       model_colonized.n_cols - 1) = arma::conv_to<arma::mat>::from(uniform_mat < initial_colonized_probability);
  
  r (arma::uword day = 0; day < total_days; ++day) {
    
  //Get the total amount of patients in that day
  arma::mat total_patients_per_day = total_patients_per_ward.rows(arma::find(total_patients_per_ward.col(day_col_patients) == day));
      
  //Get the ward info for the day
  arma::mat&ward_per_day = model_colonized.rows(arma::find(model_colonized.col(day_col_wards) == day));
  
      //Get the next timestep of the model
  arma::mat model_one_timestep = progress_patients_1_timestep(ward_per_day, total_patients_per_day, parameters);
  
    sign the timestep 
    _per_day(arma::find(ward_per_day.col(next_day_col) > 0)) = model_one_timestep(arma::find(model_one_timestep.col(next_day_col) > 0));
    
      }
    
     model_colonized;
    
    }
    
    
    
    
    tients_per_ward <- matrix(
  
       ), ncol = 3)
  
ward_matrix <- matrix(
  c(
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, #Day
    1, 3, 1, 3, 3, 3, 2, 1, 2, 3, 3, 1, 2, 3, 1, #Ward
    52, 53, 17, 200, 87, 99, 52, 53, 44, 86, 100, 87, 87, 44, 52, #MRN
    1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, #New_arrival
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, #Weight
    6, 7, -1, -1, 12, -1, 15, -1, 14, -1, -1, 13, -1, -1, -1 #Next day position 
  ), ncol = 6)

colonized_day_0 <- matrix(
  c(
    0,0,1,0,0,0, #Colonized probability under model 1
    0.9,0.1,0.5,0.2,0.1,0.2, #Colonized probability under model 2
    rep(1/2, 6) #Colonized probability under model 3
  ), ncol = 3)


parameters <- c(0.1 , 0.5 , 0.9, #alpha
                0.2 , 0.35, 0.5, #beta
                0.5 , 0.6 , 0.7) |> matrix(ncol = 3) #gamma
colnames(parameters) <- c("alpha","beta","gamma")

simulate_discrete_model_internal_one(colonized_day_0, ward_matrix, total_patients_per_ward, parameters)


library(Matrix)
total_patients <- 20
ward_matrix <- c(rep(0,6), #Day
                 rep(1,6), #Ward
                 52, 53, 17, 200, 87, 99, #MRN
                 1, 1, 0, 0, 1, 0, #New_arrival
                 1,0.5,0.5,1,1,0.3, #Weight
                 rep(-1, 6), #Next day position
                 0,0,0,1,0,0, #Colonized status under model 1
                 0,0,1,0,0,1, #Colonized status under model 2
                 0,0,0,0,0,0) |>  #Colonized status under model 3
  matrix(ncol = 9)
parameters <- c(0.1 , 0.5 , 0.9, #alpha
                0.2 , 0.35, 0.5, #beta
                0.5 , 0.6 , 0.7) |> matrix(ncol = 3) #gamma
colnames(parameters) <- c("alpha","beta","gamma")
microbenchmark::microbenchmark(
progress_patients_probability_ward_1_timestep(ward_matrix, total_patients, parameters),
progress_patients_probability_ward_1_timestep_v2(ward_matrix, total_patients, parameters)
)
*/













































  


  
    
    
    
    
    
    
    
    
  






#

#

