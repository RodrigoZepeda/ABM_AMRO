#include <carma>
#include <armadillo>
#include <omp.h>

arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix,
                                                        const double& total_patients,
                                                        const arma::mat& parameters);

arma::mat progress_patients_1_timestep(arma::mat& ward_matrix,
                                       const arma::mat& total_patients_per_ward,
                                       const arma::mat& parameters);

arma::mat simulate_discrete_model_internal_one(const arma::mat& initial_colonized_probability,
                                               arma::mat& ward_matrix,
                                               const arma::mat& total_patients_per_ward,
                                               const arma::mat& parameters,
                                               const int seed);

arma::cube simulate_discrete_model_internal(const arma::mat& initial_colonized_probability,
                                            arma::mat& ward_matrix,
                                            const arma::mat& total_patients_per_ward,
                                            const arma::mat& parameters,
                                            const arma::uword n,
                                            unsigned int arma_seed,
                                            const unsigned int num_threads);
