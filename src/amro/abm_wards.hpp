#include <carma>
#include <armadillo>
#include <omp.h>

arma::mat progress_patients_probability_ward_1_timestep(arma::mat& ward_matrix,
                                                        const double& total_patients,
                                                        const arma::mat& parameters,
                                                        const arma::uword n_sims,
                                                        const int& time_to_detect,
                                                        const bool& detect_today_hospitalized,
                                                        const bool& detect_today_arrival);

arma::mat progress_patients_1_timestep(arma::mat& ward_matrix,
                                       const arma::mat& total_patients_per_ward,
                                       const arma::mat& parameters,
                                       const arma::uword n_sims,
                                       const int& time_to_detect,
                                       const bool& detect_today_hospitalized,
                                       const bool& detect_today_arrival);

arma::mat simulate_discrete_model_internal_one(const arma::mat& initial_colonized_probability,
                                               const arma::mat& initial_detected_probability,
                                               arma::mat& ward_matrix,
                                               const arma::mat& total_patients_per_ward,
                                               const arma::mat& parameters,
                                               const int& time_to_detect,
                                               const int& testing_schedule_hospitalized,
                                               const int& testing_schedule_arrivals,
                                               const int seed);

arma::mat total_positive(const arma::mat& model_colonized, const arma::uword col_init, const arma::uword col_end);

arma::mat total_positive_colonized(const arma::mat& model_colonized, const arma::uword n_sims);

arma::mat total_positive_detected(const arma::mat& model_colonized, const arma::uword n_sims);

arma::mat summary_of_total_positive(const arma::mat& model_colonized, const arma::vec quantiles);

arma::mat simulate_and_collapse(arma::mat& ward_matrix,
                                const arma::mat& total_patients_per_ward,
                                const double initial_colonized_probability,
                                const double initial_detected_probability,
                                const arma::uword initial_patients,
                                const double alpha,
                                const double beta,
                                const double gamma,
                                const double rho_hospital,
                                const double alpha2,
                                const double rho_imported,
                                const arma::uword n_sims,
                                const arma::uword time_to_detect,
                                const arma::uword testing_schedule_hospitalized,
                                const arma::uword testing_schedule_arrivals,
                                const int seed);