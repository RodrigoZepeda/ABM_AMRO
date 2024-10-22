#include <pybind11/pybind11.h>
#include "abm_wards.hpp"
#include <carma>
#include <armadillo>
#include <omp.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(amro, m) {

    m.doc() = "Module implementing a stochastic model of antimicrobial resistant organisms in a healthcare facility.";

    m.def("simulate_discrete_model", &simulate_discrete_model_internal_one,
        R"doc(
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

            @param seed Random seed for `arma_rng`.

            @return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`

        )doc",
          "initial_colonized_probability"_a=1,
          "initial_detected_probability"_a=2,
          "ward_matrix"_a=3,
          "total_patients_per_ward"_a=4,
          "parameters"_a=5,
          "seed"_a=6);

    m.def("progress_patients_probability_ward_1_timestep", &progress_patients_probability_ward_1_timestep,
        R"doc(
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

        @return A `ward_matrix` corresponding to the state of the ward at time `t + 1`
        )doc",
          "ward_matrix"_a=1,
          "total_patients"_a=2,
          "parameters"_a=3,
          "n_sims"_a=4);

    m.def("progress_patients_1_timestep", &progress_patients_1_timestep,
        R"doc(
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

        @return A `ward_matrix` corresponding to the state of all the wards at time `t + 1`
        )doc",
          "ward_matrix"_a=1,
          "total_patients_per_ward"_a=2,
          "parameters"_a=3,
          "n_sims"_a=4);

    m.def("total_positive_colonized", &total_positive_colonized,
        R"doc(
        @title Obtain daily number of colonized

        @description Given a simulation created with `simulate_discrete` this function obtains the total number of
        colonized cases per simulation and per day.

        @param model_colonized A matrix obtained from a `simulate_discrete_model` object

        @return A matrix with days being each rows and columns being the number of simulations. Each
        matrix entry corresponds to the total number of colonized cases in that day/simulation.
        )doc",
          "model_colonized"_a=1,
          "n_sims"_a=2);

    m.def("total_positive_detected", &total_positive_detected,
        R"doc(
        @title Obtain daily number of detected

        @description Given a simulation created with `simulate_discrete` this function obtains the total number of
        colonized cases per simulation and per day.

        @param model_colonized A matrix obtained from a `simulate_discrete_model` object

        @return A matrix with days being each rows and columns being the number of simulations. Each
        matrix entry corresponds to the total number of colonized cases in that day/simulation.
        )doc",
          "model_colonized"_a=1,
          "n_sims"_a=2);

    m.def("summary_of_total_positive", &summary_of_total_positive,
        R"doc(
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
        )doc",
          "model_colonized"_a=1,
          "quantiles"_a=2);

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
