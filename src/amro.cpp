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

    m.doc() = "Agent Based Model (ABM) module for antimicrobial resistance project.";

    m.def("simulate_discrete_model_internal_one", &simulate_discrete_model_internal_one,
        "A function for simulating one realization of the ABM model",
          "initial_colonized_probability"_a=1,
          "ward_matrix"_a=2,
          "total_patients_per_ward"_a=3,
          "parameters"_a=4,
          "seed"_a=5);

    m.def("progress_patients_probability_ward_1_timestep", &progress_patients_probability_ward_1_timestep,
        "A function for simulating one ward of the ABM model",
          "ward_matrix"_a=1,
          "total_patients"_a=2,
          "parameters"_a=3);

    m.def("progress_patients_1_timestep", &progress_patients_1_timestep,
        "A function for simulating one day of the ABM model",
          "ward_matrix"_a=1,
          "total_patients_per_ward"_a=2,
          "parameters"_a=3);

    m.def("simulate_discrete_model_internal", &simulate_discrete_model_internal,
        "A function for simulating one day of the ABM model",
          "initial_colonized_probability"_a=1,
          "ward_matrix"_a=2,
          "total_patients_per_ward"_a=3,
          "parameters"_a=4,
          "n"_a=5,
          "arma_seed"_a=6,
          "num_threads"_a=7);

    #ifdef VERSION_INFO
        m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
    #else
        m.attr("__version__") = "dev";
    #endif
}
