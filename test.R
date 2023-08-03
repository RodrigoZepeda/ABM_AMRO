rm(list = ls())
setwd("~/OneDrive - Columbia University Irving Medical Center/ABM/ABM_Identifiability_python/")
library(Matrix)
library(RcppArmadillo)
library(Rcpp)
library(readr)
library(dplyr)

input_wards <- read_csv("wards.csv")
patients <- read_csv("~/OneDrive - Columbia University Irving Medical Center/ABM/ABM_Identifiability_python/patients.csv")
patients <- patients |> 
  dplyr::select(day, MRN, ward, weight) |>
  dplyr::group_by(MRN) |>
  distinct(day, MRN, weight, .keep_all = T) |>
  dplyr::arrange(day) |>
  dplyr::mutate(MRN_next = abs(day - dplyr::lag(day))) |>
  dplyr::mutate(is_new = dplyr::if_else(MRN_next <= 1, 0, 1, 1)) |>
  ungroup()
  
patients <- patients |> dplyr::select(day, MRN, ward, is_new, weight)

patients_df <- patients |> group_by(MRN, day) |> mutate(count = 1:n()) |> ungroup()

wards <- input_wards |> dplyr::select(day, ward, count)

patients_df <- patients_df |> mutate(row_id = 1:n())
patients_df <- patients_df |> 
  arrange(day) |> 
  group_by(MRN) |>
  mutate(next_day = if_else(lead(is_new) == 0, lead(row_id), NA_real_)) |>
  ungroup()


wards |> write_excel_csv("wards_py.csv")
patients_df |> write_excel_csv("patients_py.csv")



patients_df <- patients |> left_join(input_wards)


simulate_discrete_model_python(initial_colonized,
                               wards,
                               input_new_arrivals,
                               input_weights,
                               input_total_patients_per_ward,
                               input_parameters) 