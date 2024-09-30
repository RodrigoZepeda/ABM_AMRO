import pandas as pd
import numpy as np
import amro
import seaborn as sns
import matplotlib_inline.backend_inline
from IPython.core.display import HTML

# Context for figures in jupyter notebook
sns.set(rc={"figure.dpi":200, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

# My obsessive brain needs centered figures:
# https://stackoverflow.com/a/66493988
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

total_patients_per_ward = pd.read_csv("../data/wards_py.csv")
total_patients_per_ward.head()

ward_matrix = pd.read_csv("../data/patients_py.csv")
ward_matrix.head()

#Convert the total patients
total_patients_per_ward = np.array(total_patients_per_ward)

#Get the initial patients to start the algorithm
initial_patients = ward_matrix[ward_matrix["day"] == 0].shape[0]

#Convert the rest of the patients
ward_matrix = np.array(ward_matrix)

#Create 100 simulations and run them in ~ 20 seconds
num_parameters = 2

#Initial colonized values represent the probability that each patient starts up colonized
colonized_day_0 = 0.2*np.ones(shape = (initial_patients, num_parameters))

#Create the parameter matrix
parameters = np.array([
                    np.linspace(0.1, 0.1, num_parameters), #Alpha
                    np.linspace(0.5, 0.5, num_parameters), #Beta
                    np.linspace(0.02, 0.02, num_parameters), #Gamma
                    ]).transpose()

simulations_100 = amro.simulate_discrete_model_internal_one(colonized_day_0, ward_matrix, total_patients_per_ward, parameters, 1)
