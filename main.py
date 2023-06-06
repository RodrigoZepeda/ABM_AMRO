import scipy.io
import numpy as np


# Read the matlab file as dictionaries
mat = scipy.io.loadmat('Figure_2/Inf_res_1.mat', simplify_cells=True)

#Access the Network part of the file (other parts are metadata)
Dits = mat["Dits"]
np.array(Network["day0"])
