import scipy.io
import numpy as np
import matlab.engine
# Read the matlab file as dictionaries
mat = scipy.io.loadmat('data/NY_Network_1.mat', simplify_cells=True)

#Accsess the Network part of the file (other parts are metadata)
Network = mat["Network"]
np.array(Network["day0"])
