""" 1FE_compare_N.py
This file is a part of Cahn-Hilliard
Authors: Cristian Lacey, Sijie Tong, Amlan Sinha
This file contains an example of using the class CahnHilliard() to quickly loop
over simulations with different input parameters.
"""
import sys
import os
import time as time
srcDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/src'
sys.path.insert(0,srcDir)

import imageio
import glob

from cahnhilliard import CahnHilliard

# -------------------------------
# INPUT PARAMETERS
# -------------------------------
global_inputs = {
# 'N' : 100, # lattice points per axis
# 'dx' : 0.5, # lattice spacing
# 'dt' : 0.01, # timestep size
# 'tsteps' : 10001, # number of timesteps
'tol' : 1e-5, # convergence criterion for implicit methods (not used for explicit)

# 'dump' : 1000, # dump an image every 'dump' steps
'phi_avg' : 0, # initial mean value of phi
'noise' : 0.1, # initial amplitude of fluctuations
'seed' : 0, # seed for random initilization (use None for random output)

#'spatial_method' : '2CD', # Choice of 2CD, 4CD
# 'time_method' : '1FE', # Choice of 1FE, RK4, RK5, 1BE, 2CN
'sparse_format' : 'csr', # Choice of dia, csc, csr

'output_gif' : True,
# 'output_path' : './output.gif',
}

# Can be easily looped over for multiple runs with different parameters.
# (comment out looped over inputs in global_inputs dict, ie N, spatial method)
specific_inputs = [
{'N':97,'dx':50/97,'spatial_method':'2CD','time_method':'1FE','dt' : 0.001,'tsteps' : 60001,'dump' : 6000,'output_path' : './output1.gif'},
{'N':97,'dx':50/97,'spatial_method':'4CD','time_method':'1FE','dt' : 0.001,'tsteps' : 60001,'dump' : 6000,'output_path' : './output2.gif'},
]

# -------------------------------
# MAIN
# -------------------------------
for inp in specific_inputs:
    ch = CahnHilliard(**global_inputs,**inp)
    tStart = time.time()
    ch.run()
    tEnd = time.time()
    print('Time taken = '+str(tEnd-tStart))
