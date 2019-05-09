""" 1FE_compare_N.py
This file is a part of Cahn-Hilliard
Authors: Cristian Lacey, Sijie Tong, Amlan Sinha
This file contains an example of using the class CahnHilliard() to quickly loop
over simulations with different input parameters.
"""
import sys
import os
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
import time as time
srcDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/src'
sys.path.insert(0,srcDir)

from cahnhilliard import CahnHilliard

def generate_gif(filenames,output_path):
    '''
    Generate gif from list of filenames.
    '''
    with imageio.get_writer(output_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

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
# 'time_method' : '1FE', # Choice of 1FE, RK4, 1BE, 2CN
'sparse_format' : 'csr', # Choice of dia, csc, csr

'output_gif' : True,
# 'output_path' : './output.gif',
}

# Can be easily looped over for multiple runs with different parameters.
# (comment out looped over inputs in global_inputs dict, ie N, spatial method)
specific_inputs = [
#{'N':97,'dx':50/97,'spatial_method':'4CD','time_method':'RK4','dt' : 0.001,'tsteps' : 60001,'dump' : 6000,'output_path' : './output1.gif'},
#{'N':97,'dx':50/97,'spatial_method':'4CD','time_method':'RK5','dt' : 0.001,'tsteps' : 60001,'dump' : 6000,'output_path' : './4CD,RK5,97.gif'},
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

filelist_1 = glob.glob('./analysis/4CD,RK4,97/*.txt')
filelist_2 = glob.glob('./analysis/4CD,RK5,97/*.txt')

err_RK4vsRK5 = open("./analysis/RK4vsRK5.txt","a")
for i, fi in enumerate(filelist_1):
    err = np.abs((np.loadtxt(fi)-np.loadtxt(filelist_2[i]))/np.loadtxt(filelist_2[i]))
    err_RK4vsRK5.write(str(err.max())+'\n')
    plt.imshow(err)
    #plt.clim(-0.001, 0.001);
    plt.colorbar()
    plt.savefig('./analysis/errt00'+str(i)+'.png', dpi=300)
    plt.clf()
err_RK4vsRK5.close()

filenames = ['./analysis/errt00'+str(i)+'.png' for i in range(0,10)]
generate_gif(filenames,"./analysis/RK4vsRK5.gif")
