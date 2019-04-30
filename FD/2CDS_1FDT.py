""" 2CDS_1FDT.py
This file is a part of Cahn-Hilliard
Authors: Cristian Lacey, Sijie Tong
This file contains a routine to solve the Cahn-Hilliard equation using second
order centered difference in space and first order forward Euler in time.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy import linalg

import imageio

# -------------------------------
# FUNCTION DEFINITIONS
# -------------------------------
def update(phi,dt,lap):
    '''
    Updates current state of phi using second order centered difference
    in space and first order forward Euler in time.
        Args:
            phi (np.array): Array in col major format of phase concentration.
            dt (float): Timestep size
            lap (sparse.dia_matrix): Laplacian operator with periodic BCs.
        Returns:
            phi_n (np.array): Array in col major format of phase concentration
            in next timestep.
    '''
    # Calculate right-hand side of PDE
    rhs = lap.dot(np.power(phi,3) - phi - (lap.dot(phi)))

    # Step forward in time with first order Euler
    phi_n = phi + dt*rhs

    return phi_n

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
N = 100 # lattice points per axis
dx = 1 # lattice spacing
dt = 0.01 # timestep size
tsteps = 10001 # number of timesteps

dump = 1000 # dump an image every 'dump' steps
phi_avg = 0 # initial mean value of phi
noise = 0.1 # initial amplitude of fluctuations
seed = 0 # seed for random initilization (use None for random output)

output_path = './output.gif'

# -------------------------------
# INITIALIZATION
# -------------------------------
# Initialize seed value
seed = np.random.seed(seed=seed)

# Initialize phi as NxN matrix with random values bounded by phi_avg + noise/2
phi = phi_avg*(np.ones((N,N))) + noise*(np.random.rand(N,N)-0.5)
t = np.arange(0, tsteps*dt, dt)

# Unravel phi in col major form
phi = np.ravel(phi, order='F')

# Define Laplace operator with periodic BCs, then convert to sparse.dia_matrix
# object to leverage faster matrix multiplication of block-banded Laplacian
A = sparse.diags([1,1,-2,1,1], [-(N-1),-1,0,1,(N-1)], shape=(N,N)).toarray()
I = np.eye(N)
lap = sparse.kron(I,A) + sparse.kron(A,I)
lap = sparse.dia_matrix(lap)
# plt.matshow(lap)
# plt.show()

# Initialize filenames for gif generation
filenames = ['t'+str(x).zfill(3)+'.png' for x in range(int(np.size(t)/dump)+1)]

# -------------------------------
# MAIN LOOP
# -------------------------------
for i in range(np.size(t)):
    print(t[i])
    if i % dump == 0:
        phi_plot = phi.reshape((N,N),order='F')
        plt.imshow(phi_plot)
        plt.colorbar()
        plt.savefig('./t'+str(int(i/dump)).zfill(3)+'.png', dpi=300)
        plt.clf()
    phi = update(phi,dt,lap)

generate_gif(filenames,output_path)
