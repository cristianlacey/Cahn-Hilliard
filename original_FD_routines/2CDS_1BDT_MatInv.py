""" 2CDS_1FDT.py
This file is a part of Cahn-Hilliard
Authors: Cristian Lacey, Sijie Tong, Amlan Sinha
This file contains a routine to solve the Cahn-Hilliard equation using second
order centered difference in space and first order backward Euler in time.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
import scipy.sparse.linalg as la
from scipy import linalg

import imageio

# -------------------------------
# FUNCTION DEFINITIONS
# -------------------------------
def update(phi,dt,lap,A_inv,tol=1e-6):
    '''
    Updates current state of phi using second order centered difference
    in space and first order backward Euler in time.
        Args:
            phi (np.array): Array in col major format of phase concentration.
            dt (float): Timestep size
            lap (sparse.dia_matrix): Laplacian operator with periodic BCs.
        Returns:
            phi_n (np.array): Array in col major format of phase concentration
            in next timestep.
    '''
    # Get initial guess from first order Euler in time
    # Calculate right-hand side of PDE
    rhs = lap.dot(np.power(phi,3) - phi - (lap.dot(phi)))
    # Step forward in time with first order Euler
    phi_n = phi + dt*rhs

    # Refine forward Euler estimate by converging to backward Euler, writing
    # the PDE in the form A(phi-)phi = b(phi-). In this case, A(phi-) is
    # constant form iter to iter, so its inverse is pre-computed and passed to
    # update(), allowing phi to be iteratively solved as phi = A_inv@b(phi-).
    phi_p = np.zeros((len(phi),1))
    while np.amax(np.absolute(phi_n - phi_p)) > tol:
        b = phi + dt*lap.dot(np.power(phi_n,3))
        phi_n = np.dot(A_inv,b)
        # phi_n = A_inv.dot(b)
        phi_p = phi_n

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
dt = 0.02 # timestep size
tsteps = 5001 # number of timesteps

dump = 250 # dump an image every 'dump' steps
phi_avg = 0 # initial mean value of phi
noise = 0.1 # initial amplitude of fluctuations
seed = 0 # seed for random initilization (use None for random output)

output_path = './output.gif'

# -------------------------------
# INITIALIZATION
# -------------------------------
# Initialize seed value
np.random.seed(seed=seed)

# Initialize phi as NxN matrix with random values bounded by phi_avg + noise/2
phi = phi_avg*(np.ones((N,N))) + noise*(np.random.rand(N,N)-0.5)
t = np.arange(0, tsteps*dt, dt)

# Unravel phi in col major form
phi = np.ravel(phi, order='F')
phi = phi.reshape(len(phi),1)

# Define Laplace operator with periodic BCs, then convert to sparse.dia_matrix
# object to leverage faster matrix multiplication of block-banded Laplacian
A = sparse.diags([1,1,-2,1,1], [-(N-1),-1,0,1,(N-1)], shape=(N,N)).toarray()
A = A/(dx*dx)
I = np.eye(N)
lap = sparse.kron(I,A) + sparse.kron(A,I)
lap = sparse.dia_matrix(lap)
# plt.matshow(lap)
# plt.show()

# Precompute A_inv
A = np.eye(N*N) + dt*(lap + lap.dot(lap))
A_inv = np.linalg.inv(A)
# A = sparse.csc_matrix(A)
# A_inv = la.inv(A)

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
    phi = update(phi,dt,lap,A_inv)

generate_gif(filenames,output_path)
