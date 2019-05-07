""" cahnhilliard.py
This file is a part of Cahn-Hilliard
Authors: Cristian Lacey, Sijie Tong, Amlan Sinha
This file contains the class CahnHilliard(), objects of which solve the
Cahn-Hilliard equation using specified spatial and temporal methods.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as la
from scipy.stats import multivariate_normal

import imageio

# -------------------------------
# CLASS DEFINITIONS
# -------------------------------
class CahnHilliard():
    '''
    Class to solve the Cahn-Hilliard equation. Objects are instantiated with
    unpacked dictionary of input parameters. Output is then simply generated
    by invoking the .run() method of the created object.
    '''
    def __init__(self,N,dx,dt,tsteps,tol,dump,phi_avg,noise,seed,\
            spatial_method,time_method,sparse_format,output_gif,output_path):
        # Initialize inputs
        self.N = N
        self.dx = dx
        self.dt = dt
        self.tsteps = tsteps
        self.tol = tol
        self.dump = dump
        self.phi_avg = phi_avg
        self.noise = noise
        self.seed = seed
        self.spatial_method = spatial_method
        self.time_method = time_method
        self.sparse_format = sparse_format
        self.output_gif = output_gif
        self.output_path = output_path


        # Initialize seed value
        #np.random.seed(seed=seed)

        # Build t vector
        self.t = np.arange(0, tsteps*dt, dt)

        # Initialize phi as NxN matrix with random values bounded by phi_avg + noise/2
        #phi = phi_avg*(np.ones((N,N))) + noise*(np.random.rand(N,N)-0.5)
        XX = np.arange(-25, 25, dx)
        YY = np.arange(-25, 25, dx)
        XX, YY = np.meshgrid(XX, YY)
        pos = np.dstack((XX,YY))
        mn  = multivariate_normal([0, 0], [[25, 0], [0, 25]])
        phi = mn.pdf(pos)
        phi = phi-0.5*phi[int(np.round(len(XX)/2)),int(np.round(len(YY)/2))]

        # Unravel phi in col major form
        self.phi = np.ravel(phi, order='F')

        # Get laplacian matrix
        self.lap = self.get_laplacian(N,dx,spatial_method,sparse_format)

        # Get B matrix (only used for implicit methods) None if explicit method
        self.B = self.get_B(N,dt,self.lap,spatial_method=spatial_method,time_method=time_method,sparse_format=sparse_format)

        # Get k^2 for spectral method
        self.k2 = self.get_k2(N,spatial_method=spatial_method)

        # Initialize filenames for gif generation
        self.filenames = ['t'+str(x).zfill(3)+'.png' for x in range(int(tsteps/dump)+1)]

    def run(self):
        '''
        Steps the Chan-Hilliard equation forward in time according to
        user-specified steps and input parameters.
        '''
        # Initialize variables for readability
        N = self.N
        phi = self.phi
        dt = self.dt
        lap = self.lap
        t = self.t
        tsteps = self.tsteps
        dump = self.dump
        filenames = self.filenames
        output_gif = self.output_gif
        output_path = self.output_path
        spatial_method = self.spatial_method
        time_method = self.time_method
        B = self.B
        tol = self.tol

        path = "./analysis/"+self.spatial_method+","+self.time_method+","+str(N)
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" %path)
        else:
            print("Successfully created the directory %s" %path)

        for i in range(tsteps):
            print(t[i])
            if i % dump == 0:
                phi_plot = phi.reshape((N,N),order='F')
                plt.imshow(phi_plot)
                plt.colorbar()
                plt.savefig('./t'+str(int(i/dump)).zfill(3)+'.png', dpi=300)
                plt.clf()
                np.savetxt(path+'/t'+str(int(i/dump)).zfill(3)+'.txt', phi_plot)
            phi = self.step(phi,dt,lap,spatial_method,time_method,B,tol)

        if output_gif:
            self.generate_gif(filenames,output_path)

    def step(self,phi,dt,lap,spatial_method,time_method,B,tol):
        '''
        Updates current state of phi using specified temporal method.
            Args:
                phi (np.array): Array in col major format of phase
                    concentration.
                dt (float): Timestep size.
                lap (sparse.X_matrix): Laplacian operator with periodic BCs.
                time_method (str): Time method for stepping.
                B (sparse.X_matrix): Matrix used for implicit methods.
                tol (float): Convergence tolerance used for implicit methods.
            Returns:
                phi_n (np.array): Array in col major format of phase
                    concentration in next timestep.
        '''
        if spatial_method == 'spectral':
            if time_method == 'semi-implicit':
                phi_n = self.spectral_semi_implicit(phi,dt)
            if time_method == 'explicit':
                phi_n = self.spectral_explicit(phi,dt)


        elif time_method == '1FE':
            phi_n = self.forward_euler(phi,lap,dt)

        elif time_method == 'RK4':
            phi_n = self.rk4(phi,lap,dt)

        elif time_method == '1BE':
            # Take forward euler result as initial guess
            phi_n = self.forward_euler(phi,lap,dt)
            phi_n = self.backward_euler(phi,lap,dt,phi_n,B,tol)

        elif time_method == '2CN':
            # Take forward euler result as initial guess
            phi_n = self.forward_euler(phi,lap,dt)
            phi_n = self.crank_nicolson(phi,lap,dt,phi_n,B,tol)


        return phi_n

    def spectral_semi_implicit(self,phi,dt):
        '''
        Updates current state of phi using semi-implicit spectral method
        '''
        k2 = self.k2
        N = self.N
        phi = phi.reshape((N,N),order='F')
        phi_hat = np.fft.fft2(phi)
        g = phi**3-phi
        g_hat = np.fft.fft2(g)
        phi_hat = (-dt*k2*g_hat+phi_hat)/(1+dt*k2**2)
        phi = np.real(np.fft.ifft2(phi_hat))
        phi = np.ravel(phi, order='F')
        return phi

    def spectral_explicit(self,phi,dt):
        '''
        Updates current state of phi using explicit spectral method
        '''
        k2 = self.k2
        N = self.N
        phi = phi.reshape((N,N),order='F')
        phi_hat = np.fft.fft2(phi)
        g = phi**3-phi
        g_hat = np.fft.fft2(g)
        phi_hat = phi_hat-k2*dt*(g_hat+k2*phi_hat)
        phi = np.real(np.fft.ifft2(phi_hat))
        phi = np.ravel(phi, order='F')
        return phi


    def forward_euler(self,phi,lap,dt):
        '''
        Updates current state of phi using first order forward Euler in time.
        '''
        rhs = self.calc_rhs(phi,lap)
        phi_n = phi + dt*rhs
        return phi_n

    def rk4(self,phi,lap,dt):
        '''
        Updates current state of phi using fourth order Runge-Kutta in time.
        '''
        k1 = dt*self.calc_rhs(phi,lap)
        k2 = dt*self.calc_rhs(phi+k1/2,lap)
        k3 = dt*self.calc_rhs(phi+k2/2,lap)
        k4 = dt*self.calc_rhs(phi+k3,lap)
        phi_n = phi + (k1 + 2*k2 + 2*k3 + k4)/6
        return phi_n

    @staticmethod
    def backward_euler(phi,lap,dt,phi_n,B,tol):
        '''
        Refine forward Euler estimate by converging to backward Euler, writing
        the PDE in the form B @ phi_n+ = b(phi,phi_n-) where phi_n is phi in
        next time step, phi_n+ is the next guess for phi_n, and phi_n- is the
        current guess for phi_n. In this case, B is constant from iter to iter,
        so it's passed, allowing phi_n to be iteratively solved for using the
        method of conjugate gradient.
        '''
        phi_p = np.zeros((len(phi),1))
        while np.amax(np.absolute(phi_n - phi_p)) > tol:
            phi_p = phi_n
            b = phi + dt*lap.dot(np.power(phi_n,3))
            phi_n = la.cg(B,b,x0=phi_n)[0]
        return phi_n

    def crank_nicolson(self,phi,lap,dt,phi_n,B,tol):
        '''
        Refine forward Euler estimate by converging to Crank-Nicolson, writing
        the PDE in the form B @ phi_n+ = b(phi,phi_n-) where phi_n is phi in
        next time step, phi_n+ is the next guess for phi_n, and phi_n- is the
        current guess for phi_n. In this case, B is constant from iter to iter,
        so it's passed, allowing phi_n to be iteratively solved for using the
        method of conjugate gradient.
        '''
        phi_p = np.zeros((len(phi),1))
        while np.amax(np.absolute(phi_n - phi_p)) > tol:
            phi_p = phi_n
            b = phi + dt*(lap.dot(np.power(phi_n,3)) + self.calc_rhs(phi,lap))/2
            phi_n = la.cg(B,b,x0=phi_n)[0]
        return phi_n

    @staticmethod
    def calc_rhs(phi,lap):
        '''
        Calculates the right-hand side of the Cahn-Hilliard PDE, with D = gamma = 1.
        '''
        return lap.dot(np.power(phi,3) - phi - (lap.dot(phi)))

    @staticmethod
    def get_laplacian(N,dx,spatial_method,sparse_format):
        '''
        Define Laplace operator with periodic BCs, then convert to sparse
        object to leverage faster matrix multiplication of block-banded Laplacian.
        '''
        if spatial_method == 'spectral':
            return None
        elif spatial_method == '2CD':
            A = sparse.diags([1,1,-2,1,1],[-(N-1),-1,0,1,(N-1)],shape=(N,N)).toarray()
            A = A/(dx*dx)
        elif spatial_method == '4CD':
            A = sparse.diags([16,-1,-1,16,-30,16,-1,-1,16],\
                [-(N-1),-(N-2),-2,-1,0,1,2,(N-2),(N-1)],shape=(N,N)).toarray()
            A = A/(12*dx*dx)

        I = sparse.eye(N)
        lap = sparse.kron(I,A) + sparse.kron(A,I)
        if sparse_format == 'dia':
            lap = sparse.dia_matrix(lap)
        elif sparse_format == 'csc':
            lap = sparse.csc_matrix(lap)
        elif sparse_format == 'csr':
            lap = sparse.csr_matrix(lap)

        return lap

    @staticmethod
    def get_B(N,dt,lap,spatial_method,time_method='1FE',sparse_format='csr'):
        '''
        Returns B matrix corresponding to time method. Returns None for
        explicit methods. See implementation of implicit methods for more
        detail on B.
        '''
        if spatial_method == 'spectral':
            return None
        if time_method in ['1FE','RK4']:
            return None
        elif time_method == '1BE':
            B = sparse.eye(N*N) + dt*(lap + lap.dot(lap))
        elif time_method == '2CN':
            B = sparse.eye(N*N) + dt*(lap + lap.dot(lap))/2

        if sparse_format == 'dia':
            B = sparse.dia_matrix(B)
        elif sparse_format == 'csc':
            B = sparse.csc_matrix(B)
        elif sparse_format == 'csr':
            B = sparse.csr_matrix(B)

        return B

    def get_k2(self,N,spatial_method='spectral'):

        if spatial_method == 'spectral':
            kx1d = np.append(np.linspace(0,N//2,N//2+1),np.linspace(-((N-1)//2),-1,(N-1)//2))*2*np.pi/N
            ky1d = kx1d
            kx, ky =np.meshgrid(kx1d,ky1d)
            k2 = kx**2+ky**2
            return k2
        else:
            return None

    @staticmethod
    def generate_gif(filenames,output_path):
        '''
        Generate gif from list of filenames.
        '''
        with imageio.get_writer(output_path, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

# # -------------------------------
# # INPUT PARAMETERS
# # -------------------------------
# # Sample run. Comment out for imports
# inputs = {
# 'N' : 100, # lattice points per axis
# 'dx' : 0.5, # lattice spacing
# 'dt' : 0.001, # timestep size
# 'tsteps' : 100001, # number of timesteps
# 'tol' : 1e-5, # convergence criterion for implicit methods (not used for explicit)
#
# 'dump' : 10000, # dump an image every 'dump' steps
# 'phi_avg' : 0, # initial mean value of phi
# 'noise' : 0.1, # initial amplitude of fluctuations
# 'seed' : 0, # seed for random initilization (use None for random output)
#
# 'spatial_method' : '4CD', # Choice of 2CD, 4CD
# 'time_method' : 'RK4', # Choice of 1FE, RK4, 1BE, 2CN
# 'sparse_format' : 'csr', # Choice of dia, csc, csr
#
# 'output_gif' : True,
# 'output_path' : './output.gif',
# }
#
# # -------------------------------
# # MAIN
# # -------------------------------
# # Sample run. Comment out for imports
# ch = CahnHilliard(**inputs)
# ch.run()
