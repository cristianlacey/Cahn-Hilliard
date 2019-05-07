import os
import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt

filelist_1 = glob.glob('./analysis/2CD,RK4,25/*.txt')
filelist_2 = glob.glob('./analysis/4CD,RK4,100/*.txt')

for i, fi in enumerate(filelist_1):
    mat_A = np.loadtxt(fi)
    mat_B = np.loadtxt(filelist_2[i])
    pts_A = mat_A[(0,0,0,int(len(mat_A)/2),int(len(mat_A)/2),int(len(mat_A)/2),-1,-1,-1),\
                (0,int(len(mat_A)/2),-1,0,int(len(mat_A)/2),-1,0,int(len(mat_A)/2),-1)]
    pts_B = mat_B[(0,0,0,int(len(mat_A)/2),int(len(mat_A)/2),int(len(mat_A)/2),-1,-1,-1),\
                (0,int(len(mat_A)/2),-1,0,int(len(mat_A)/2),-1,0,int(len(mat_A)/2),-1)]
    err   = np.abs(pts_A-pts_B)
    print(err)
