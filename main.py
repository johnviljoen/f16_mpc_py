# In[] imports

# from ctypes import *
from ctypes import CDLL
#import ctypes
import os

# import numpy and sin, cos for convenience
import numpy as np

# handbuilt functions for all this
from utils import tic, toc, vis
from trim import trim
from sim import upd_sim
from mpc import linearise, dmom, calc_MC, calc_x_seq, calc_HFG, dlqr, square_mat_degen_2d

# import progressbar for convenience
import progressbar

# import parameters
from parameters import initial_state_vector_ft_rad, simulation_parameters, paras_mpc

# import exit() function for debugging
from sys import exit

# from scipy.linalg import expm, inv, pinv
from scipy.signal import cont2discrete


# In[]

#----------------------------------------------------------------------------#
#-------------------------prepare data for nlplant.c-------------------------#
#----------------------------------------------------------------------------#

# unwrap simulation parameters
time_step, time_start, time_end, stab_flag, fi_flag = simulation_parameters

# create interface with c shared library .so file in folder "C"
if stab_flag == 1:
    so_file = os.getcwd() + "/C/nlplant_xcg35.so"
elif stab_flag == 0:
    so_file = os.getcwd() + "/C/nlplant_xcg25.so"
    
nlplant = CDLL(so_file)

# initialise x
x = initial_state_vector_ft_rad


# In[]

#----------------------------------------------------------------------------#
#---------------------------------Simulate-----------------------------------#
#----------------------------------------------------------------------------#

output_vars = [6,7,8,9,10,11]

# trim aircraft
h_t = 10000
v_t = 700

x, opt_res = trim(h_t, v_t, fi_flag, nlplant)



u = x[12:16]
# x = x[np.newaxis].T

# turn x, u into matrices
x = x[np.newaxis].T
u = u[np.newaxis].T
x0 = np.copy(x)


rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))

# create storage
x_storage = np.zeros([len(rng),len(x)])
A = np.zeros([len(x),len(x),len(rng)])
B = np.zeros([len(x),len(u),len(rng)])
C = np.zeros([len(output_vars),len(x),len(rng)])
D = np.zeros([len(output_vars),len(u),len(rng)])

# Q = np.eye(A.shape[0])
# Q[0,0] = 0
# Q[1,1] = 0
# Q[2,2] = 0.1
# Q[3,3] = 0.1
# Q[4,4] = 0.1
# Q[5,5] = 0
# Q[6,6] = 0.5
# Q[7,7] = 1
# Q[8,8] = 1
# Q[9,9] = 100
# Q[10,10] = 100
# Q[11,11] = 100

# Q[12,12] = 0
# Q[13,13] = 0
# Q[14,14] = 0
# Q[15,15] = 0
# Q[16,16] = 0
# Q[17,17] = 0

# R = np.eye(B.shape[1])
# R[0,0] = 1000
# R[1,1] = 10
# R[2,2] = 100
# R[3,3] = 1

Q = np.eye(9)
R = np.eye(4)

bar = progressbar.ProgressBar(maxval=len(rng)).start()

tic()

for idx, val in enumerate(rng):
    
    #----------------------------------------#
    #------------linearise model-------------#
    #----------------------------------------#
    
    [A[:,:,idx], B[:,:,idx], C[:,:,idx], D[:,:,idx]] = linearise(x, u, output_vars, fi_flag, nlplant)
    Ad, Bd, Cd, Dd = cont2discrete((A[:,:,idx],B[:,:,idx],C[:,:,idx],D[:,:,idx]), time_step)[0:4]
    
    #----------------------------------------#
    #--------------Take Action---------------#
    #----------------------------------------#
    
    degen_idx = [2,3,4,6,7,8,9,10,11]
    Ad = square_mat_degen_2d(Ad, degen_idx)
    Bd = np.vstack((Bd[2:5,:], Bd[6:12,:]))
    
    x_degen = np.array([x[i] for i in degen_idx])
    x0_degen = np.array([x0[i] for i in degen_idx])
    
    K = dlqr(Ad,Bd,Q,R)
    u = - (K @ (x_degen - x0_degen))
    
    
    #----------------------------------------#
    #--------------Integrator----------------#
    #----------------------------------------#    
    
    x = upd_sim(x, u, fi_flag, time_step, nlplant)
    
    #----------------------------------------#
    #------------Store History---------------#
    #----------------------------------------#
    
    x_storage[idx,:] = x[:,0]
    
    bar.update(idx)

toc()

# In[]

#----------------------------------------------------------------------------#
#---------------------------------Visualise----------------------------------#
#----------------------------------------------------------------------------#

#%matplotlib qt

vis(x_storage, rng)

