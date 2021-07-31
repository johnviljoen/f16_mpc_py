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
from mpc import linearise, dmom, calc_MC, calc_x_seq, calc_HFG

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

# # find the continuous time A,B,C,D
# A_c, B_c, C_c, D_c = linearise(x, u, output_vars, fi_flag, nlplant)

# # calculate the discrete time A,B,C,D
# A_d, B_d, C_d, D_d = cont2discrete((A_c, B_c, C_c, D_c), time_step)[0:4]

# turn x, u into matrices
x = x[np.newaxis].T
u = u[np.newaxis].T

# stack u's vertically
u_seq_vert = np.concatenate(tuple(u for _ in range(paras_mpc[0])))

# calculate the sequence of x
# x_seq = calc_x_seq(A_d, B_d, x, u_seq_vert, paras_mpc[0])


K = np.zeros((4,18))

K[0,12] = 1
K[1,13] = 20.2
K[2,14] = 20.2
K[3,15] = 20.2

R = 0.01

# H, F, G = calc_HFG(A_d, B_d, C_d, K, R, paras_mpc[0])

# # immediate K to apply
# K = -np.matmul(np.linalg.inv(H), F)[0:B_d.shape[1],:]

u_next = np.matmul(K,x)


# exit()

######################TESTING##################



# A = np.array([[1.1, 2],[0, 0.95]])
# B = np.array([[0],[0.0787]])
# C = np.array([-1,1])[np.newaxis]
# hzn = 4

# Q = np.matmul(C.T, C)
# R = 0.01

# H, F, G = calc_HFG(A, B, C, hzn, Q, R)

# x0 = np.array([0,0])[np.newaxis]

# L = -np.matmul(np.linalg.inv(H),F)

# K = L[0,:][np.newaxis]

# RHS = Q + np.matmul(K.T,K)

# clp = A + np.matmul(B, K)



##############################################

# A = np.array([[-2, 1],[0, 1]])
# B = np.array([[1],[1]])
# C = np.array([1, 1])[np.newaxis]

# K = np.array([2, -1])[np.newaxis]

# RHS = np.matmul(C.T,C) + np.matmul(K.T,K)

# clp = A + np.matmul(B,K)





# scipy.linalg.solve_discrete_lyapunov

# exit()

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))

# create storage
x_storage = np.zeros([len(rng),len(x)])
A = np.zeros([len(x),len(x),len(rng)])
B = np.zeros([len(x),len(u),len(rng)])
C = np.zeros([len(output_vars),len(x),len(rng)])
D = np.zeros([len(output_vars),len(u),len(rng)])



bar = progressbar.ProgressBar(maxval=len(rng)).start()

tic()

for idx, val in enumerate(rng):
    
    #----------------------------------------#
    #------------linearise model-------------#
    #----------------------------------------#
    
    [A[:,:,idx], B[:,:,idx], C[:,:,idx], D[:,:,idx]] = linearise(x, u, output_vars, fi_flag, nlplant)
    
    #----------------------------------------#
    #--------------Take Action---------------#
    #----------------------------------------#
    
    # MPC prediction using squiggly C and M matrices
    #CC, MM = calc_MC(paras_mpc[0], A[:,:,idx], B[:,:,idx], time_step)
    
    
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

