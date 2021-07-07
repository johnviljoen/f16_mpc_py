# In[] imports

# from ctypes import *
from ctypes import CDLL
#import ctypes
import os

# import numpy and sin, cos for convenience
import numpy as np
from numpy import pi

# handbuilt functions for all this
from utils import tic, toc, vis
from trim import trim
from sim import calc_xdot, upd_sim, calc_out
from mpc import linearise, MC, dmom

# import progressbar for convenience
import progressbar

# import parameters
from parameters import initial_state_vector_ft_rad, simulation_parameters

# import exit() function for debugging
from sys import exit

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

def x_traj(hzn, A, B, x0):
    
    x1 = np.matmul(A,x0) + np.matmul(B, u_seq[0,:])
    x2 = np.matmul(np.linalg.matrix_power(A,2),x0) + np.matmul(np.matmul(A, B),u_seq[0]) + np.matmul(B, u_seq[1])
    
    return x1, x2

def x_traj_nl(hzn, x0, u_seq, nlplant, dt):
    x_temp = np.zeros([len(x0)])
    x_seq = np.zeros((hzn,len(x0)))
    for idx in range(hzn):
        x_temp = upd_sim(x, u[idx,:], fi_flag, dt, nlplant)
        x_seq[idx,:] = x_temp
    return x_seq 
        

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))
bar = progressbar.ProgressBar(maxval=len(rng)).start()

#linearisation eps
eps = 1e-05

output_vars = [6,7,8,9,10,11]

# trim aircraft
h_t = 10000
v_t = 700

x, opt_res = trim(h_t, v_t, fi_flag, nlplant)

u = x[12:16]

# create storage
x_storage = np.zeros([len(rng),len(x)])
A = np.zeros([len(x),len(x),len(rng)])
B = np.zeros([len(x),len(u),len(rng)])
C = np.zeros([len(output_vars),len(x),len(rng)])
D = np.zeros([len(output_vars),len(u),len(rng)])

hzn = 5

# A,B,C,D = linearise(x, u, output_vars, fi_flag, nlplant, eps)

# # the correct shape according to lec2-p4 of slides
# u_seq = np.concatenate([u,u,u,u,u])

# # the correct shape according to lec2-p4 slides also
# #x_seq = np.array([x1, x2, x3,...@t0],[x1, x2, x3,.. @t1]...)

# MM, CC = MC(hzn, A, B, 1)

# x_seq = np.matmul(MM, x) + np.matmul(CC, u_seq)

# xdotA1 = np.matmul(A,x) + np.matmul(B,u)
# x1 = x + xdotA1*time_step
# xdotA2 = np.matmul(A,x1) + np.matmul(B,u)
# x2 = x1 + xdotA2*time_step


# 
# Q = np.matmul(C.T,C)

# 
# Q_mat = dmom(Q, hzn)

# terminal weight matrix
# P = Q # make equal to Q for this

# x1, x2 = x_traj(hzn, A, B, x)

#exit()

tic()

for idx, val in enumerate(rng):
    
    #----------------------------------------#
    #------------linearise model-------------#
    #----------------------------------------#
    
    [A[:,:,idx], B[:,:,idx], C[:,:,idx], D[:,:,idx]] = linearise(x, u, output_vars, fi_flag, nlplant, eps)
    
    #----------------------------------------#
    #--------------Take Action---------------#
    #----------------------------------------#
    
    # predict
    
    CC, MM = MC(hzn, A[:,:,idx], B[:,:,idx], time_step)
    
    
    
    #----------------------------------------#
    #--------------Integrator----------------#
    #----------------------------------------#    
    
    x = upd_sim(x, u, fi_flag, time_step, nlplant)
    
    #----------------------------------------#
    #------------Store History---------------#
    #----------------------------------------#
    
    x_storage[idx,:] = x
    
    bar.update(idx)

toc()

# In[]

#----------------------------------------------------------------------------#
#---------------------------------Visualise----------------------------------#
#----------------------------------------------------------------------------#

#%matplotlib qt

vis(x_storage, rng)

# %%
