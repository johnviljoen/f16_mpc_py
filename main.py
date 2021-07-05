# In[] imports

# from ctypes import *
from ctypes import CDLL
import ctypes
import os

# import numpy and sin, cos for convenience
import numpy as np
from numpy import pi

# import matplotlib for visualisation
import matplotlib.pyplot as plt

# import progressbar for convenience
import progressbar

# import parameters
from parameters import initial_state_vector, simulation_parameters

# import exit() function for debugging
from sys import exit

# In[]

#----------------------------------------------------------------------------#
#-------------------------prepare data for nlplant.c-------------------------#
#----------------------------------------------------------------------------#

# unwrap initial inputs for debugging
npos, epos, h, phi, theta, psi, vt, alpha, beta, P, Q, R, P3, dh, da, dr, lef, fi_flag = initial_state_vector
# Ixx, Iyy, Izz, Ixz, weight, b, S, cbar, He, x_cg, x_cg_ref = aircraft_properties
time_step, time_start, time_end, g, stab_flag = simulation_parameters
fi_flag = 1

# convert inputs to correct units for nlplant.c
m2f = 3.28084 # metres to feet conversion
f2m = 1/m2f # feet to metres conversion
initial_state_vector_ft_rad = np.array([npos*m2f, epos*m2f, h*m2f, phi, theta, psi, vt*m2f, alpha, beta, P, Q, R, P3, dh, da, dr, lef, -alpha*180/pi])

# create interface with c shared library .so file in folder "C"
if stab_flag == 1:
    so_file = os.getcwd() + "/C/nlplant_xcg35.so"
elif stab_flag == 0:
    so_file = os.getcwd() + "/C/nlplant_xcg25.so"
    
nlplant = CDLL(so_file)

# initialise x and xdot
x = initial_state_vector_ft_rad
xdot = np.zeros(18)

# initialise Mach, qbar, ps storage
coeff = np.zeros(3)

from utils import tic, toc

# In[]

#----------------------------------------------------------------------------#
#---------------------------------Simulate-----------------------------------#
#----------------------------------------------------------------------------#

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))
bar = progressbar.ProgressBar(maxval=len(rng)).start()

#linearisation eps
eps = 1e-05

T_cmd = 2886.6468
dstab_cmd = -2.0385
ail_cmd = -0.087577
rud_cmd = -0.03877

u = [T_cmd, dstab_cmd, ail_cmd, rud_cmd]

output_vars = [6,7,8,9,10,11]

# create storage
x_storage = np.zeros([len(rng),len(x)])
xdot_storage = np.zeros([len(rng),len(xdot)])
A = np.zeros([len(x),len(x),len(rng)])
B = np.zeros([len(x),len(u),len(rng)])
C = np.zeros([len(output_vars),len(x),len(rng)])
D = np.zeros([len(output_vars),len(u),len(rng)])


def upd_thrust(T_cmd, T_state, time_step):
    # command saturation
    T_cmd = np.clip(T_cmd,1000,19000)
    # rate saturation
    T_err = np.clip(T_cmd - T_state, -10000, 10000)
    # integrate
    T_state += T_err*time_step
    return T_state, T_err

def upd_dstab(dstab_cmd, dstab_state, time_step):
    # command saturation
    dstab_cmd = np.clip(dstab_cmd,-25,25)
    # rate saturation
    dstab_err = np.clip(20.2*(dstab_cmd - dstab_state), -60, 60)
    # integrate
    dstab_state += dstab_err*time_step
    return dstab_state, dstab_err

def upd_ail(ail_cmd, ail_state, time_step):
    # command saturation
    ail_cmd = np.clip(ail_cmd,-21.5,21.5)
    # rate saturation
    ail_err = np.clip(20.2*(ail_cmd - ail_state), -80, 80)
    # integrate
    ail_state += ail_err*time_step
    return ail_state, ail_err

def upd_rud(rud_cmd, rud_state, time_step):
    # command saturation
    rud_cmd = np.clip(rud_cmd,-30,30)
    # rate saturation
    rud_err = np.clip(20.2*(rud_cmd - rud_state), -120, 120)
    # integrate
    rud_state += rud_err*time_step
    return rud_state, rud_err

def upd_lef(lef_state_1, lef_state_2, time_step, nlplant):
    
    nlplant.atmos(ctypes.c_double(x[2]),ctypes.c_double(x[6]),ctypes.c_void_p(coeff.ctypes.data))
    atmos_out = coeff[1]/coeff[2] * 9.05
    alpha_deg = x[7]*180/pi
    
    LF_err = alpha_deg - (lef_state_1 + (2 * alpha_deg))
    lef_state_1 += LF_err*7.25*time_step
    LF_out = (lef_state_1 + (2 * alpha_deg)) * 1.38
    
    lef_cmd = LF_out + 1.45 - atmos_out
    
    # command saturation
    lef_cmd = np.clip(lef_cmd,0,25)
    # rate saturation
    lef_err = np.clip((1/0.136) * (lef_cmd - lef_state_2),-25,25)
    # integrate
    lef_state_2 += lef_err*time_step
    
    return lef_state_1, lef_state_2, LF_err*7.25, lef_err

def calc_xdot(x, u, fi_flag, nlplant):
    
    # initialise variables
    xdot = np.zeros(18)
    temp = np.zeros(6)
    
    #--------------Thrust Model--------------#
    temp[0] = upd_thrust(u[0], x[12], time_step)[1]
    #--------------Dstab Model---------------#
    temp[1] = upd_dstab(u[1], x[13], time_step)[1]
    #-------------aileron model--------------#
    temp[2] = upd_ail(u[2], x[14], time_step)[1]
    #--------------rudder model--------------#
    temp[3] = upd_rud(u[3], x[15], time_step)[1]
    #--------leading edge flap model---------#
    temp[5], temp[4] = upd_lef(x[17], x[16], time_step, nlplant)[2:4]
    
    #----------run nlplant for xdot----------#
    nlplant.Nlplant(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(xdot.ctypes.data), ctypes.c_int(fi_flag))    
    
    xdot[12:18] = temp
    
    return xdot

def upd_sim(x, u, fi_flag, time_step, nlplant):
    
    # find xdot
    xdot = calc_xdot(x, u, fi_flag, nlplant)
    
    # update x
    x += xdot*time_step
    
    return x

def calc_out(x, u, output_vars):
    # return the variables    
    return x[output_vars]

def linearise(x, u, output_vars, fi_flag, nlplant, eps):
    
    A = np.zeros([len(x),len(x)])
    B = np.zeros([len(x),len(u)])
    C = np.zeros([len(output_vars),len(x)])
    D = np.zeros([len(output_vars),len(u)])
    
    # Perturb each of the state variables and compute linearization
    for i in range(len(x)):
        
        dx = np.zeros((len(x),))
        dx[i] = eps
        
        A[:, i] = (calc_xdot(x + dx, u, fi_flag, nlplant) - calc_xdot(x, u, fi_flag, nlplant)) / eps
        C[:, i] = (calc_out(x + dx, u, output_vars) - calc_out(x, u, output_vars)) / eps
        
    # Perturb each of the input variables and compute linearization
    for i in range(len(u)):
        
        du = np.zeros((len(u),))
        du[i] = eps
                
        B[:, i] = (calc_xdot(x, u + du, fi_flag, nlplant) - calc_xdot(x, u, fi_flag, nlplant)) / eps
        D[:, i] = (calc_out(x, u + du, output_vars) - calc_out(x, u, output_vars)) / eps
    
    return A, B, C, D

tic()

for idx, val in enumerate(rng):
    
    #----------------------------------------#
    #------------linearise model-------------#
    #----------------------------------------#
    
    [A[:,:,idx], B[:,:,idx], C[:,:,idx], D[:,:,idx]] = linearise(x, u, output_vars, fi_flag, nlplant, eps)
    
    #----------------------------------------#
    #--------------Take Action---------------#
    #----------------------------------------#
    
    x = upd_sim(x, u, fi_flag, time_step, nlplant)
    
    #----------------------------------------#
    #------------Store History---------------#
    #----------------------------------------#
    
    x_storage[idx,:] = x
    xdot_storage[idx,:] = xdot
    
    bar.update(idx)

toc()

# In[]

#----------------------------------------------------------------------------#
#---------------------------------Visualise----------------------------------#
#----------------------------------------------------------------------------#

#%matplotlib qt

fig, axs = plt.subplots(12, 1)
#fig.suptitle('Vertically stacked subplots')
axs[0].plot(rng, x_storage[:,0])
axs[0].set_ylabel('npos (ft)')

axs[1].plot(rng, x_storage[:,1])
axs[1].set_ylabel('epos (ft)')

axs[2].plot(rng, x_storage[:,2])
axs[2].set_ylabel('h (ft)')

axs[3].plot(rng, x_storage[:,3])
axs[3].set_ylabel('$\phi$ (rad)')

axs[4].plot(rng, x_storage[:,4])
axs[4].set_ylabel('$\theta$ (rad)')

axs[5].plot(rng, x_storage[:,5])
axs[5].set_ylabel('$\psi$ (rad)')

axs[6].plot(rng, x_storage[:,6])
axs[6].set_ylabel("V_t (ft/s)")

axs[7].plot(rng, x_storage[:,7]*180/pi)
axs[7].set_ylabel('alpha (deg)')

axs[8].plot(rng, x_storage[:,8]*180/pi)
axs[8].set_ylabel('beta (deg)')

axs[9].plot(rng, x_storage[:,9]*180/pi)
axs[9].set_ylabel('p (deg/s)')

axs[10].plot(rng, x_storage[:,10]*180/pi)
axs[10].set_ylabel('q (deg/s)')

axs[11].plot(rng, x_storage[:,11]*180/pi)
axs[11].set_ylabel('r (deg/s)')
axs[11].set_xlabel('time (s)')

fig2, axs2 = plt.subplots(5,1)

axs2[0].plot(rng, x_storage[:,12])
axs2[0].set_ylabel('P3')

axs2[1].plot(rng, x_storage[:,13])
axs2[1].set_ylabel('dh')

axs2[2].plot(rng, x_storage[:,14])
axs2[2].set_ylabel('da')

axs2[3].plot(rng, x_storage[:,15])
axs2[3].set_ylabel('dr')

axs2[4].plot(rng, x_storage[:,16])
axs2[4].set_ylabel('lef')


# %%
