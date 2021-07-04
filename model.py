#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 21:31:03 2021

@author: johnviljoen
"""

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

import control

from utils import tic, toc

# In[]

#----------------------------------------------------------------------------#
#-------------------------prepare data for nlplant.c-------------------------#
#----------------------------------------------------------------------------#

# unwrap initial inputs for debugging
npos, epos, h, phi, theta, psi, vt, alpha, beta, P, Q, R, P3, dh, da, dr, lef, fi_flag = initial_state_vector
# Ixx, Iyy, Izz, Ixz, weight, b, S, cbar, He, x_cg, x_cg_ref = aircraft_properties
time_step, time_start, time_end, g, stability_flag = simulation_parameters
fi_flag = 1

# convert inputs to correct units for nlplant.c
m2f = 3.28084 # metres to feet conversion
f2m = 1/m2f # feet to metres conversion
initial_state_vector_ft_rad = np.array([npos*m2f, epos*m2f, h*m2f, phi, theta, psi, vt*m2f, alpha, beta, P, Q, R, P3, dh, da, dr, lef])

# create interface with c shared library .so file in folder "C"
if stability_flag == 1:
    so_file = os.getcwd() + "/C/nlplant_xcg35.so"
elif stability_flag == 0:
    so_file = os.getcwd() + "/C/nlplant_xcg25.so"
    
nlplant = CDLL(so_file)

# initialise xu and xdot
x = initial_state_vector_ft_rad
xdot = np.zeros(17)

# initialise Mach, qbar, ps storage
coeff = np.zeros(3)

# initialise LF_state
LF_state = -x[7] * 180/pi

# In[]

#----------------------------------------------------------------------------#
#---------------control systems library attempted implementation-------------#
#----------------------------------------------------------------------------#

def updfcn(t, x, u, params):
    
    # Parameter setup
    fi_flag = params.get('fi_flag')
    dt = params.get('dt')
    
    # initialise variable to pass pointer of which to the C
    xdot = np.zeros(17)
    
    # model actuators
    
    
    # Call nlplant for xdot
    nlplant.Nlplant(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(xdot.ctypes.data), ctypes.c_int(fi_flag))
    
    dx = xdot * dt
    
    return dx

# function returning the output at the given state -> redundant right now
def outfcn(t, x, u, params):
    
    return x

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))

params = {
    "fi_flag": 1,
    "dt": 0.001
    }

# tic()

# for idx, val in enumerate(rng):

#     xdot = updfcn(1,x,1,params)

# toc()

# In[]

def predprey_rhs(t, x, u, params):
    # Parameter setup
    a = params.get('a', 3.2)
    b = params.get('b', 0.6)
    c = params.get('c', 50.)
    d = params.get('d', 0.56)
    k = params.get('k', 125)
    r = params.get('r', 1.6)

    # Map the states into local variable names
    H = x[0]
    L = x[1]

    # Compute the control action (only allow addition of food)
    u_0 = u if u > 0 else 0

    # Compute the discrete updates
    dH = (r + u_0) * H * (1 - H/k) - (a * H * L)/(c + H)
    dL = b * (a * H *  L)/(c + H) - d * L
    
    moo = [dH, dL]
    
    print(type(moo))
    
    return [dH, dL]

updfcn(1,x,1,params)

params2 = {
    "a":1,
    "b":1,
    "c":1,
    "d":1,
    "k":1,
    "r":1,
    }
predprey_rhs(1,x,1,params2)

# In[]

io_predprey = control.NonlinearIOSystem(
    predprey_rhs, None, inputs=('u'), outputs=('H', 'L'),
    states=('H', 'L'), name='predprey')

io_F16 = control.NonlinearIOSystem(predprey_rhs, None, )

#io_F16 = control.NonlinearIOSystem(predprey_rhs, None, inputs=4, outputs=17, states=17, params={"fi_flag":1,"dt":0.001}, name='F16')
    
x0 = x
u = np.zeros([4,10000])
t = np.linspace(0,10,10000)

#io_F16

# In[]

#t,y = control.input_output_response(io_F16, t, u, x0)

#F16 = control.iosys.NonlinearIOSystem(updfcn, kwargs)

lin_F16 = io_F16.linearize(x0, u[:,0])

# In[]

class F16:
    
    def __init__(self, fi_flag, stab_flag, xu, xdot, u):
        self.fi_flag = fi_flag
        self.stab_flag = stab_flag
        self.xu = xu
        self.xdot = xdot
        self.u = u
        
    def reset(self, xu, xdot, u):
        self.xu = xu
        self.xdot = xdot
        self.u = u
        
    def step(self):
        print(f'fidelity flag is {self.fi_flag}')
        
    def linmod(self):
        pass
        
f16 = F16(1,1,1,1,1)
f16.step()