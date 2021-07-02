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
initial_state_vector_ft_rad = np.array([npos*m2f, epos*m2f, h*m2f, phi, theta, psi, vt*m2f, alpha, beta, P, Q, R, P3, dh, da, dr, lef, fi_flag])

# create interface with c shared library .so file in folder "C"
if stability_flag == 1:
    so_file = os.getcwd() + "/C/nlplant_xcg35.so"
elif stability_flag == 0:
    so_file = os.getcwd() + "/C/nlplant_xcg25.so"
    
nlplant = CDLL(so_file)

# initialise xu and xdot
x = initial_state_vector_ft_rad
xdot = np.zeros(18)

# initialise Mach, qbar, ps storage
coeff = np.zeros(3)

# initialise LF_state
LF_state = -x[7] * 180/pi


def updfcn(t, x, u, params):
    
    # Parameter setup
    fi_flag = params.get('fi_flag')
    
    # initialise variable to pass pointer of which to the C
    xdot = np.zeros(18)
    
    # model actuators
    
    
    # Call nlplant for xdot
    nlplant.Nlplant(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(xdot.ctypes.data), ctypes.c_int(fi_flag))
    
    return xdot

# function returning the output at the given state -> redundant right now
def outfcn(t, x, u, params):
    
    return x

rng = np.linspace(time_start, time_end, int((time_end-time_start)/time_step))

params = {
    "fi_flag": 1
    }

tic()

for idx, val in enumerate(rng):

    xdot = updfcn(1,x,1,params)

toc()
    

#F16 = control.iosys.NonlinearIOSystem(updfcn, kwargs)

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