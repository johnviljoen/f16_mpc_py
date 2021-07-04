#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:06:54 2021

@author: johnviljoen
"""

import numpy as np

import os

from ctypes import CDLL
import ctypes

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
# In[]

import control

control.iosys.InputOutputSystem.linearize

# port the MATLAB 'linmodv5' according to: https://uk.mathworks.com/help/slcontrol/ug/linearizing-nonlinear-models.html#:~:text=Linearization%20is%20a%20linear%20approximation,y%20%3D%202%20x%20%E2%88%92%201%20.
def linearise(x, u, lin_paras):
    
    nlplant = lin_paras.get("nlplant")
    fi_flag = lin_paras.get("fi_flag")
    eps = lin_paras.get("eps")
    
    # para[0] = delta perturb value
    # para[1] = linearisation time
    # para[2] = flag for removing redundant states
    para = [eps, 0, 0]
    
    # x, u are the current state and inputs respectively
    # xpert and upert are their perturbations for linearisation
    # all values are default according to the aforementioned MATLAB documentation
    xpert = para[0] + 1e-3*para[0]*abs(x)
    upert = para[0] + 1e-3*para[0]*abs(u)
    
    # create array to find xdot
    xdot_nopert = np.zeros(18)
    xdot_pert = np.zeros(18)
    
    # find non perturbed xdot
    nlplant.Nlplant(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(xdot_nopert.ctypes.data), ctypes.c_int(fi_flag))
    nlplant.Nlplant(ctypes.c_void_p(xpert.ctypes.data), ctypes.c_void_p(xdot_pert.ctypes.data), ctypes.c_int(fi_flag))
        
    # Create empty matrices that we can fill up with linearizations
    A = np.zeros((17, 17))     # Dynamics matrix (nstates, nstates)
    B = np.zeros((17, 4))      # Input matrix (nstates, ninputs)
    C = np.zeros((6, 17))      # Output matrix (noutputs, nstates)
    D = np.zeros((6, 4))       # Direct term (noutputs, ninputs)
    
    # Perturb each of the state variables and compute linearization
    for i in range(17):
        dx = np.zeros((17,))
        dx[i] = eps
        A[:, i] = (xdot_pert[0:17] - xdot_nopert[0:17]) / eps
        #C[:, i] = (self._out(t, x0 + dx, u0) - xpert[3,4,6,9,10,11]) / eps

    # Perturb each of the input variables and compute linearization
    # for i in range(ninputs):
    #     du = np.zeros((ninputs,))
    #     du[i] = eps
    #     B[:, i] = (self._rhs(t, x0, u0 + du) - F0) / eps
    #     D[:, i] = (self._out(t, x0, u0 + du) - H0) / eps
    
    return A

so_file = os.getcwd() + "/C/nlplant_xcg35.so"

# extract nlplant
nlplant = CDLL(so_file)

lin_paras = {
    "nlplant": nlplant,
    "fi_flag": 1,
    "eps": 1e-06
    }


x = np.array([ 1.97022009e+03,  1.85914048e+00,  1.03286930e+04,  1.12673860e-01,
        9.81710712e-01,  7.01193663e-02,  6.01968835e+02,  4.11195845e-01,
       -4.03661643e-03,  1.12353455e-01,  4.68585209e-01,  3.90562575e-02,
        2.88664680e+03, -2.03850000e+00, -8.75770000e-02, -3.87700000e-02,
        2.49913499e+01])

u = np.array([ 2.8866468e+03, -2.0385000e+00, -8.7577000e-02, -3.8770000e-02])

A = linearise(x,u,lin_paras)

# In[]

