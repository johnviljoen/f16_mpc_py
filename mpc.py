#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:55:50 2021

@author: johnviljoen
"""

import numpy as np

from sim import calc_xdot, calc_out

from scipy.linalg import solve_discrete_lyapunov

from sys import exit

import scipy

# In[ discrete linear quadratic regulator ]
# from https://github.com/python-control/python-control/issues/359:
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    
    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*(x_ref[k] - x[k])
    select the states that you want considered and make x[k] the difference
    between the current x and the desired x.
      
    How to apply the function:    
        K = dlqr(A_d,B_d,Q,R)
      
    Inputs:
      A_d, B_d, Q, R  -> all numpy arrays  (simple float number not allowed)
      
    Returns:
      K: state feedback gain
    
    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ P @ B+R) @ (B.T @ P @ A))
    return K

def square_mat_degen_2d(mat, degen_idx):
    
    degen_mat = np.zeros([len(degen_idx),len(degen_idx)])
    
    for i in range(len(degen_idx)):
        
        degen_mat[:,i] = mat[degen_idx, [degen_idx[i] for x in range(len(degen_idx))]]
        
    return degen_mat

# In[]

def linearise(x, u, output_vars, fi_flag, nlplant):
    
    eps = 1e-05
    
    A = np.zeros([len(x),len(x)])
    B = np.zeros([len(x),len(u)])
    C = np.zeros([len(output_vars),len(x)])
    D = np.zeros([len(output_vars),len(u)])
    
    # Perturb each of the state variables and compute linearization
    for i in range(len(x)):
        
        dx = np.zeros([len(x),1])
        dx[i] = eps
        
        A[:, i] = (calc_xdot(x + dx, u, fi_flag, nlplant)[:,0] - calc_xdot(x, u, fi_flag, nlplant)[:,0]) / eps
        C[:, i] = (calc_out(x + dx, u, output_vars)[:,0] - calc_out(x, u, output_vars)[:,0]) / eps
        
    # Perturb each of the input variables and compute linearization
    for i in range(len(u)):
        
        du = np.zeros([len(u),1])
        du[i] = eps
                
        B[:, i] = (calc_xdot(x, u + du, fi_flag, nlplant)[:,0] - calc_xdot(x, u, fi_flag, nlplant)[:,0]) / eps
        D[:, i] = (calc_out(x, u + du, output_vars)[:,0] - calc_out(x, u, output_vars)[:,0]) / eps
    
    return A, B, C, D

# In[]

def calc_MC(A, B, hzn):

    # hzn is the horizon
    nstates = A.shape[0]
    ninputs = B.shape[1]
    
    # x0 is the initial state vector of shape (nstates, 1)
    # u is the matrix of input vectors over the course of the prediction of shape (ninputs,horizon)
    
    # initialise CC, MM, Bz
    CC = np.zeros([nstates*hzn, ninputs*hzn])
    MM = np.zeros([nstates*hzn, nstates])
    Bz = np.zeros([nstates, ninputs])
    
    for i in range(hzn):
        MM[nstates*i:nstates*(i+1),:] = np.linalg.matrix_power(A,i+1)
        for j in range(hzn):
            if i-j >= 0:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = np.matmul(np.linalg.matrix_power(A,(i-j)),B)
            else:
                CC[nstates*i:nstates*(i+1),ninputs*j:ninputs*(j+1)] = Bz

    return MM, CC

# In[]

def calc_x_seq(A_d, B_d, x0, u_seq, hzn):
    
    # find MM, CC
    MM, CC = calc_MC(A_d, B_d, hzn)
    
    return np.matmul(MM,x0) + np.matmul(CC,u_seq)

# In[]

def calc_HFG(A_d, B_d, C_d, K, R, hzn):
    
    # calculate Q_mat
    Q = np.matmul(C_d.T, C_d)
    
    # calc R_mat
    R_mat = np.eye(B_d.shape[1]) * R
    
    Q_bar = solve_discrete_lyapunov((A_d + np.matmul(B_d, K)).T, Q + np.matmul(np.matmul(K.T,R_mat), K))
    
    Q_mat = dmom(Q, hzn)
    
    Q_mat[-Q.shape[0]:, -Q.shape[1]:] = Q_bar
    
    MM, CC = calc_MC(A_d, B_d, hzn)
    
    H = np.matmul(np.matmul(CC.T,Q_mat),CC) + dmom(R_mat,hzn)
    F = np.matmul(np.matmul(CC.T,Q_mat),MM)
    G = np.matmul(np.matmul(MM.T,Q_mat),MM)
    
    return H, F, G

# In[]

def dmom(mat, num_mats):
    # diagonal matrix of matrices -> dmom
    
    # dimension extraction
    nrows = mat.shape[0]
    ncols = mat.shape[1]
    
    # matrix of matrices matomats -> I thought it sounded cool
    matomats = np.zeros((nrows*num_mats,ncols*num_mats))
    
    for i in range(num_mats):
        for j in range(num_mats):
            if i == j:
                matomats[nrows*i:nrows*(i+1),ncols*j:ncols*(j+1)] = mat
                
    return matomats

# In[]

# def calc_HFG(A, B, C, hzn, Q, R):
    
#     MM, CC = calc_MC(hzn, A, B, 1)

#     Q = np.matmul(C.T,C)
    
#     Q_full = dmom(Q, hzn)
#     # Q_full = np.eye(hzn)
    
#     R_full = np.eye(hzn) * 0.01
    
#     H = np.matmul(np.matmul(CC.T, Q_full),CC) + R_full
    
#     F = np.matmul(np.matmul(CC.T, Q_full), MM)
    
#     G = np.matmul(np.matmul(MM.T, Q_full), MM)
    
#     return H, F, G

# In[]

# dual mode predicted HFG
def calc_dm_HFG(A, B, C, K, hzn, Q, R):
    
    MM, CC = calc_MC(hzn, A, B, 1)

    Q = np.matmul(C.T,C)
    
    Q_full = dmom(Q, hzn)
    # Q_full = np.eye(hzn)
    
    rhs = Q + np.matmul(np.matmul(K.T,R), K)
    
    Qbar = np.array([])
    
    R_full = np.eye(hzn) * 0.01
    
    H = np.matmul(np.matmul(CC.T, Q_full),CC) + R_full
    
    F = np.matmul(np.matmul(CC.T, Q_full), MM)
    
    G = np.matmul(np.matmul(MM.T, Q_full), MM)
    
    return H, F, G