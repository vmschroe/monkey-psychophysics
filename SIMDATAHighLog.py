#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:11:56 2025

@author: vmschroe
"""


import scipy.statsgp
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import os
import math
from scipy.stats import binom
from scipy.stats import truncnorm
from scipy.stats import beta
from scipy.stats import gamma

import sys
import FunctionsForGeneralized as ffg
import seaborn as sns
import pickle
import time
import sys
from datetime import datetime
import subprocess

np.random.seed(12345)
# Simulating for higherarchical weibul

# params = gam, lam, b0, b1 = gam_h, gam_l,bet0,bet1
#### NOTE: For a GAMMA distribution Gamma[alpha,beta]:
    #       alpha = shape, beta = rate
    #       scipy.stats.gamma : a = alpha = shape, scale = 1/beta = 1/rate
    #       pm.Gamma : alpha = shape, beta = rate
    
#### NOTE: For a BETA distribution Beta[alpha,beta]:
    #       alpha, beta are both shape parameters
    #       scipy.stats.beta : a = alpha, b = beta
    #       pm.Beta : alpha, beta
    
    
thetas_fixed = np.array([0.01, 0.05, -2.8, 0.1])  #gam, lam, b0, b1 = gam_h, gam_l,bet0,bet1
xi_for_sim = np.transpose(np.array([[4,16],[6,16],[13,3],[6,25]])) #narrow
xi_fixed = np.transpose(np.array([[2, 5], [2, 5], [1.15, 0.05], [1.25, 2.5]]))  # cols for each theta, rows for a and b
params_prior_scale = np.array([0.25, 0.25, -1, 1])

hhps_mvtn = np.load('hhps_prior_mv_trunc_norm.npy') #8 cols (for each xi), 2 rows (for mean and std)

def sim_xi(hhps, low_trunc = 0.00001):
    xi = []
    b = np.inf
    for i in range(8):
        mu = hhps[0,i]
        sig = hhps[1,i]
        a=(low_trunc-mu)/sig
        xi_i = truncnorm.rvs(a, b, loc=mu, scale=sig)
        xi.append(xi_i)
    xi_mat = np.transpose( np.array(xi).reshape(4,2) )
    return xi_mat

def sim_theta(xi_mat, Nsess):
    #gamma_h
    j=0
    gamh_sim = params_prior_scale[j]*beta.rvs(a = xi_mat[0,j], b = xi_mat[1,j], size = Nsess)
    #gamma_l
    j=1
    gaml_sim = params_prior_scale[j]*beta.rvs(a = xi_mat[0,j], b = xi_mat[1,j], size = Nsess)
    #beta_0
    j=2
    bet0_sim = params_prior_scale[j]*gamma.rvs(a = xi_mat[0,j], scale = 1/xi_mat[1,j], size = Nsess)
    #beta_1
    j=3
    bet1_sim = params_prior_scale[j]*gamma.rvs(a = xi_mat[0,j], scale = 1/xi_mat[1,j], size = Nsess)
    theta_mat = np.transpose(np.vstack([gamh_sim,gaml_sim,bet0_sim,bet1_sim]))
    return theta_mat

def sim_C_data(theta_mat,N_mat=10):
    theta_mat = np.array(theta_mat)
    if theta_mat.ndim == 1:
        theta_mat = theta_mat.reshape(1,-1)
        if np.shape(theta_mat)[1] != 4:
            return "Check Theta!"
    theta_mat = theta_mat.reshape([-1,4])
    Nsess = np.shape(theta_mat)[0]
    if type(N_mat) == int:
        N_mat = np.full((Nsess,8),N_mat)
    C_mat = []
    for s in range(Nsess):
        nsvec = N_mat[s]
        theta_s = theta_mat[s]
        ps = ffg.phi_L(theta_s)
        c_s = binom.rvs(nsvec,ps)
        C_mat.append(c_s)
    C_mat = np.array(C_mat)
    return [C_mat, N_mat]
        
def gen_sim_data(xi = xi_for_sim, Nsess = 45, N_mat=10, fix_theta_indices = None):
    theta_mat_sim = sim_theta(xi_mat=xi, Nsess=Nsess)
    if fix_theta_indices != None:
        for i in fix_theta_indices:
            theta_mat_sim[:,i] = np.full(Nsess,thetas_fixed[i])
    [C_mat, N_mat] = sim_C_data(theta_mat_sim,N_mat=N_mat)
    sim_data = {'theta_mat':theta_mat_sim,'C_mat':C_mat,'N_mat':N_mat}
    return sim_data

sim_data_Log1p2hp = gen_sim_data(fix_theta_indices = [0,1,3])
sim_data_Log4p8hp = gen_sim_data()

# Save the dictionary to a pickle file
with open('sim_data_Log1p2hp.pkl', 'wb') as file:
    pickle.dump(sim_data_Log1p2hp, file)

with open('sim_data_Log4p8hp.pkl', 'wb') as file:
    pickle.dump(sim_data_Log4p8hp, file)



# with open("file_name.pkl", "rb") as f:
#     variable_name = pickle.load(f)







# # checking simulation of xis
# for j in range(100):
#     xi = sim_xi(hhps=hhps_mvtn)
#     theta = sim_theta(xi, Nsess=5)
#     x_vals = np.linspace(0,60,200)
#     for i in range(5):
#         y_vals = ffg.phi_L(theta[i],x_vals)
#         plt.plot(x_vals,y_vals)
    
# #checking sim of thetas
# theta = sim_theta(xi_for_sim, Nsess=200)
# x_vals = np.linspace(0,60,200)
# for i in range(200):
#     y_vals = ffg.phi_L(theta[i],x_vals)
#     plt.plot(x_vals,y_vals)
