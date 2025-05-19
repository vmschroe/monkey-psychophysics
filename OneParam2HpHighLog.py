#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:11:56 2025

@author: vmschroe
"""



import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.stats import binom
import sys
import FunctionsForGeneralized as ffg
import seaborn as sns
import pickle
import time
from datetime import datetime
import subprocess
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm
import aesara.tensor as at


np.random.seed(12345)

num_post_samps = 1000

thetas_fixed = np.array([0.01, 0.05, -2.8, 0.1])  #gam, lam, b0, b1 = gam_h, gam_l,bet0,bet1
xi_for_sim = np.transpose(np.array([[4,16],[6,16],[13,3],[6,25]])) #narrow
xi_fixed = np.transpose(np.array([[2, 5], [2, 5], [1.15, 0.05], [1.25, 2.5]]))  # cols for each theta, rows for a and b
params_prior_scale = np.array([0.25, 0.25, -1, 1])

hhps_mvtn = np.load('hhps_prior_mv_trunc_norm.npy') #8 cols (for each xi), 2 rows (for mean and std)

with open("sim_data_Log1p2hp.pkl", "rb") as f:
    sim_data_dict = pickle.load(f)    
    
true_thetas = sim_data_dict['theta_mat']
C_data = sim_data_dict['C_mat']
N_mat = sim_data_dict['N_mat']

Ns = np.shape(C_data)[0]



# def phi_L_as(params, X):
#     """
#     params: shape (S, 4) = [gamma, lambda, beta0, beta1]
#     X: shape (N,) or (S, N)
#     Returns: psychometric probabilities, shape (S, N)
#     """
#     gamma = params[:, 0][:, None]  # (S, 1)
#     lambda_ = params[:, 1][:, None]
#     beta0 = params[:, 2][:, None]
#     beta1 = params[:, 3][:, None]

#     if X.ndim == 1:
#         X = X[None, :]  # shape (1, N) → broadcast to (S, N)

#     z = beta0 + beta1 * X
#     logistic = 1 / (1 + at.exp(-z))
#     return gamma + (1 - gamma - lambda_) * logistic

# def phi_inv_L_as(params, p):
#     gamma, lambda_, beta0, beta1 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
#     inner = (gamma - p) / (-1 + lambda_ + p)
#     inner = at.clip(inner, 1e-6, 1e6)
#     return -(beta0 - at.log(inner)) / beta1

# def PSE_L_as(params):
#     return phi_inv_L_aesara(params, 0.5)

# def JND_L_as(params):
#     x25 = phi_inv_L_aesara(params, 0.25)
#     x75 = phi_inv_L_aesara(params, 0.75)
#     return 0.5 * (x75 - x25)




with pm.Model() as model:
    # Hyperpriors (truncated normals)
    mu_xi = hhps_mvtn[0]
    sigma_xi = hhps_mvtn[1]
    xi = pm.TruncatedNormal("xi", mu=mu_xi, sigma=sigma_xi, lower=0, shape=8)
    
    alpha_h, beta_h = xi[0], xi[1]
    alpha_l, beta_l = xi[2], xi[3]
    a0, b0 = xi[4], xi[5]
    a1, b1 = xi[6], xi[7]

    S = Ns
    L_h = pm.Beta("gamma_h", alpha=alpha_h, beta=beta_h, shape=S)
    gamma_h =  pm.Deterministic("gamma_h", L_h*params_prior_scale[0])
    L_l = pm.Beta("gamma_l", alpha=alpha_l, beta=beta_l, shape=S)
    gamma_l =  pm.Deterministic("gamma_l", L_l*params_prior_scale[1])
    L0 = pm.Gamma("beta0", alpha=a0, beta=b0, shape=S)
    beta0 =  pm.Deterministic("beta0", L0*params_prior_scale[2])
    L1 = pm.Gamma("beta1", alpha=a1, beta=b1, shape=S)
    beta1 = pm.Deterministic("beta1", L1*params_prior_scale[3])

    #PSE = pm.Deterministic("PSE", PSE_L_as([gamma_h, gamma_l, beta0, beta1]))
    #JND = pm.Deterministic("JND", JND_L_as([gamma_h, gamma_l, beta0, beta1]))
    
    # Likelihood for each session
    for s in range(S):
        p = ffg.phi_L([gamma_h[s], gamma_l[s], beta0[s], beta1[s]])
        pm.Binomial(f"c_obs_{s}", n=N_mat[s], p=p, observed=C_data[s])
    
    #trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True)
