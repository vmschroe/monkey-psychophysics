# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:32:14 2025

@author: schro
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
sys.path.append("C:\\Users\\schro\\Miniconda3\\envs\\pymc-env\\monkey-psychophysics\\Two_Stage_Models")
import FunctionsForGeneralized as ffg
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm
import pytensor.tensor as pt  # Use pytensor instead of aesara
from arviz.stats.density_utils import kde
sys.path.append("monkey-psychophysics\\Three_Stage_Models")
from Three_Stage_Models.High3sLogFuncs import phi_L_as

x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
x_new = (x_orig - np.mean(x_orig) ) / np.std(x_orig)

def compute_PSE_JND(gamma_h, gamma_l, beta0, beta1):
    jnd_log_term = np.log(1-gamma_h)+np.log(1-gamma_l)-np.log(3-gamma_h)-np.log(3-gamma_l)
    jnd = jnd_log_term/(2*beta1)
    
    pse_log_term = np.log(1-2*gamma_l)+np.log(1-2*gamma_h)
    pse = (-beta0+pse_log_term)/beta1
    return pse, jnd

def compute_betas(gamma_h, gamma_l, pse, jnd):
    jnd_log_term = np.log(1-gamma_h)+np.log(1-gamma_l)-np.log(3-gamma_h)-np.log(3-gamma_l)
    beta1 = jnd_log_term/(2*jnd)
    
    pse_log_term = np.log(1-2*gamma_l)+np.log(1-2*gamma_h)
    beta0 = -beta1*pse + pse_log_term
    return beta0, beta1


def compute_PSE_JND_pt(gamma_h, gamma_l, beta0, beta1, eps=1e-12):
    jnd_log_term = pt.log(pt.clip(1 - gamma_h, eps, 1e12)) \
                 + pt.log(pt.clip(1 - gamma_l, eps, 1e12)) \
                 - pt.log(pt.clip(3 - gamma_h, eps, 1e12)) \
                 - pt.log(pt.clip(3 - gamma_l, eps, 1e12))
    
    jnd = jnd_log_term / (2 * pt.clip(beta1, eps, 1e12))

    pse_log_term = pt.log(pt.clip(1 - 2 * gamma_l, eps, 1e12)) \
                 + pt.log(pt.clip(1 - 2 * gamma_h, eps, 1e12))

    pse = (pse_log_term - beta0) / pt.clip(beta1, eps, 1e12)

    return pse, jnd

def compute_betas_pt(gamma_h, gamma_l, pse, jnd, eps=1e-12):
    gamma_h = pt.as_tensor_variable(gamma_h)
    gamma_l = pt.as_tensor_variable(gamma_l)
    pse = pt.as_tensor_variable(pse)
    jnd = pt.as_tensor_variable(jnd)

    jnd_log_term = pt.log(pt.clip(1 - gamma_h, eps, 1e12)) \
                 + pt.log(pt.clip(1 - gamma_l, eps, 1e12)) \
                 - pt.log(pt.clip(3 - gamma_h, eps, 1e12)) \
                 - pt.log(pt.clip(3 - gamma_l, eps, 1e12))

    beta1 = jnd_log_term / (2 * pt.clip(jnd, eps, 1e12))

    pse_log_term = pt.log(pt.clip(1 - 2 * gamma_l, eps, 1e12)) \
                 + pt.log(pt.clip(1 - 2 * gamma_h, eps, 1e12))

    beta0 = -beta1 * pse + pse_log_term
    return beta0, beta1


def rescaled_analysis(data_dict,  mu_xi = np.array([1,4,1,4,0,0.7,0.82,0.21]), sigma_xi = np.array([4,16,4,16,0.7,1.4,0.21,0.42]), lower = np.array([0,0,0,0,-1.51,0,0,0]), upper = np.array([np.inf,np.inf,np.inf,np.inf,1.51,np.inf,1.51,np.inf])
, num_post_samps=2000):
    """Samples from posterior dist. of 3-stage higherarchical model with logistic form of psych func

     Parameters:
     data_dict (dict):data_dict['C_mat'] = np array, shape = (Nsess,8), counts of "high"
                     data_dict['N_mat'] = np array, shape = (Nsess,8), nums of trials
     mu_xi (array): shape = (8,) prior means for trunc normal distributions of hyperparams
     sigma_xi (array): shape = (8,) prior stdevs for trunc normal distributions of hyperparams

     Returns:
     trace: posterior samples
    
    """
    params_prior_scale = np.array([0.25, 0.25, -1., 1.])
    
    C_data = data_dict['ld']['C_mat']
    N_mat = data_dict['ld']['N_mat']
    
    Ns = np.shape(C_data)[0]
    
    with pm.Model() as model:
        # Load prior scales as symbolic variables
        scale_gamma_h = pt.as_tensor_variable(params_prior_scale[0])
        scale_gamma_l = pt.as_tensor_variable(params_prior_scale[1])
        scale_beta0   = pt.as_tensor_variable(params_prior_scale[2])
        scale_beta1   = pt.as_tensor_variable(params_prior_scale[3])
        
        # Hyperpriors (truncated normals)
       
        xi_unbounded = pm.Normal("xi", mu=mu_xi, sigma=sigma_xi, shape=8)
        xi = pm.Bound("xi", xi_unbounded, lower=lower, upper=upper)
        
        
        alpha_h, beta_h = xi[0], xi[1]
        alpha_l, beta_l = xi[2], xi[3]
        mu_pse, sig_pse = xi[4], xi[5]
        mu_jnd, sig_jnd = xi[6], xi[7]
    
        S = Ns
        
        # Latent individual-level parameters
        L_h = pm.Beta("L_h", alpha=alpha_h, beta=beta_h, shape=S)
        gamma_h = pm.Deterministic("gamma_h", L_h * scale_gamma_h)
        
        L_l = pm.Beta("L_l", alpha=alpha_l, beta=beta_l, shape=S)
        gamma_l = pm.Deterministic("gamma_l", L_l * scale_gamma_l)
        
        PSE = pm.TruncatedNormal("PSE", mu=mu_pse, sigma=sig_pse, lower=-1.51, upper=1.51)
        JND = pm.TruncatedNormal("JND", mu=mu_jnd, sigma=sig_jnd, lower=0, upper=1.51)
        beta0_val, beta1_val = compute_betas_pt(gamma_h, gamma_l, PSE, JND)
        beta0 = pm.Deterministic("beta0", beta0_val)
        beta1 = pm.Deterministic("beta1", beta1_val)
        
        # Fixed stimulus levels
        x_vals = pt.as_tensor_variable(x_new)
        
        # Likelihood per session
        for s in range(S):
            p = phi_L_as(gamma_h[s], gamma_l[s], beta0[s], beta1[s], x_vals)
            # Fix the dimensionality issue - p might be returning shape [1, n_stim_levels]
            # but we need it to match C_data[s] which is [n_stim_levels]
            p = p.flatten()  # Flatten to ensure dimensions match
            pm.Binomial(f"c_obs_{s}", n=N_mat[s], p=p, observed=C_data[s])
        
        # Sampling
        #trace = pm.sample(num_post_samps, initvals={"xi": [1,4,1,4,0,0.7,0.82,0.21]}, return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True})
    return trace

def phi_L_pt(gam, lambda_, beta0, beta1, X):
    """
    Vectorized psychometric function.
    gam, lambda_, beta0, beta1: shape (S,)
    X: shape (n_stim_levels,)
    
    Returns: p, shape (S, n_stim_levels)
    """
    # Convert inputs to PyTensor variables if not already
    gam = pt.as_tensor_variable(gam)
    lambda_ = pt.as_tensor_variable(lambda_)
    beta0 = pt.as_tensor_variable(beta0)
    beta1 = pt.as_tensor_variable(beta1)
    X = pt.as_tensor_variable(X)

    # Reshape for broadcasting
    # (S, 1) to multiply with (1, n_stim_levels)
    gam = gam.reshape((-1, 1))   # (S, 1)
    lambda_ = lambda_.reshape((-1, 1))
    beta0 = beta0.reshape((-1, 1))
    beta1 = beta1.reshape((-1, 1))
    X = X.reshape((1, -1))           # (1, n_stim_levels)

    # Compute logistic
    z = beta0 + beta1 * X            # (S, n_stim_levels)
    logistic = 1 / (1 + pt.exp(-z)) # (S, n_stim_levels)

    # Final probabilities
    p = gam + (1 - gam - lambda_) * logistic  # (S, n_stim_levels)
    return p


#%%

with open(r"Data\psych_vecs_all.pkl", "rb") as f:
    data = pickle.load(f)  

with open(r"Data/session_summary.pkl", "rb") as f:
    sess_sum = pickle.load(f)  
    

Cagg_mats = []
Nagg_mats = []
data_dict = {}

for grp in ['ld','ln','rd','rn']:
    data_dict[grp] = {}
    C = np.array([*data['NY'][grp].values()])
    data_dict[grp]['C_mat'] = C
    Cagg_mats.append(sum(C))
    
    N = np.array([*data['N'][grp].values()])
    data_dict[grp]['N_mat'] = N
    Nagg_mats.append(sum(N))

data_dict['agg'] = {}
data_dict['agg']['C_mat'] = np.array(Cagg_mats)
data_dict['agg']['N_mat'] = np.array(Nagg_mats)



#%%
traces = {}

grps = ['ld','ln','rd','rn']

for grp in ['ld']:
    traces[grp] = rescaled_analysis(data_dict[grp])

params_prior_scale = np.array([0.25, 0.25, -1., 1.])

C_data = data_dict['ld']['C_mat']
N_mat = data_dict['ld']['N_mat']

Ns = np.shape(C_data)[0]

mu_xi = np.array([1,4,1,4,0,0.7,0.82,0.21])
sigma_xi = np.array([4,16,4,16,0.7,1.4,0.21,0.42])
lower = np.array([0,0,0,0,-1.51,0,0,0])
upper = np.array([np.inf,np.inf,np.inf,np.inf,1.51,np.inf,1.51,np.inf])

with pm.Model() as model:
    # Load prior scales as symbolic variables
    scale_gamma_h = pt.as_tensor_variable(params_prior_scale[0])
    scale_gamma_l = pt.as_tensor_variable(params_prior_scale[1])
    scale_beta0   = pt.as_tensor_variable(params_prior_scale[2])
    scale_beta1   = pt.as_tensor_variable(params_prior_scale[3])
    
    # Hyperpriors (truncated normals)
    S = Ns
    
    xi = pm.TruncatedNormal("xi",mu = mu_xi, sigma = sigma_xi, lower=lower, upper=upper, shape=8)
    
    
    alpha_h, beta_h = xi[0], xi[1]
    alpha_l, beta_l = xi[2], xi[3]
    mu_pse, sig_pse = xi[4], xi[5]
    mu_jnd, sig_jnd = xi[6], xi[7]

   
    
    # Latent individual-level parameters
    L_h = pm.Beta("L_h", alpha=alpha_h, beta=beta_h, shape=S)
    gamma_h = pm.Deterministic("gamma_h", L_h * scale_gamma_h)
    
    L_l = pm.Beta("L_l", alpha=alpha_l, beta=beta_l, shape=S)
    gamma_l = pm.Deterministic("gamma_l", L_l * scale_gamma_l)
    
    PSE = pm.TruncatedNormal("PSE", mu=mu_pse, sigma=sig_pse, lower=-1.51, upper=1.51, shape=S)
    JND = pm.TruncatedNormal("JND", mu=mu_jnd, sigma=sig_jnd, lower=0, upper=1.51, shape=S)
    beta0_val, beta1_val = compute_betas_pt(gamma_h, gamma_l, PSE, JND)
    beta0 = pm.Deterministic("beta0", beta0_val)
    beta1 = pm.Deterministic("beta1", beta1_val)
    
    # Fixed stimulus levels
    x_vals = pt.as_tensor_variable(x_new)
    
    
    p = phi_L_pt(gamma_h, gamma_l, beta0, beta1, x_vals)  # shape (S, n_stim_levels)

    # N_mat and C_data must have shape (S, n_stim_levels)
    pm.Binomial("c_obs", n=N_mat, p=p, observed=C_data)
    # # Likelihood per session
    # for s in range(S):
    #     p = phi_L_as(gamma_h[s], gamma_l[s], beta0[s], beta1[s], x_vals)
    #     # Fix the dimensionality issue - p might be returning shape [1, n_stim_levels]
    #     # but we need it to match C_data[s] which is [n_stim_levels]
    #     p = p.flatten()  # Flatten to ensure dimensions match
    #     pm.Binomial(f"c_obs_{s}", n=N_mat[s], p=p, observed=C_data[s])
    
    # Sampling
    trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True})


#%%

with model:
    trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True})



with model:
    prior = pm.sample_prior_predictive()
    
## beta0 values are coming out negative for some reason