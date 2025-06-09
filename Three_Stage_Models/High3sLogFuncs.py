# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:57:29 2025

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
sys.path.append("Two_Stage_Models")
import FunctionsForGeneralized as ffg
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm
import pytensor.tensor as pt  # Use pytensor instead of aesara
from arviz.stats.density_utils import kde


# Fixed versions of the psychometric functions for PyMC 5.x
def phi_L_as(gamma, lambda_, beta0, beta1, X):
    """
    Psychometric function using logistic function.
    Returns probabilities of positive response given stimulus intensity X.
    Parameters should be scalar values to avoid broadcasting issues.
    """
    # Ensure X is properly shaped
    if isinstance(X, np.ndarray):
        X = pt.as_tensor_variable(X)
    
    # Handle scalar inputs differently
    if gamma.ndim == 0:  # Scalar inputs
        z = beta0 + beta1 * X
        logistic = 1 / (1 + pt.exp(-z))
        return gamma + (1 - gamma - lambda_) * logistic
    else:  # Vector inputs for a subject
        # Reshape for broadcasting
        X_reshaped = X.reshape((1, -1)) if X.ndim == 1 else X
        gamma_reshaped = gamma.reshape((-1, 1)) if gamma.ndim > 0 else gamma
        lambda_reshaped = lambda_.reshape((-1, 1)) if lambda_.ndim > 0 else lambda_
        beta0_reshaped = beta0.reshape((-1, 1)) if beta0.ndim > 0 else beta0
        beta1_reshaped = beta1.reshape((-1, 1)) if beta1.ndim > 0 else beta1
        
        z = beta0_reshaped + beta1_reshaped * X_reshaped
        logistic = 1 / (1 + pt.exp(-z))
        return gamma_reshaped + (1 - gamma_reshaped - lambda_reshaped) * logistic


def phi_inv_L_as(gamma, lambda_, beta0, beta1, p_scalar):
    """
    Inverse psychometric function.
    Returns stimulus intensity that would produce a given probability p_scalar.
    """
    # Create tensor of p_scalar with correct shape for broadcasting
    if gamma.ndim == 0:  # Scalar case
        p = p_scalar
    else:  # Vector case
        p = pt.ones_like(gamma) * p_scalar
    
    # Calculate inner value with clipping to avoid numerical issues
    inner = (gamma - p) / (-1 + lambda_ + p)
    inner = pt.clip(inner, 1e-6, 1e6)  
    
    # Calculate inverse
    return -(beta0 - pt.log(inner)) / beta1


def PSE_L_as(gamma, lambda_, beta0, beta1):
    return phi_inv_L_as(gamma, lambda_, beta0, beta1, 0.5)


def JND_L_as(gamma, lambda_, beta0, beta1):
    x25 = phi_inv_L_as(gamma, lambda_, beta0, beta1, 0.25)
    x75 = phi_inv_L_as(gamma, lambda_, beta0, beta1, 0.75)
    return 0.5 * (x75 - x25)

hhps_mvtn = np.load('Three_Stage_Models/hhps_prior_mv_trunc_norm.npy') #8 cols (for each xi), 2 rows (for mean and std)


def HighLogAnalysis(data_dict, mu_xi = np.full(8,8), sigma_xi = np.full(8,16), num_post_samps=2000):
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
    
    C_data = data_dict['C_mat']
    N_mat = data_dict['N_mat']
    
    Ns = np.shape(C_data)[0]
    
    with pm.Model() as model:
        # Load prior scales as symbolic variables
        scale_gamma_h = pt.as_tensor_variable(params_prior_scale[0])
        scale_gamma_l = pt.as_tensor_variable(params_prior_scale[1])
        scale_beta0   = pt.as_tensor_variable(params_prior_scale[2])
        scale_beta1   = pt.as_tensor_variable(params_prior_scale[3])
        
        # Hyperpriors (truncated normals)
        xi = pm.TruncatedNormal("xi", mu=mu_xi, sigma=sigma_xi, lower=0, shape=8)
        
        alpha_h, beta_h = xi[0], xi[1]
        alpha_l, beta_l = xi[2], xi[3]
        a0, b0 = xi[4], xi[5]
        a1, b1 = xi[6], xi[7]
    
        S = Ns
        
        # Latent individual-level parameters
        L_h = pm.Beta("L_h", alpha=alpha_h, beta=beta_h, shape=S)
        gamma_h = pm.Deterministic("gamma_h", L_h * scale_gamma_h)
        
        L_l = pm.Beta("L_l", alpha=alpha_l, beta=beta_l, shape=S)
        gamma_l = pm.Deterministic("gamma_l", L_l * scale_gamma_l)
        
        L0 = pm.Gamma("L0", alpha=a0, beta=b0, shape=S)
        beta0 = pm.Deterministic("beta0", L0 * scale_beta0)
        
        L1 = pm.Gamma("L1", alpha=a1, beta=b1, shape=S)
        beta1 = pm.Deterministic("beta1", L1 * scale_beta1)
        
        # Psychometric function transformations
        PSE = pm.Deterministic("PSE", PSE_L_as(gamma_h, gamma_l, beta0, beta1))
        JND = pm.Deterministic("JND", JND_L_as(gamma_h, gamma_l, beta0, beta1))
        
        # Fixed stimulus levels
        x_vals = pt.as_tensor_variable([6, 12, 18, 24, 32, 38, 44, 50])
        
        # Likelihood per session
        for s in range(S):
            p = phi_L_as(gamma_h[s], gamma_l[s], beta0[s], beta1[s], x_vals)
            # Fix the dimensionality issue - p might be returning shape [1, n_stim_levels]
            # but we need it to match C_data[s] which is [n_stim_levels]
            p = p.flatten()  # Flatten to ensure dimensions match
            pm.Binomial(f"c_obs_{s}", n=N_mat[s], p=p, observed=C_data[s])
        
        # Sampling
        trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True)
    return trace
