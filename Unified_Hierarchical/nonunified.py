# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:23:20 2025

@author: schro
"""

#%% packages
import pickle
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
import sys
import aesara.tensor as at
import numpy as np

#%%
with open(r"data_unif_hier.pkl", "rb") as f:
    data_unif_hier = pickle.load(f)  


sides_idx = data_unif_hier['side_index']
obs_data = data_unif_hier['obs_data']
sessions_idx = data_unif_hier['sessions_idx']
sess_distAMP = data_unif_hier['sess_distAMP']
cov_mat = data_unif_hier['cov_mat']
sessions = data_unif_hier['sessions']

x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
amp_mu = np.mean(x_orig)
amp_sig = np.std(x_orig)
x_new = (x_orig-amp_mu)/amp_sig


#%% construct priors


supp_PSE = np.array([min(x_new),max(x_new)])
lik_PSE = np.array([x_new[2],x_new[5]])

supp_JND = np.array([0.04,max(x_new)])
lik_JND = np.array([0.14,0.75])

supp_d = np.unique(cov_mat[:,2])[[1,-1]]
lik_d = np.unique(cov_mat[:,2])[[1,-1]]

[l_p, u_p] = supp_PSE
[l_j, u_j] = supp_JND
[l_d, u_d] = supp_d

b0_supp = 1-np.log(3)*np.array([u_p/l_j, l_p/u_j])
b1_supp = np.log(3)*np.array([1/u_j, 1/l_j])
b2_supp = (1/l_d)*np.array([ 1-b0_supp[1]-b1_supp[1]*u_p , 1-b0_supp[0]-b1_supp[1]*l_p ])

[l_p, u_p] = lik_PSE
[l_j, u_j] = lik_JND
[l_d, u_d] = lik_d

b0_lik = 1-np.log(3)*np.array([u_p/l_j, l_p/u_j])
b1_lik = np.log(3)*np.array([1/u_j, 1/l_j])
b2_lik = (1/l_d)*np.array([ 1-b0_lik[1]-b1_lik[1]*u_p , 1-b0_lik[0]-b1_lik[1]*l_p ])

sup_beta = np.vstack([b0_supp,b1_supp,b2_supp])
lik_beta = np.vstack([b0_lik,b1_lik,b2_lik])

beta_exp_rates = 1/(lik_beta[:,1]-lik_beta[:,0])

#%% pick 1 hand

hand = 0

obs_data1 = obs_data[sides_idx==hand] #shape (6467,)
sessions_idx1 = sessions_idx[sides_idx==hand] #shape (6467,)
cov_mat1 = cov_mat[sides_idx==hand,:] #shape (6467, 3)
biman_idx = cov_mat1[:,2]>0 #shape (6467,)
#%% model

coords = {
    'sessions': sessions, 
    'betas': ["b0", "b1", "b2"],
    'trials': range(len(obs_data1))  # Add trials dimension
}

with pm.Model(coords=coords) as hier_model:
    pm.MutableData("cov_mat1", cov_mat1, dims=("trials", "betas"))
    pm.MutableData("sessions_idx1", sessions_idx, dims=("trials",))
    pm.MutableData("sess_distAMP", sess_distAMP, dims=("sessions",))
    pm.MutableData("biman_idx", biman_idx, dims=("trials",))
    
    z_b_s = pm.Normal("z_b_s", dims=("betas","sessions"))
    mu_b = pm.Uniform("mu_b", lower = sup_beta[:,0], upper = sup_beta[:,1], dims = ("betas",))
    sig_b = pm.Exponential("sig_b", lam = beta_exp_rates, dims = ("betas",))
    # print("mu_b shape (symbolic):", mu_b.shape)
    # print("sig_b shape (symbolic):", sig_b.shape)
    # print("z_b_s shape (symbolic):", z_b_s.shape)
    beta_vec = pm.Deterministic("beta_vec", (mu_b[:, None] + sig_b[:, None] * z_b_s).T, dims = ('sessions', 'betas'))
    
    # gammas

    mu_h_uni = pm.Uniform("mu_h_uni", lower=0.0, upper=1.0)
    z_var_h_uni = pm.Uniform("z_var_h_uni", lower=0.0, upper=1.0)
    sig_h_uni = pm.Deterministic("sig_h_uni", pm.math.sqrt(mu_h_uni*(1-mu_h_uni)*z_var_h_uni))
    z_h_uni_s = pm.Beta("z_h_uni_s", mu = mu_h_uni, sigma = sig_h_uni, dims=("sessions",))
    gam_h_uni_s = pm.Deterministic("gam_h_uni_s", 0.25*z_h_uni_s, dims=("sessions",))
    
    mu_h_bi = pm.Uniform("mu_h_bi", lower=0.0, upper=1.0)
    z_var_h_bi = pm.Uniform("z_var_h_bi", lower=0.0, upper=1.0)
    sig_h_bi = pm.Deterministic("sig_h_bi", pm.math.sqrt(mu_h_bi*(1-mu_h_bi)*z_var_h_bi))
    z_h_bi_s = pm.Beta("z_h_bi_s", mu = mu_h_bi, sigma = sig_h_bi, dims=("sessions",))
    gam_h_bi_s = pm.Deterministic("gam_h_bi_s", 0.25*z_h_bi_s, dims=("sessions",))
    
    mu_l_uni = pm.Uniform("mu_l_uni", lower=0.0, upper=1.0)
    z_var_l_uni = pm.Uniform("z_var_l_uni", lower=0.0, upper=1.0)
    sig_l_uni = pm.Deterministic("sig_l_uni", pm.math.sqrt(mu_l_uni*(1-mu_l_uni)*z_var_l_uni))
    z_l_uni_s = pm.Beta("z_l_uni_s", mu = mu_l_uni, sigma = sig_l_uni, dims=("sessions",))
    gam_l_uni_s = pm.Deterministic("gam_l_uni_s", 0.25*z_l_uni_s, dims=("sessions",))
    
    mu_l_bi = pm.Uniform("mu_l_bi", lower=0.0, upper=1.0)
    z_var_l_bi = pm.Uniform("z_var_l_bi", lower=0.0, upper=1.0)
    sig_l_bi = pm.Deterministic("sig_l_bi", pm.math.sqrt(mu_l_bi*(1-mu_l_bi)*z_var_l_bi))
    z_l_bi_s = pm.Beta("z_l_bi_s", mu = mu_l_bi, sigma = sig_l_bi, dims=("sessions",))
    gam_l_bi_s = pm.Deterministic("gam_l_bi_s", 0.25*z_l_bi_s, dims=("sessions",))
    
    
    
    gam_h = pm.Deterministic(
        "gam_h", biman_idx*gam_h_bi_s[sessions_idx1] + (1-biman_idx)*gam_h_uni_s[sessions_idx1], 
        dims=("trials",))
    gam_l = pm.Deterministic(
        "gam_l", biman_idx*gam_l_bi_s[sessions_idx1] + (1-biman_idx)*gam_l_uni_s[sessions_idx1], 
        dims=("trials",))
    
    # Matrix multiplication
    logistic_arg = pm.Deterministic(
        'logistic_arg',
        pm.math.sum(cov_mat1 * beta_vec[sessions_idx1], axis=1),
        dims=('trials',))
    
    # Probability
    p = pm.Deterministic(
        'p', 
        gam_h + (1 - gam_h - gam_l)*pm.math.invlogit(logistic_arg), 
        dims=('trials',))
    
    # # Likelihood
    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-5,1-1e-5), observed=obs_data1, dims=('trials',))

print("Model created successfully!")
print(f"Total trials across all sessions: {len(obs_data1)}")



