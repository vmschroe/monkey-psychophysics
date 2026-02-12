#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:30:55 2026

@author: vmschroe
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import pickle
import ast
import xarray as xr

#%% LOAD DATA (delete from this script later)
#os.getcwd() if doesnt work
#load and unpack data
with open("ReadyData_Synth_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
#%% 
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']


#%% Construct model

coords = {
    'groups': ['left_uni','left_bi','right_uni','right_bi'],
    'betas': ["b0", "b1"],
    'trials': range(len(obs_data)),
    'sessions':sessions}


with pm.Model(coords=coords) as model_B:
    cov_mat_mut = pm.Data("cov_mat", cov_mat, dims=("trials", "betas"))
    grp_idx_mut = pm.Data("grp_idx", grp_idx, dims=("trials",))
    sess_idx_mut = pm.Data("sess_idx", sess_idx, dims=("trials",))
    
    #HYPERPRIORS
    #Beta vector
    #   Beta_0 ~N[mu_0,sig_0]
    #       mu_0 ~ Unif[-12,12]
    #       sig_0 ~ Exp[lambda = 1/12]
    #   Beta_1 ~N[mu_1,sig_1]
    #       mu_1 ~ Unif[0,8]
    #       sig_1 ~ Exp[lambda = 1/4]
    # REPARAMETRIZATION Exponential: X ~ Exp[lam]
    #   U ~ Unif[0,1]
    #   X = ( -1 / lam ) * ln(U)
    # REPARAMETRIZATION Uniform: X ~ Unif[ a , b ]
    #   U ~ Unif[0,1]
    #   X = a + (b-a) * U
    # REPARAMETRIZATION Normal: X ~ Normal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = sig*Z + mu
    u_mu_betas = pm.Uniform('u_mu_betas', lower = 0, upper=1, dims = ('betas', 'groups'))
    mu_betas = pm.Deterministic('mu_betas', pm.math.stack([ -12 + 24 * u_mu_betas[0] , 8 * u_mu_betas[1] ], axis=0), dims = ("betas", "groups"))
    #u_sig_betas = pm.Uniform('u_sig_betas', lower=0, upper=1, dims = ('betas', 'groups'))
    u_sig_betas = pm.Uniform("u_sig_betas", 1e-9, 1-1e-9, dims=("betas","groups"))
    sig_betas = pm.Deterministic('sig_betas', pm.math.stack([ (-1/(1/12)) * pm.math.log(u_sig_betas[0]) , (-1/(1/4)) * pm.math.log(u_sig_betas[1]) ], axis=0), dims = ("betas", "groups"))
    z_betas = pm.Normal('z_betas', mu = 0, sigma = 1, dims = ("betas", "groups", 'sessions'))
    #beta_vec = pm.Deterministic('beta_vec', sig_betas * z_betas + mu_betas, dims = ("betas", "groups", 'sessions'))
    beta_vec = pm.Deterministic("beta_vec", sig_betas[..., None] * z_betas + mu_betas[..., None], dims=("betas", "groups", "sessions"),)

    
    #Gamma (gamma_h and gamma_l hyperprior iid)
    #   gamma = 0.25 * z_gam
    #   z_gam ~ Beta[ mu = mu_gam , sigma = sig_gam ]
    #       mu_gam ~ Beta[ mu = 0.2, sigma = 0.1 ]
    #       sig_gam^2 ~ Unif[0, mu_gam(1-mu_gam)]
    mu_gams = pm.Beta("mu_gams", mu=0.2, sigma=0.1, dims = ('betas', 'groups'))
    #u_sig2_gams = pm.Uniform('u_sig2_gams', lower=0, upper=1, dims=('betas', 'groups'))
    u_sig2_gams = pm.Uniform("u_sig2_gams", 1e-9, 1-1e-9, dims=("betas","groups"))
    sig_gams = pm.Deterministic('sig_gams', pm.math.sqrt( mu_gams*(1-mu_gams)*u_sig2_gams ), dims=('betas', 'groups'))
    #z_gams = pm.Beta("z_gams", mu=mu_gams, sigma=sig_gams, dims = ('betas', 'groups', 'sessions'))
    z_gams = pm.Beta("z_gams", mu = mu_gams[..., None], sigma=sig_gams[..., None], dims=("betas", "groups", "sessions"),)

    gam_h = pm.Deterministic("gam_h", 0.25*z_gams[0], dims = ('groups', 'sessions'))
    gam_l = pm.Deterministic("gam_l", 0.25*z_gams[1], dims = ('groups', 'sessions'))

    PSE = pm.Deterministic("PSE", (-beta_vec[0] + pm.math.log( (1-2*gam_h) / (1-2*gam_l) ))/ beta_vec[1] , dims=("groups",'sessions'))
    JND = pm.Deterministic("JND", (pm.math.log( ((3-4*gam_h)*(3-4*gam_l)) / ((1-4*gam_h)*(1-4*gam_l)) )) / (2*beta_vec[1]) , dims=("groups",'sessions'))

    # logistic_arg = pm.Deterministic(
    #     'logistic_arg',
    #     pm.math.sum(cov_mat_mut * beta_vec[:, grp_idx_mut, sess_idx_mut].T, axis=1),
    #     dims=("trials",))
    beta_trial = pm.Deterministic('beta_trial', beta_vec[:, grp_idx_mut, sess_idx_mut], dims= ('betas', 'trials') )
    
    logistic_arg = pm.Deterministic( "logistic_arg", pm.math.sum(cov_mat_mut.T * beta_trial, axis=0), dims=("trials",),)

    
    p = pm.Deterministic(
        'p', 
        gam_h[grp_idx_mut, sess_idx_mut] + (1 - gam_h[grp_idx_mut, sess_idx_mut] - gam_l[grp_idx_mut, sess_idx_mut])*pm.math.invlogit(logistic_arg), 
        dims=("trials",))
    
    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-8,1-1e-8), observed=obs_data, dims=('trials',))
    
print('model is built!')

#%% 

model_B.debug()

#%%

with model_B:
    idata = pm.sample(500, tune=500, chains=2, target_accept=0.9)
