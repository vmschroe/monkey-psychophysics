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


coords = {
    'groups': ['left_uni','left_bi','right_uni','right_bi'],
    'betas': ["b0", "b1"],
    'trials': range(len(obs_data)),
    'sessions':sessions}


with pm.Model(coords=coords) as model_B:
    cov_mat_mut = pm.Data("cov_mat", cov_mat, dims=("trials", "betas"))
    grp_idx_mut = pm.Data("grp_idx", grp_idx, dims=("trials",))
    sess_idx_mut = pm.Data("sess_idx", sess_idx, dims=("trials",))
    obs_data_mut = pm.Data("obs_data", obs_data, dims=("trials",))
    
    #HYPERPRIORS
    #Beta vector
    #   Beta_0 ~N[mu_0,sig_0]
    #       mu_0 ~ N[ mu = 0 , sigma = 2 ]   (x = 0 is amplitude 28, the category boundary)
    #       sig_0 ~ HalfNormal[sig=2]
    #   Beta_1 ~N[mu_1,sig_1]
    #       mu_1 ~ LogNormal[ mu = log(4) , sigma = 0.5 ]   (median 4, 95% in [1.5, 10.7])
    #       sig_1 ~ Exp[lambda = 1]
    # REPARAMETRIZATION Exponential: X ~ Exp[lam]
    #   U ~ Unif[0,1]
    #   X = ( -1 / lam ) * ln(U)
    # REPARAMETRIZATION Normal: X ~ Normal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = sig*Z + mu
    # REPARAMETRIZATION LogNormal: X ~ LogNormal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = exp( sig*Z + mu )
    # REPARAMETRIZATION HalfNormal: X ~ HalfNormal[ sig ]
    #   Z ~ Normal[0,1]
    #   X = abs(  sig * Z  )
    z_mu_b0 = pm.Normal('z_mu_b0', mu = 0, sigma = 1, dims = ('groups',))
    z_mu_b1 = pm.Normal('z_mu_b1', mu = 0, sigma = 1, dims = ('groups',))
    mu_betas = pm.Deterministic('mu_betas', pm.math.stack([ 2 * z_mu_b0 , pm.math.exp(np.log(4) + 0.5 * z_mu_b1) ], axis=0), dims = ("betas", "groups"))
    z_sig_b0 = pm.Normal('z_sig_b0', mu = 0, sigma = 1, dims = ('groups',))
    u_sig_b1 = pm.Uniform("u_sig_b1", 1e-9, 1-1e-9, dims=('groups',))
    sig_betas = pm.Deterministic('sig_betas', pm.math.stack([ pm.math.abs(2 * z_sig_b0) , (-1/1) * pm.math.log(u_sig_b1) ], axis=0), dims = ("betas", "groups"))
    z_betas = pm.Normal('z_betas', mu = 0, sigma = 1, dims = ("betas", "groups", 'sessions'))
    #beta_vec = pm.Deterministic('beta_vec', sig_betas * z_betas + mu_betas, dims = ("betas", "groups", 'sessions'))
    beta_vec = pm.Deterministic("beta_vec", sig_betas[..., None] * z_betas + mu_betas[..., None], dims=("betas", "groups", "sessions"),)

    
    #Gamma (gamma_h and gamma_l hyperprior iid)
    #   gamma = 0.25 * w_gam
    #   w_gam ~ LogitNormal[ mu = mu_gam , sigma = sig_gam ]
    #       mu_gam ~ N[ mu = -3, sigma = 1.5 ]   (typical gamma: median 0.012, 95% in [0.0007, 0.12])
    #       sig_gam ~ HalfNormal[sig=0.5]   (median 0.34: at a typical gamma of 0.05, sessions mostly within 0.03-0.07)
    # REPARAMETRIZATION LogitNormal: X ~ LogitNormal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = 1 / ( 1+ exp (- (sig*Z + mu) ) )
    # REPARAMETRIZATION HalfNormal: X ~ HalfNormal[ sig ]
    #   Z ~ Normal[0,1]
    #   X = abs(  sig * Z  )
    z_mu_gams = pm.Normal("z_mu_gams", mu=0, sigma=1, dims = ('betas', 'groups'))
    mu_gams = pm.Deterministic("mu_gams", 1.5*z_mu_gams - 3, dims = ('betas', 'groups'))
    z_sig_gams = pm.Normal("z_sig_gams", mu=0, sigma=1, dims = ('betas', 'groups'))
    sig_gams = pm.Deterministic('sig_gams', pm.math.abs(0.5*z_sig_gams), dims=('betas', 'groups'))
    z_gams = pm.Normal("z_gams", mu=0, sigma=1, dims=("betas", "groups", "sessions"))
    gams = pm.Deterministic("gams", 0.25 * pm.math.invlogit(sig_gams[..., None] * z_gams + mu_gams[..., None]), dims=("betas", "groups", "sessions"),)
    
    gam_h = pm.Deterministic("gam_h", gams[0], dims = ('groups', 'sessions'))
    gam_l = pm.Deterministic("gam_l", gams[1], dims = ('groups', 'sessions'))

    PSE = pm.Deterministic("PSE", (-beta_vec[0] + pm.math.log( (1-2*gam_h) / (1-2*gam_l) ))/ beta_vec[1] , dims=("groups",'sessions'))
    JND = pm.Deterministic("JND", (pm.math.log( ((3-4*gam_h)*(3-4*gam_l)) / ((1-4*gam_h)*(1-4*gam_l)) )) / (2*beta_vec[1]) , dims=("groups",'sessions'))

    
    # Per-trial quantities are plain tensors, not pm.Deterministic: storing one value
    # per trial for every draw would add ~2 GB per variable to the trace
    beta_trial = beta_vec[:, grp_idx_mut, sess_idx_mut]

    logistic_arg = pm.math.sum(cov_mat_mut.T * beta_trial, axis=0)


    p = gam_h[grp_idx_mut, sess_idx_mut] + (1 - gam_h[grp_idx_mut, sess_idx_mut] - gam_l[grp_idx_mut, sess_idx_mut])*pm.math.invlogit(logistic_arg)
    
    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-8,1-1e-8), observed=obs_data_mut, dims=('trials',))
    
print('model is built!')
