# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 19:19:30 2025

@author: schro
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
    'trials': range(len(obs_data)) }

with pm.Model(coords=coords) as model_A:
    pm.MutableData("cov_mat", cov_mat, dims=("trials", "betas"))
    pm.MutableData("grp_idx", grp_idx, dims=("trials",))

    z_beta = pm.Normal("z_beta", mu = 0, sigma = 1, dims = ("betas", "groups"))
    beta_vec = pm.Deterministic("beta_vec", pm.math.stack([ 6 * z_beta[0] , 2 * z_beta[1] + 4 ], axis=0), dims = ("betas", "groups"))

    z_gam_h = pm.Beta("z_gam_h", mu = 0.2, sigma = 0.15, dims=("groups",) )
    gam_h = pm.Deterministic("gam_h", 0.25*z_gam_h, dims=("groups",) )
    z_gam_l = pm.Beta("z_gam_l", mu = 0.2, sigma = 0.15, dims=("groups",) )
    gam_l = pm.Deterministic("gam_l", 0.25*z_gam_l, dims=("groups",) )
    
    PSE = pm.Deterministic("PSE", (-beta_vec[0] + pm.math.log( (1-2*gam_h) / (1-2*gam_l) ))/ beta_vec[1] , dims=("groups",))
    JND = pm.Deterministic("JND", (pm.math.log( ((3-4*gam_h)*(3-4*gam_l)) / ((1-4*gam_h)*(1-4*gam_l)) )) / (2*beta_vec[1]) , dims=("groups",))
    
    logistic_arg = pm.Deterministic(
        'logistic_arg',
        pm.math.sum(cov_mat * beta_vec[:, grp_idx].T, axis=1),
        dims=("trials",))

    p = pm.Deterministic(
        'p', 
        gam_h[grp_idx] + (1 - gam_h[grp_idx] - gam_l[grp_idx])*pm.math.invlogit(logistic_arg), 
        dims=("trials",))
    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-8,1-1e-8), observed=obs_data, dims=('trials',))
    
print('model is built!')