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
    pm.Data("cov_mat", cov_mat, dims=("trials", "betas"))
    pm.Data("grp_idx", grp_idx, dims=("trials",))
    pm.Data("sess_idx", sess_idx, dims=("trials",))
    
    #HYPERPRIORS
    #Beta vector
    #   Beta_0 ~N[mu_0,sig_0]
    #       mu_0 ~ Unif[-12,12]
    #       sig_0 ~ Exp[lambda = 1/12]
    #   Beta_1 ~N[mu_1,sig_1]
    #       mu_1 ~ Unif[0,8]
    #       sig_1 ~ Exp[lambda = 1/4]
    #Gamma (gamma_h and gamma_l hyperprior iid)
    #   gamma = 0.25 * z_gam
    #   z_gam ~ Beta[ mu = mu_gam , sigma = sig_gam ]
    #       mu_gam ~ Beta[ mu = 0.2, sigma = 0.1 ]
    #       sig_gam^2 ~ Unif[0, mu_gam(1-mu_gam)]






