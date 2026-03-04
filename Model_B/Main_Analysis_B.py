#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:41:58 2026

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
with open("ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
#%% 
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']

sess_summary = data_dict['session_summary']
#%%%

exec(open("Build_Model_B.py").read())


#%% Sample from posteriors

with model_B:
    trace = pm.sample(return_inferencedata=True, chains = 4, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")


#%% Look at r_hats and effective sample sizes

result_df = az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])

#result_df[ result_df['r_hat']>1 ]
    # r_hat = 1 and ess is large, so sampling was successful
#%% Look at traceplots

az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0", "b1"], 
    'sessions':[sessions[5],sessions[10], '05-30']}, compact=False,  backend_kwargs={"constrained_layout": True})


#%% plot joint posteriors

sess_choice = '05-30'

for grp_num, grp_choice in enumerate(coords['groups']):
     az.plot_pair(trace, var_names=['gam_h', 'gam_l'
                                    ,'beta_vec'
                                    #,'PSE', 'JND'
                                    ], 
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice], 'sessions': [sess_choice]}, 
             kind = 'kde', marginals=True)
     

#%%

az.plot_trace(trace, var_names=('gam_h', 'mu_gams', 'sig_gams'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0"], 
    'sessions':['05-30'],}, compact=False,  backend_kwargs={"constrained_layout": True})


#%% weird joint posterior

 az.plot_pair(trace, var_names=['gam_h', 'mu_gams', 'sig_gams'], 
         coords = {
             'groups': ['left_bi'],
             'betas': ["b0"], 
             'sessions':['05-30'],},  kind = 'kde', marginals=True)
 