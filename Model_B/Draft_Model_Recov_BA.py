#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:10:14 2026

@author: vmschroe
"""
#fit synth data A to models A and B
#fit synth data B to models A and B
#do model comparison

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

##fit synth data A to models A and B
#%% load synth data A
#os.getcwd() if doesnt work
#load and unpack data
with open("ReadyData_Synth_A.pkl", "rb") as f:
    data_dict = pickle.load(f)

cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
params_fixed = data_dict['params_fixed']

data_dict_A = data_dict
obs_data_A = obs_data

#%% initialize model A with data A

exec(open("Build_Model_A.py").read())

#%% fit model A with data A, sample from posteriors

with model_A:
    trace_mA_dA = pm.sample(return_inferencedata=True, chains = 4, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")

#%%

with open("ReadyData_Synth_B.pkl", "rb") as f:
    data_dict_B = pickle.load(f)
#%% 
sessions = list(data_dict_B['dates_sess_idx'])
cov_mat = data_dict_B['cov_mat']
grp_idx = data_dict_B['grp_idx']
sess_idx = data_dict_B['sess_idx']


#%%

exec(open("Build_Model_B.py").read())


#%% Sample from posteriors

with model_B:
    trace_mB_dA = pm.sample(return_inferencedata=True, chains = 4, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")

#%% data A comparison

df_comp_A = az.compare({"A_pooled": trace_mA_dA, "B_hierarchical": trace_mB_dA})

df_comp_A

az.plot_compare(df_comp_A, insample_dev=False)

#%%

obs_data_B = data_dict_B['resp']

with model_A:
    model_A.set_data("obs_data", obs_data_B)
    trace_mA_dB = pm.sample(return_inferencedata=True, chains = 4, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})