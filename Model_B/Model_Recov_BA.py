#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:09:04 2026

Fit models A and B to synthetic data A and B, do model comparisons.

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


# Initialize Model A

#%% Load Data and Info A
#os.getcwd() if doesnt work
#load and unpack data
with open("ReadyData_Synth_A.pkl", "rb") as f:
    data_dict_A = pickle.load(f)

cov_mat = data_dict_A['cov_mat']
grp_idx = data_dict_A['grp_idx']
obs_data = data_dict_A['resp']

obs_data_A = obs_data

#%% Load Model A

exec(open("Build_Model_A.py").read())

#  Initialize Model B

#%% Load Data and Info B
#load and unpack data
with open("ReadyData_Synth_B.pkl", "rb") as f:
    data_dict_B = pickle.load(f)

cov_mat = data_dict_B['cov_mat']
grp_idx = data_dict_B['grp_idx']
obs_data = data_dict_B['resp']

sessions = list(data_dict_B['dates_sess_idx'])
sess_idx = data_dict_B['sess_idx']

obs_data_B = obs_data

#%% Load Model B

exec(open("Build_Model_B.py").read())


#%% fit model A with data A, sample from posteriors

with model_A:
    model_A.set_data("obs_data", obs_data_A)
    trace_mA_dA = pm.sample(return_inferencedata=True, chains = 4, progressbar=True, idata_kwargs={"log_likelihood": True})
print("FINISHED SAMPLING!")

#%% fit model B with data A, sample from posteriors

with model_B:
    model_B.set_data("obs_data", obs_data_A)
    trace_mB_dA = pm.sample(return_inferencedata=True, chains = 4, progressbar=True, idata_kwargs={"log_likelihood": True})
print("FINISHED SAMPLING!")

#%% data A comparison

df_comp_A = az.compare({"A_pooled": trace_mA_dA, "B_hierarchical": trace_mB_dA})
az.plot_compare(df_comp_A, insample_dev=False, plot_ic_diff=True, legend=True)


#%%
print(df_comp_A.to_string())

#%% fit model A with data B, sample from posteriors

with model_A:
    model_A.set_data("obs_data", obs_data_B)
    trace_mA_dB = pm.sample(return_inferencedata=True, chains = 4, progressbar=True, idata_kwargs={"log_likelihood": True})
print("FINISHED SAMPLING!")

#%% fit model B with data B, sample from posteriors

with model_B:
    model_B.set_data("obs_data", obs_data_B)
    trace_mB_dB = pm.sample(return_inferencedata=True, chains = 4, progressbar=True, idata_kwargs={"log_likelihood": True})
print("FINISHED SAMPLING!")

#%% data B comparison

df_comp_B = az.compare({"A_pooled": trace_mA_dB, "B_hierarchical": trace_mB_dB})
az.plot_compare(df_comp_B, insample_dev=False, plot_ic_diff=True, legend=True)


#%%
print(df_comp_B.to_string())


