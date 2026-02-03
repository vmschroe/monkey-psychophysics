#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:17:05 2026

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
from scipy.stats import binom

#%%

#load raw data
with open("Sirius_Data.pkl", "rb") as f:
    DataDict = pickle.load(f)  

# if doesn't load, os.getcwd()
#%% 
#########%%
## Format Experimental Data


Raw_DataFrame = DataDict['data']
stims = np.array(Raw_DataFrame['stim_amp'])

#normalize stim amps
x_old = np.unique(stims)
x_mu = np.mean(x_old)
x_sig = np.std(x_old)
stims_normed = ((stims-x_mu)/x_sig)

#construct covarate matrix
cov_mat = np.vstack([np.full_like(stims_normed, 1.0), stims_normed]).T

#hand/distraction group indices
grp_idx = np.array(Raw_DataFrame['group_idx'])

#binary responses
resp = np.array(Raw_DataFrame['response'])

#construct dictionary
ReadyData_Sirius_A = {'cov_mat':cov_mat,'grp_idx':grp_idx,'resp': resp}

ReadyData_Sirius_A['names_grp_idx'] = {'left_uni':0, 'left_bi':1, 'right_uni':2, 'right_bi':3}


#save processed data
with open("Model_A\\Data_A\\ReadyData_Sirius_A.pkl","wb") as f:
    pickle.dump(ReadyData_Sirius_A, f)
    
#check if saved correctly
#with open("Model_A\\Data_A\\ReadyData_Sirius_A.pkl","rb") as f:
#    testing  = pickle.load(f)

#%% Generate Synthetic Dataset

#fix parameters for each group
gam_h_fix =  np.array([0.03, 0.06, 0.06, 0.03])
gam_l_fix =  np.array([0.005, 0.01, 0.005, 0.01])
beta0_fix = np.array([-1, 1, -1, 1])
beta1_fix = np.array([3, 3, 6, 6])
params_fixed = [gam_h_fix, gam_l_fix, beta0_fix, beta1_fix]

#generates synthetic data from same covariates as experimental
def synth_generator_A(params_fixed, cov_mat, grp_idx):
    [gam_h_fix, gam_l_fix, beta0_fix, beta1_fix] = params_fixed
    beta_mat = np.vstack([beta0_fix, beta1_fix])
    log_arg = (cov_mat @ beta_mat)
    psis_mat = gam_h_fix + (1-gam_h_fix-gam_l_fix)/( 1 + np.exp( - log_arg ))
    psis = psis_mat[np.arange(len(grp_idx)), grp_idx]
    synth_resp = binom.rvs(n=1, p=psis)
    return synth_resp

synth_resp = synth_generator_A(params_fixed, cov_mat, grp_idx)

#construct dictionary
ReadyData_Synth_A = {'cov_mat': cov_mat,
                     'grp_idx': grp_idx,
                     'resp': synth_resp, 
                     'params_fixed': params_fixed,
                     'names_grp_idx': ReadyData_Sirius_A['names_grp_idx']}

#save processed data
with open("Model_A\\Data_A\\ReadyData_Synth_A.pkl","wb") as f:
    pickle.dump(ReadyData_Synth_A, f)
#%%
#check if synth data makes sense



freq_df = pd.DataFrame({'stim': cov_mat[:,1], 'grp_idx': grp_idx, 'responses': synth_resp})
freqs = pd.pivot_table(
    freq_df, 
    values='responses',
    index='stim',
    columns='grp_idx',
    aggfunc='mean'
)

plt.scatter(freqs.index, freqs[3])










