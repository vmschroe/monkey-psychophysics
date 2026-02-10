#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use DataProcessing_A and nonunified as a guide
Created on Tue Feb  3 17:59:51 2026

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
names_grp_idx = {'left_uni':0, 'left_bi':1, 'right_uni':2, 'right_bi':3}

#session date indices

sess_idx = np.array(Raw_DataFrame['sess_idx'])
dates_sess_idx = dict(zip( DataDict['session_summary'].index.tolist() , DataDict['session_summary']['sess_idx'].tolist() ))
sess_sum = DataDict['session_summary']

#binary responses
resp = np.array(Raw_DataFrame['response'])

#construct dictionary
ReadyData_Sirius_B = {'cov_mat':cov_mat,
                    'grp_idx':grp_idx, 
                    'sess_idx': sess_idx, 
                    'resp': resp, 
                    'names_grp_idx': names_grp_idx, 
                    'dates_sess_idx': dates_sess_idx, 
                    'session_summary': sess_sum}

#%% save processed data (choose active directory Model_B\\Data_B)

with open("ReadyData_Sirius_B.pkl","wb") as f:
    pickle.dump(ReadyData_Sirius_B, f)

#%% Generate Synthetic Dataset




