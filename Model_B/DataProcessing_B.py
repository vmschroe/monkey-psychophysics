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
