# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:05:04 2025

@author: schro
"""
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.stats import binom
import sys
import FunctionsForGeneralized as ffg
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm
import pytensor.tensor as pt  # Use pytensor instead of aesara
from arviz.stats.density_utils import kde
from High3sLogFuncs import HighLogAnalysis

with open("psych_vecs_all.pkl", "rb") as f:
    data = pickle.load(f)  
    
with open("session_summary.pkl", "rb") as f:
    sess_sum = pickle.load(f)  
    

Cagg_mats = []
Nagg_mats = []
data_dict = {}

for grp in ['ld','ln','rd','rn']:
    data_dict[grp] = {}
    C = np.array([*data['NY'][grp].values()])
    data_dict[grp]['C_mat'] = C
    Cagg_mats.append(sum(C))
    
    N = np.array([*data['N'][grp].values()])
    data_dict[grp]['N_mat'] = N
    Nagg_mats.append(sum(N))

data_dict['agg'] = {}
data_dict['agg']['C_mat'] = np.array(Cagg_mats)
data_dict['agg']['N_mat'] = np.array(Nagg_mats)

trace_agg = HighLogAnalysis(data_dict['agg'])


grps = ['ld','ln','rd','rn']
for i in range():
    az.plot_pair(trace_agg, var_names=['gamma_h', 'gamma_l', 'beta0','beta1'], coords={'gamma_h_dim_0': [i], 'gamma_l_dim_0': [i],'beta0_dim_0': [i],'beta1_dim_0': [i]}, marginals=True, kind="kde")


