#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:34:04 2026

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
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import uniform


#%% USING LOGIT_NORMAL for gamma params
nsamps = 10000

z_all = norm.rvs(size=nsamps*3).reshape((-1,3))
z_s = z_all[:,0]
z_mu = z_all[:,1]
z_sig = z_all[:,2]

mu_gam = 1*z_mu+(-2.5)
sig_gam = np.abs(1.25*z_sig)
gam_samps = 0.25*(1 / (1 + np.exp(-(sig_gam*z_s+mu_gam))))


plt.hist(gam_samps, density=True, bins=20) 


#%%



#%% Defunct below


#%% DONT USE BETA ANYMORE! screws w hmc

def get_beta_params(mu_gh,sig_gh):
    nu_gh  = (mu_gh * (1.0 - mu_gh) / (sig_gh**2)) - 1.0  # <-- fixed *
    a_gh   = mu_gh * nu_gh
    b_gh   = (1.0 - mu_gh) * nu_gh
    return [a_gh,b_gh]

#%%

mu_tru = 0.2
sig_tru = 0.15
ab = get_beta_params(mu_tru,sig_tru)
samps_true_beta = beta.rvs(a=ab[0], b=ab[1], size=1000)

mus = beta.rvs(a=ab[0], b=ab[1], size=1000)
uvar = uniform.rvs(loc=0, scale=1, size=1000)
sigs = np.sqrt(mus*(1-mus)*uvar)
abweird = get_beta_params(mus,sigs)
samps_w = beta.rvs(a=abweird[0][:], b=abweird[1][:], size=1000)


#%% 

plt.hist(samps_true_beta, density=True)
plt.hist(samps_w, density=True) 
