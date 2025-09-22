# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 15:25:55 2025

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
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm

#%%
x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
x_new = (x_orig - np.mean(x_orig) ) / np.std(x_orig)


def compute_beta_params(mu, sig):
    v = ( mu*(1-mu) )/(sig**2) - 1
    a = mu*v
    b = (1-mu)*v
    return a, b


def compute_beta_priors(supp,lik,name = None):
    """
    ( PARAM - 'shift' ) / 'scale'   ~   Beta( 'alpha' , 'beta' )

    """
    [L,U] = supp
    lik_z = (lik-L)/(U-L)
    mean_z = np.mean(lik_z)
    std_z = lik_z[1]-mean_z
    a,b = compute_beta_params(mean_z, std_z)
    if name:
        print(f"({name} - {round(L,3)}) / {round(U-L,3)}   ~   Beta( {round(a,2)} , {round(b,2)} )")
    param_prior = {}
    param_prior['shift'] = L
    param_prior['scale'] = U-L
    param_prior['alpha'] = a
    param_prior['beta'] = b
    return param_prior

#%%


supp_PSE = np.array([min(x_new),max(x_new)])
lik_PSE = np.array([x_new[2],x_new[5]])

PSE_prior = compute_beta_priors(supp_PSE,lik_PSE,'PSE')

#%%

supp_JND = np.array([0,max(x_new)])
lik_JND = np.array([0.1,1.0])
JND_prior = compute_beta_priors(supp_JND,lik_JND,'JND')

#%%

supp_gamma = np.array([0,0.25])
lik_gamma = np.array([0,0.05])
gamma_prior = compute_beta_priors(supp_gamma,lik_gamma,'gamma')
