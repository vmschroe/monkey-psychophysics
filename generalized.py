#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:02:39 2025

@author: vmschroe
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
import pandas as pd
import bayesfit as bf
import statsmodels.api as sm
import os
import math
from scipy.optimize import minimize
from scipy.stats import binom
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
from scipy.special import gammaln
import pymc as pm
import arviz as az
import sys
import FunctionsForGeneralized
import seaborn as sns
import pickle
import numpy as np

x = [6, 12, 18, 24, 32, 38, 44, 50]
### Load in data: 
    
with open("psych_vecs_all.pkl", "rb") as f:
    psych_vecs_all = pickle.load(f)

exec(open('FunctionsForGeneralized.py').read())

grps = ['ld', 'ln', 'rd', 'rn']
sessions = list(psych_vecs_all['NY']['ld'].keys())

#combine sessions
NY = {}
N = {}
Y = {}
psych_vector = {}
labels = {
    'ld': 'Left hand, Distracted',
    'ln': 'Left hand, Not distracted',
    'rd': 'Right hand, Distracted',
    'rn': 'Right hand, Not distracted'
    }

for grp in grps:
    NY[grp] = sum(psych_vecs_all['NY'][grp].values())
    N[grp] = sum(psych_vecs_all['N'][grp].values())
    Y[grp] = NY[grp] / N[grp]
    psych_vector[grp] = [ N[grp], NY[grp]]


# Choose selected model

selected_model = "weibull"

if selected_model not in model_details:
        raise ValueError(f"Unknown model '{selected_model}'. Choose from {list(model_details.keys())}.")
phi_func = model_details[selected_model]["phi"]
phi_inv_func = model_details[selected_model]["phi_inv"]
pse_func = model_details[selected_model]["PSE"]
jnd_func = model_details[selected_model]["JND"]
params_prior_params = model_details[selected_model]["param_priors"]["params"]
params_prior_scale = model_details[selected_model]["param_priors"]["scales"]
    

traces_all = {'ld': None, 'ln': None, 'rd': None, 'rn': None}

for i, grp in enumerate(grps):
    traces_all[grp] = data_analysis(psych_vector[grp], grp_name = grp)
    plot_post(traces_all[grp], grp_name = labels[grp])

for grp_pair in [['ld','ln'],['rd','rn'],['ln','rn'],['ld', 'rd']]:
    [grp1,grp2] = grp_pair
    plot_rec_curves([traces_all[grp1], traces_all[grp2]], [Y[grp1],Y[grp2]], [labels[grp1],labels[grp2]])
    plot_attr_dist_with_hdi([traces_all[grp1], traces_all[grp2]], [labels[grp1],labels[grp2]])