# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:14:19 2025

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

def organizer_A(data, sess_sum):
    sessions = list(sess_sum.index)
    newdf = pd.DataFrame()
    for sess_idx, sess in enumerate(sessions):
        for grp_idx, grp in enumerate(['ld', 'ln', 'rd', 'rn']):
            dftemp = data[grp][sess][['stimAMP','lowORhighGUESS']].copy()
            dftemp['session'] = sess_idx
            dftemp['side'] = 0
            dftemp['distracted'] = 0
            if grp in ['rd', 'rn']:
                dftemp['side'] = 1
            if grp in ['ld', 'rd']:
                dftemp['distracted'] = 1
            dftemp['grp_idx'] = grp_idx
            newdf = pd.concat([newdf, dftemp], axis=0)
    newdf.columns = ['stim', 'response', 'sess_idx','side_idx','dist_idx', 'grp_idx']
    x_old = np.array([6, 12, 18, 24, 32, 38, 44, 50])
    newdf['stim'] = (newdf['stim'] - np.mean(x_old))/np.std(x_old)
    return newdf

def preparer_A(newdf):
    stim_vals = np.array(newdf['stim']).reshape(-1,1)
    ones = np.full_like(stim_vals, 1).reshape(-1,1)
    cov_mat = np.hstack([ones, stim_vals])
    grp_idx = np.array(newdf['grp_idx'])
    obs_data = np.array(newdf['response'])
    
    ReadyData = {'cov_mat': cov_mat, 'grp_idx': grp_idx, 'obs_data': obs_data}
    return ReadyData

def synth_generator_A(true_data, beta0_fix, beta1_fix, gam_h_fix, gam_l_fix):
    data = true_data.copy()
    data['response'] = 0
    data['psi'] = gam_h_fix[ data['grp_idx'] ] + (1- gam_h_fix[ data['grp_idx'] ] - gam_l_fix[ data['grp_idx'] ])/ (1 + np.exp(-(beta0_fix[ data['grp_idx'] ] + beta1_fix[ data['grp_idx'] ] * data['stim'])))
    data['response'] = binom.rvs(n=1, p=np.array(data['psi']))
    return data
    