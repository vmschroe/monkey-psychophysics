#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 19:04:20 2026

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

#%%
#os.getcwd() if doesnt work
#load and unpack data
with open("ReadyData_Sirius_A.pkl", "rb") as f:
    data_dict = pickle.load(f)

cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']

#%%

exec(open("Build_Model_A.py").read())

#%% Sample from posteriors

with model_A:
    trace = pm.sample(return_inferencedata=True, chains = 4, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")


#%% Look at r_hats and effective sample sizes

az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])


    # r_hat = 1 and ess is large, so sampling was successful
#%% Look at traceplots

az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_uni'],
    'betas': ["b0", "b1"]}, compact=False,  backend_kwargs={"constrained_layout": True})
#%% plot joint posteriors


for grp_num, grp_choice in enumerate(coords['groups']):
     az.plot_pair(trace, var_names=['gam_h', 'gam_l', 'beta_vec', 'PSE', 'JND'], 
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice]}, 
             kind = 'kde', marginals=True)
     
#%% 
     

param_samps = trace.posterior[['beta_vec', 'gam_h', 'gam_l']]

gam_h_samps = {}
gam_l_samps = {}
beta_0_samps = {}
beta_1_samps = {}


for grp in ["left_bi","left_uni","right_bi","right_uni"]:
    gam_h_samps[grp] = param_samps['gam_h'].sel(groups = grp).values.flatten()
    gam_l_samps[grp] = param_samps['gam_l'].sel(groups = grp).values.flatten()
    beta_0_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b0').values.flatten()
    beta_1_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b1').values.flatten()

def psychfunc(params, X):
    """
    Psychometric function with lapses

    Parameters:
    params : [gamma, lambda_, beta0, beta1]
    X : Stimulus amplitude level

    Returns:
    probability of guess "high"

    """
    X = np.asarray(X)
    gam_h, gam_l, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gam_h + (1 - gam_h - gam_l) * logistic

#frquencies for each level
freq_df = pd.DataFrame({'stim': cov_mat[:,1], 'grp_idx': grp_idx, 'obs_data': obs_data})
freqs = pd.pivot_table(
    freq_df, 
    values='obs_data',
    index='stim',
    columns='grp_idx',
    aggfunc='mean'
)

xfit = np.linspace(-1.6,1.6,500)
y_samples = {}
hdis = {}
rec_params = {}
yrec = {}
fixed = np.array(params_fixed)

for grp_i, grp in enumerate(['left_uni','left_bi','right_uni','right_bi']):
    y_samples[grp] = np.array([psychfunc([gam_h,gam_l,beta_0,beta_1], xfit) 
                          for gam_h,gam_l,beta_0,beta_1 in zip(
                              gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp])])
    hdis[grp] = az.hdi(y_samples[grp], hdi_prob=0.95)
    rec_params[grp] = np.mean(np.array([gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp]]), axis = 1)
    
    
    yrec[grp] = psychfunc(rec_params[grp], xfit)
    
#%% 
x_old = [6,12,18,24,32,38,44,50]
x_mu = np.mean(x_old)
x_sig = np.std(x_old)

#%%
    
plt.plot(xfit*x_sig-x_mu,yrec['left_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig-x_mu, hdis['left_uni'][:, 0], hdis['left_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig-x_mu,np.array(freqs[0]),label='Data', color = 'blue')
plt.plot(xfit*x_sig-x_mu,yrec['left_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig-x_mu, hdis['left_bi'][:, 0], hdis['left_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig-x_mu,np.array(freqs[1]),label='Data', color = 'red')

plt.xlabel('Stimulus Amplitude')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Left Hand Psychometric Curves")
plt.show()     
     
     
#%%


plt.plot(xfit*x_sig-x_mu,yrec['right_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig-x_mu, hdis['right_uni'][:, 0], hdis['right_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig-x_mu,np.array(freqs[2]),label='Data', color = 'blue')
plt.plot(xfit*x_sig-x_mu,yrec['right_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig-x_mu, hdis['right_bi'][:, 0], hdis['right_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig-x_mu,np.array(freqs[3]),label='Data', color = 'red')

plt.xlabel('Stimulus Amplitude')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Right Hand Psychometric Curves")
plt.show()     
     
     
#%%
     
     
     
     
     
     
     
     
     
     
     
     
     