#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:40:06 2026

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
with open("ReadyData_Synth_A.pkl", "rb") as f:
    data_dict = pickle.load(f)

cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
params_fixed = data_dict['params_fixed']

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
    
#%% Look at joint posteriors for parameters. Some identifiability issues, but ok

grp_num = 0
grp_choice = coords['groups'][grp_num]

ref_vals = {}

for grp_num, grp_choice in enumerate(coords['groups']):
    temp_dict = {}
    
    ref_vals[grp_choice] = {
        "gam_h": xr.DataArray([params_fixed[0][grp_num]], dims=("groups",), coords={"groups": [grp_choice]}),
        "gam_l": xr.DataArray([params_fixed[1][grp_num]], dims=("groups",), coords={"groups": [grp_choice]}),
        "beta_vec": xr.DataArray(
            np.array([[params_fixed[2][grp_num]], [params_fixed[3][grp_num]]]),  # shape (2, 1)
            dims=("betas", "groups"),
            coords={"betas": ["b0", "b1"], "groups": [grp_choice]},
        ),
        }
    


for grp_num, grp_choice in enumerate(coords['groups']):
     az.plot_pair(trace, var_names=['gam_h', 'gam_l', 'beta_vec'], 
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice]}, 
             reference_values=ref_vals[grp_choice],
             reference_values_kwargs=dict(marker="o", color="red", markersize=8),
             kind = 'kde', marginals=True)



#%% Look at joint posteriors for PSE and JND

ref_vals2 = {}

for grp_num, grp_choice in enumerate(coords['groups']):
    temp_dict = {}
    ref_vals2[grp_choice] = {
        "gam_h": xr.DataArray([params_fixed[0][grp_num]], dims=("groups",), coords={"groups": [grp_choice]}),
        "gam_l": xr.DataArray([params_fixed[1][grp_num]], dims=("groups",), coords={"groups": [grp_choice]})
        }
    PSE = (-params_fixed[2][grp_num] + np.log( (1-2*params_fixed[0][grp_num]) / (1-2*params_fixed[1][grp_num]) ))/ params_fixed[3][grp_num]
    JND = np.log( ((3-4*params_fixed[0][grp_num])*(3-4*params_fixed[1][grp_num])) / 
                 ((1-4*params_fixed[0][grp_num])*(1-4*params_fixed[1][grp_num])) ) / (2*params_fixed[3][grp_num])
    ref_vals2[grp_choice]['PSE'] = xr.DataArray([PSE], dims=("groups",), coords={"groups": [grp_choice]})
    ref_vals2[grp_choice]['JND'] = xr.DataArray([JND], dims=("groups",), coords={"groups": [grp_choice]})

for grp_num, grp_choice in enumerate(coords['groups']):
    az.plot_pair(trace, var_names=['gam_h', 'gam_l', 'PSE', 'JND'], 
             coords = {'groups': [grp_choice]},  
             reference_values = ref_vals2[grp_choice],
                       reference_values_kwargs=dict(marker="o", color="red", markersize=8),
             kind = 'kde', marginals=True)



#%% Plot original vs recovered curves

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
    
    plt.plot(xfit,yrec[grp],label='Recovered Curve',color='green')
    plt.plot(xfit, psychfunc(fixed[:,grp_i], xfit), label='Original Curve',color='red')
    plt.fill_between(xfit, hdis[grp][:, 0], hdis[grp][:, 1], color='green', alpha=0.3, label='95% HDI')
    plt.scatter(np.array(freqs.index),np.array(freqs[grp_i]),label='Data', color = 'red')
    plt.title(grp)
    plt.xlabel('Stimulus Amplitude')
    plt.legend(loc='upper left', fontsize=9.5)
    plt.show()   

#%% Posterior predictive

with model_A:
    pm.sample_posterior_predictive(trace,extend_inferencedata=True)

az.plot_ppc(trace, num_pp_samples=100)

#%% cross validation

az.loo(trace)

#%% compare prior and posterior

with model_A:
    prior = pm.sample_prior_predictive(samples=3000)
trace.extend(prior)


#%%
true_gam_h = {
    "left_uni": 0.03,
    "left_bi": 0.06,
    "right_uni": 0.06,
    "right_bi": 0.03,
}
groups = trace.posterior.coords["groups"].values

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()

for ax, g in zip(axes, groups):
    prior_vals = trace.prior["gam_h"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["gam_h"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]
    post_vals  = post_vals[np.isfinite(post_vals)]

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    x_true = true_gam_h[g]
    ax.axvline(x_true, linestyle="--", linewidth=2, label="truth", color='green')

    ax.set_title(g)

    ax.legend()
fig.suptitle("Parameter Recovery: gamma_h", fontsize=16)
plt.show()


#%% 

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()

for ax, g in zip(axes, groups):
    prior_vals = trace.prior["PSE"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["PSE"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]
    post_vals  = post_vals[np.isfinite(post_vals)]

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    #x_true = true_gam_h[g]
    #ax.axvline(x_true, linestyle="--", linewidth=2, label="truth", color='green')
    ax.set_xlim(-0.5, 0.5)
    ax.set_title(g)

    ax.legend()
fig.suptitle("Parameter Recovery: PSE", fontsize=16)
plt.show()

