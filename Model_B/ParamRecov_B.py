#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:34:27 2026

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

#%% LOAD DATA (delete from this script later)
#os.getcwd() if doesnt work
#load and unpack data
with open("ReadyData_Synth_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
#%% 
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']


#%%%

exec(open("Build_Model_B.py").read())


#%% Sample from posteriors

with model_B:
    trace = pm.sample(return_inferencedata=True, chains = 4, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")


#%% Look at r_hats and effective sample sizes

az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])


    # r_hat ~= 1 and ess is large, so sampling was successful
    
#%% Look at traceplots

az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_uni'],
    'betas': ["b0", "b1"], 'sessions':['04-11', '07-11']}, compact=False,  backend_kwargs={"constrained_layout": True})


#%% helper functions


GROUP_TO_HAND_MANUAL = {
    "left_uni":  ("left",  "uni"),
    "left_bi":   ("left",  "bi"),
    "right_uni": ("right", "uni"),
    "right_bi":  ("right", "bi"),
}

def _singleton(v):
    # az/coords often uses lists like ["left_uni"]; .sel() wants a label
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        return v[0]
    return v

def build_reference_values_for_pairplot(data_dict, trace, group, session_label, sessions_list):
    """
    Returns reference_values with correct xarray shapes for:
      var_names=("gam_h","gam_l","beta_vec")
      coords={"groups":[group], "sessions":[session_label], "betas":["b0","b1"]}
    """

    # --- map to synth_session_params indexing ---
    hand, manual = GROUP_TO_HAND_MANUAL[group]
    sess_idx = sessions_list.index(session_label)

    sp = data_dict["synth_session_params"].sel(session=sess_idx, hand=hand, manual=manual)

    ref_gam_h = float(sp.sel(par_type="gamma", par_idx="h0"))
    ref_gam_l = float(sp.sel(par_type="gamma", par_idx="l1"))
    ref_b0    = float(sp.sel(par_type="beta",  par_idx="h0"))
    ref_b1    = float(sp.sel(par_type="beta",  par_idx="l1"))

    # --- build 1x1 DataArrays for gam_h/gam_l with dims (groups, sessions) ---
    gam_coords = {"groups": [group], "sessions": [session_label]}
    ref_gam_h_da = xr.DataArray([[ref_gam_h]], coords=gam_coords, dims=("groups", "sessions"))
    ref_gam_l_da = xr.DataArray([[ref_gam_l]], coords=gam_coords, dims=("groups", "sessions"))

    # --- build 2x1x1 DataArray for beta_vec with dims (betas, groups, sessions) ---
    beta_coords = {"betas": ["b0", "b1"], "groups": [group], "sessions": [session_label]}
    ref_beta_da = xr.DataArray([[[ref_b0]], [[ref_b1]]],
                               coords=beta_coords, dims=("betas", "groups", "sessions"))

    return {
        "gam_h": ref_gam_h_da,
        "gam_l": ref_gam_l_da,
        "beta_vec": ref_beta_da,
    }

#%% Look at joint posteriors for parameters
grp = "left_bi"
sess_label = sessions[7]

ref_vals = build_reference_values_for_pairplot(data_dict, trace, grp, sess_label, sessions)

az.plot_pair(
    trace,
    var_names=("gam_h", "gam_l", "beta_vec"),
    coords={"betas": ["b0", "b1"], "groups": [grp], "sessions": [sess_label]},
    kind="kde",
    marginals=True,
    reference_values=ref_vals,
    reference_values_kwargs={"color": "crimson", "lw": 2, "alpha": 0.9},
)

#%% look at joint posteriors for PSE and JND
grp = "left_bi"
sess_label = sessions[7]

az.plot_pair(
    trace,
    var_names=("gam_h", "gam_l", "PSE", "JND"),
    coords={"groups": [grp], "sessions": [sess_label]},
    kind="kde",
    marginals=True,
)

#%% Plot original vs recovered curves
sess = 11
sess_label = sessions[sess]

param_samps = trace.posterior[['beta_vec', 'gam_h', 'gam_l']]

gam_h_samps = {}
gam_l_samps = {}
beta_0_samps = {}
beta_1_samps = {}


for grp in ["left_bi","left_uni","right_bi","right_uni"]:
    gam_h_samps[grp] = param_samps['gam_h'].sel(groups = grp, sessions=sess_label).values.flatten()
    gam_l_samps[grp] = param_samps['gam_l'].sel(groups = grp, sessions=sess_label).values.flatten()
    beta_0_samps[grp] = param_samps['beta_vec'].sel(groups = grp, sessions=sess_label, betas='b0').values.flatten()
    beta_1_samps[grp] = param_samps['beta_vec'].sel(groups = grp, sessions=sess_label, betas='b1').values.flatten()

fixed = {}
for hh in ['left', 'right']:
    for mm in ['uni','bi']:
        grp_name = hh + '_' + mm
        fixed[grp_name] = data_dict['synth_session_params'].loc[dict(hand=hh, manual=mm, session=sess)].values.flatten()
        
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
freq_df = pd.DataFrame({'stim': cov_mat[sess_idx==sess][:,1], 'grp_idx': grp_idx[sess_idx==sess], 'obs_data': obs_data[sess_idx==sess]})
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
#fixed = np.array(params_fixed)

for grp_i, grp in enumerate(['left_uni','left_bi','right_uni','right_bi']):
    y_samples[grp] = np.array([psychfunc([gam_h,gam_l,beta_0,beta_1], xfit) 
                          for gam_h,gam_l,beta_0,beta_1 in zip(
                              gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp])])
    hdis[grp] = az.hdi(y_samples[grp], hdi_prob=0.95)
    rec_params[grp] = np.mean(np.array([gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp]]), axis = 1)
    
    
    yrec[grp] = psychfunc(rec_params[grp], xfit)
    
    plt.plot(xfit,yrec[grp],label='Recovered Curve',color='green')
    plt.plot(xfit, psychfunc(fixed[grp], xfit), label='Original Curve',color='red')
    plt.fill_between(xfit, hdis[grp][:, 0], hdis[grp][:, 1], color='green', alpha=0.3, label='95% HDI')
    plt.scatter(np.array(freqs.index),np.array(freqs[grp_i]),label='Data', color = 'red')
    plt.title(grp+" "+sess_label)
    plt.xlabel('Stimulus Amplitude')
    plt.legend(loc='upper left', fontsize=9.5)
    plt.show()   


#%% Posterior predictive

with model_B:
    pm.sample_posterior_predictive(trace,extend_inferencedata=True)

az.plot_ppc(trace, num_pp_samples=100)

#%% cross validation

az.loo(trace)

#%% compare prior and posterior

with model_B:
    prior = pm.sample_prior_predictive(samples=3000)
trace.extend(prior)


#%%

sess = 13
sess_label = sessions[sess]

sp = data_dict["synth_session_params"].sel(session=sess, par_type="gamma", par_idx="h0")


true_gam_h = {
    "left_uni": float(sp.sel(hand="left", manual="uni")),
    "left_bi": float(sp.sel(hand="left", manual="bi")),
    "right_uni": float(sp.sel(hand="right", manual="uni")),
    "right_bi": float(sp.sel(hand="right", manual="bi"))
}
groups = trace.posterior.coords["groups"].values

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()

for ax, g in zip(axes, groups):
    prior_vals = trace.prior["gam_h"].sel(groups=g, sessions=sess_label).values.reshape(-1)
    post_vals  = trace.posterior["gam_h"].sel(groups=g, sessions=sess_label).values.reshape(-1)

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



