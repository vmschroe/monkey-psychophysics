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

result_df = az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])


    # r_hat = 1 and ess is large, so sampling was successful
#%% Look at traceplots

az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_uni'],
    'betas': ["b0", "b1"]}, compact=False,  backend_kwargs={"constrained_layout": True})
#%% plot joint posteriors


for grp_num, grp_choice in enumerate(coords['groups']):
     az.plot_pair(trace, var_names=['gam_h', 'gam_l'
                                    #,'beta_vec'
                                    ,'PSE', 'JND'
                                    ], 
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
    
plt.plot(xfit*x_sig+x_mu,yrec['left_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig+x_mu, hdis['left_uni'][:, 0], hdis['left_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[0]),label='Data', color = 'blue')
plt.plot(xfit*x_sig+x_mu,yrec['left_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig+x_mu, hdis['left_bi'][:, 0], hdis['left_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[1]),label='Data', color = 'red')
plt.hlines(y= 0.5, xmin=0, xmax=51, colors='gray',lw=0.5)
plt.vlines(x=28, ymin=-0.05,ymax=1.05, label="trained threshold", color='green', linestyles='dashed')
plt.xlim(5,51)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Stimulus Amplitude ($\mu m$)')
plt.ylabel('Prob[response = "high"]')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Left Hand Psychometric Curves")
plt.show()     
     
     
#%%


plt.plot(xfit*x_sig+x_mu,yrec['right_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig+x_mu, hdis['right_uni'][:, 0], hdis['right_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[2]),label='Data', color = 'blue')
plt.plot(xfit*x_sig+x_mu,yrec['right_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig+x_mu, hdis['right_bi'][:, 0], hdis['right_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[3]),label='Data', color = 'red')
plt.hlines(y= 0.5, xmin=0, xmax=51, colors='gray',lw=0.5)
plt.vlines(x=28, ymin=-0.05,ymax=1.05, label="trained threshold", color='green', linestyles='dashed')
plt.xlim(5,51)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Stimulus Amplitude ($\mu m$)')
plt.ylabel('Prob[response = "high"]')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Right Hand Psychometric Curves")
plt.show()     
     
     
#%% sample priors and post pred

with model_A:
    pm.sample_posterior_predictive(trace,extend_inferencedata=True)
    prior = pm.sample_prior_predictive(samples=3000)
trace.extend(prior) 




#%% compare prior and posteriors for parameters
#PSE
groups = coords['groups']
fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()
showlegend=0
for ax, g in zip(axes, groups):
    prior_vals = trace.prior["PSE"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["PSE"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]*x_sig+x_mu
    post_vals  = post_vals[np.isfinite(post_vals)]*x_sig+x_mu

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    #x_true = true_gam_h[g]
    ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
    ax.set_xlim(22, 34)
    ax.set_ylim(0,1.7)
    ax.set_title(g)
    if showlegend==0:
        ax.legend(fontsize='small')
        showlegend=1
    else:
        ax.get_legend().remove()
fig.suptitle("PSE", fontsize=16)
plt.show()


#%% JND
fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()
showlegend=0
for ax, g in zip(axes, groups):
    prior_vals = trace.prior["JND"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["JND"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]*x_sig
    post_vals  = post_vals[np.isfinite(post_vals)]*x_sig

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    #x_true = true_gam_h[g]
    #ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
    ax.set_xlim(0, 12)
    ax.set_ylim(0,2.7)
    ax.set_title(g)
    if showlegend==0:
        ax.legend(fontsize='small')
        showlegend=1
    else:
        ax.get_legend().remove()
fig.suptitle("JND", fontsize=16)
plt.show()


#%% gam_h
     

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()
showlegend=0
for ax, g in zip(axes, groups):
    prior_vals = trace.prior["gam_h"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["gam_h"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]
    post_vals  = post_vals[np.isfinite(post_vals)]

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    #x_true = true_gam_h[g]
    #ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0,200)
    ax.set_title(g)
    if showlegend==0:
        ax.legend(fontsize='small')
        showlegend=1
    else:
        ax.get_legend().remove()
fig.suptitle(r"$\gamma_h$", fontsize=16)
plt.show()
     
#%% gam_l
     

fig, axes = plt.subplots(2, 2, constrained_layout=True)
axes = axes.ravel()
showlegend=0
for ax, g in zip(axes, groups):
    prior_vals = trace.prior["gam_l"].sel(groups=g).values.reshape(-1)
    post_vals  = trace.posterior["gam_l"].sel(groups=g).values.reshape(-1)

    # (optional) keep only finite values, safe habit
    prior_vals = prior_vals[np.isfinite(prior_vals)]
    post_vals  = post_vals[np.isfinite(post_vals)]

    az.plot_dist(prior_vals, ax=ax, label="prior", color='blue')
    az.plot_dist(post_vals,  ax=ax, label="posterior", color='purple')
    #x_true = true_gam_h[g]
    #ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0,120)
    ax.set_title(g)
    if showlegend==0:
        ax.legend(fontsize='small')
        showlegend=1
    else:
        ax.get_legend().remove()
fig.suptitle(r"$\gamma_l$", fontsize=16)
plt.show()   


#%% compare unimanual vs bimanual plots
#PSE

uni_grp = 'right_uni'
bi_grp= 'right_bi'

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax=axes


uni_vals  = trace.posterior["PSE"].sel(groups=uni_grp).values.reshape(-1)
bi_vals  = trace.posterior["PSE"].sel(groups=bi_grp).values.reshape(-1)
# (optional) keep only finite values, safe habit

uni_vals  = uni_vals[np.isfinite(uni_vals)]*x_sig+x_mu
bi_vals  = bi_vals[np.isfinite(bi_vals)]*x_sig+x_mu

az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue')
az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue', 
             fill_kwargs={'alpha': 0.2, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red')
az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red', 
             fill_kwargs={'alpha': 0.2, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


#x_true = true_gam_h[g]
ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
#ax.set_xlim(22, 34)
#ax.set_ylim(0,1.7)
ax.set_title("Right Hand")
ax.legend(fontsize='small', loc='upper center')
fig.suptitle("Posterior Estimates: PSE", fontsize=16)
plt.show()

#%% JND
     

uni_grp = 'left_uni'
bi_grp= 'left_bi'

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax=axes


uni_vals  = trace.posterior["JND"].sel(groups=uni_grp).values.reshape(-1)
bi_vals  = trace.posterior["JND"].sel(groups=bi_grp).values.reshape(-1)
# (optional) keep only finite values, safe habit

uni_vals  = uni_vals[np.isfinite(uni_vals)]*x_sig
bi_vals  = bi_vals[np.isfinite(bi_vals)]*x_sig

az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue')
az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue', 
             fill_kwargs={'alpha': 0.3, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red')
az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red', 
             fill_kwargs={'alpha': 0.3, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


#x_true = true_gam_h[g]
#ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
#ax.set_xlim(22, 34)
#ax.set_ylim(0,1.7)
ax.set_title("Left Hand")
ax.legend(fontsize='small')
fig.suptitle("Posterior Estimates: JND", fontsize=16)
plt.show()

#%% lapse parameters
uni_grp = 'right_uni'
bi_grp= 'right_bi'
# uni_grp = 'left_uni'
# bi_grp= 'left_bi'
gam_type = 'gam_l'

fig, axes = plt.subplots(1, 1, constrained_layout=True)
ax=axes


uni_vals  = trace.posterior[gam_type].sel(groups=uni_grp).values.reshape(-1)
bi_vals  = trace.posterior[gam_type].sel(groups=bi_grp).values.reshape(-1)
# (optional) keep only finite values, safe habit

uni_vals  = uni_vals[np.isfinite(uni_vals)]
bi_vals  = bi_vals[np.isfinite(bi_vals)]

az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue')
az.plot_dist(uni_vals,  ax=ax, label="Unimanual", color='blue', 
             fill_kwargs={'alpha': 0.3, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red')
az.plot_dist(bi_vals,  ax=ax, label="Bimanual", color='red', 
             fill_kwargs={'alpha': 0.3, 'label':"95% HDI and mean"}, 
             quantiles= [0.025, 0.5, 0.9725])


#x_true = true_gam_h[g]
#ax.axvline(28, linestyle="--", linewidth=2, label="trained threshold", color='green')
#ax.set_xlim(22, 34)
#ax.set_ylim(0,1.7)
ax.set_title("Right Hand")
#ax.set_title("Left Hand")
ax.legend(fontsize='small')
fig.suptitle(r"Posterior Estimates: $\gamma_l$", fontsize=16)
plt.show()


#%%

az.plot_ppc(trace, num_pp_samples=100)

LOO_results = az.loo(trace)

fit_results = {'az_summary_trace': result_df,
               'az_loo_trace': LOO_results}

with open("Results_A.pkl","wb") as f:
    pickle.dump(fit_results, f)
