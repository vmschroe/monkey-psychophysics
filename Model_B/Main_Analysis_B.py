#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:41:58 2026

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
with open("ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
#%% 
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']

sess_summary = data_dict['session_summary']
#%%%

exec(open("Build_Model_B.py").read())


#%% Sample from posteriors

with model_B:
    trace = pm.sample(draws = 2000, return_inferencedata=True, chains = 4, cores = 1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")


#%% Look at r_hats and effective sample sizes

result_df = az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])

#result_df[ result_df['r_hat']>1 ]
    # r_hat = 1 and ess is large, so sampling was successful
#%% Look at traceplots

az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0", "b1"], 
    'sessions':[sessions[5],sessions[10], '05-30']}, compact=False,  backend_kwargs={"constrained_layout": True})


#%% plot joint posteriors

sess_choice = '05-30'

for grp_num, grp_choice in enumerate(coords['groups']):
     az.plot_pair(trace, var_names=['gam_h', 'gam_l'
                                    ,'beta_vec'
                                    #,'PSE', 'JND'
                                    ], 
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice], 'sessions': [sess_choice]}, 
             kind = 'kde', marginals=True)
     

#%%

az.plot_trace(trace, var_names=('gam_h', 'mu_gams', 'sig_gams'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0"], 
    'sessions':['05-30'],}, compact=False,  backend_kwargs={"constrained_layout": True})


#%% weird joint posterior

az.plot_pair(trace, var_names=['gam_h', 'mu_gams', 'sig_gams'], 
         coords = {
             'groups': ['left_bi'],
             'betas': ["b0"], 
             'sessions':['05-30'],},  kind = 'kde', marginals=True)

#%% Plot curves per session and group
sess = 25
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

#%%

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
    
    # plt.plot(xfit,yrec[grp],label='Recovered Curve',color='green')
    # plt.fill_between(xfit, hdis[grp][:, 0], hdis[grp][:, 1], color='green', alpha=0.3, label='95% HDI')
    # plt.scatter(np.array(freqs.index),np.array(freqs[grp_i]),label='Data', color = 'red')
    # plt.title(grp+" "+sess_label)
    # plt.xlabel('Stimulus Amplitude')
    # plt.legend(loc='upper left', fontsize=9.5)
    # plt.show()   


#%% 
x_old = [6,12,18,24,32,38,44,50]
x_mu = np.mean(x_old)
x_sig = np.std(x_old)


    
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
plt.title("Left Hand Psychometric Curves, Session " + sess_label)
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
plt.title("Right Hand Psychometric Curves, Session " + sess_label)
plt.show()     
     

#%%
 