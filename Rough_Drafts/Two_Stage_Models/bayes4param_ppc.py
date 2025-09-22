# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:38:22 2025

Updated and organized bayesian model with sessions pooled

@author: schro
"""



import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import os
import math

import pymc as pm
import numpy as np
import arviz as az
import sys
sys.path.append("monkey-psychophysics/Two_Stage_Models")
from Two_Stage_Models import FunctionsForBayes as ffb
import pickle
import seaborn as sns

x = [6, 12, 18, 24, 32, 38, 44, 50] #stimulus amplitudes
params_prior_params = [ [], [], [], [] ]
params_prior_scale = [1,1,1,1]
#Parameters of prior beta distribution of W_gamma
params_prior_params[0] = [2,5] #[alpha,beta] 
params_prior_scale[0] = 0.25 # param = scale * W
#Parameters of prior beta distribution of W_lambda
params_prior_params[1] = [2,5] #[alpha,beta] 
params_prior_scale[1] = 0.25 # [min,max]
#Parameters of prior gamma distribution of W_b0
params_prior_params[2] = [4,1] #[alpha,beta] 
params_prior_scale[2] = -1 # [min,max]
#Parameters of prior gamma distribution of W_b1
params_prior_params[3] = [1.25,2.4] #[alpha,beta] 
params_prior_scale[3] = 1 # [min,max]
plot_all_posteriors = False


#data, n vector and c vector
with open("Data/psych_vecs_all.pkl", "rb") as f:
    data = pickle.load(f)  


Cagg_mats = {}
Nagg_mats = {}
data_dict = {}

for grp in ['ld','ln','rd','rn']:
    data_dict[grp] = {}
    C = np.array([*data['NY'][grp].values()])
    data_dict[grp]['C_mat'] = C
    Cagg_mats[grp] = sum(C)
    
    N = np.array([*data['N'][grp].values()])
    data_dict[grp]['N_mat'] = N
    Nagg_mats[grp] = sum(N)


##

trace_pools = {}
prior_pred_pools = {}
post_pred_pools = {}

for grp in ['ld','ln','rd','rn']:
    
    with pm.Model() as model_pooled:
        # Define priors for the parameters
        W_gam = pm.Beta("W_gam",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_scale[0]*W_gam)
        W_lam = pm.Beta("W_lam",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_scale[1]*W_lam)
        W_b0 = pm.Gamma("W_b0",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        b0 = pm.Deterministic("b0", params_prior_scale[2]*W_b0)
        W_b1 = pm.Gamma("b1_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        b1 = pm.Deterministic("b1", params_prior_scale[3]*W_b1)
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", ffb.PSE(gam, lam, b0, b1))
        jnd = pm.Deterministic("jnd", ffb.JND(gam, lam, b0, b1))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=Nagg_mats[grp], p=ffb.phi_with_lapses([gam, lam, b0, b1],x), observed=Cagg_mats[grp])
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace_pooled = pm.sample(1000, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
        prior_pred_pooled = pm.sample_prior_predictive()
        post_pred_pooled = pm.sample_posterior_predictive(trace_pooled)
    
    trace_pools[grp] = trace_pooled
    prior_pred_pools[grp] = prior_pred_pooled
    post_pred_pools[grp] = post_pred_pooled
    
    post_pred_mean = np.array(az.summary(post_pred_pooled.posterior_predictive)['mean'])
    post_pred_hdi3 = np.array(az.summary(post_pred_pooled.posterior_predictive)['hdi_3%'])
    post_pred_hdi97 = np.array(az.summary(post_pred_pooled.posterior_predictive)['hdi_97%'])
    obs = Cagg_mats[grp]
    #
    
    #plot post pred vs observed
    plt.scatter(obs,post_pred_mean, label = 'posterior predicted mean')
    plt.plot(obs,obs,':')
    plt.fill_between(obs,post_pred_hdi3,post_pred_hdi97, alpha = 0.4, label = 'posterior predicted 95% HDI')
    plt.xlabel('observed counts')
    plt.ylabel('posterior predicted counts')
    plt.title(f'Posterior Predictive Check, Group: {grp}')
    plt.legend()
    plt.show()
    
    #plot post pred and observed on psychometric
    rec_params = np.array(az.summary(trace_pooled.posterior)['mean'][['gam', 'lam','b0','b1']])
    xfit = np.linspace(6,50,1000)
    yfit = ffb.phi_with_lapses(rec_params, xfit)
    plt.plot(xfit,yfit,color = 'green', label = "Computed Psychometric Curve")
    plt.errorbar(x, post_pred_mean/Nagg_mats[grp], yerr=[(post_pred_mean-post_pred_hdi3)/Nagg_mats[grp],(post_pred_hdi97-post_pred_mean)/Nagg_mats[grp]], fmt = '_', capsize = 6, ms = 12, elinewidth=2, zorder=0, label = 'PP mean & 95% HDI')
    plt.scatter(x, obs/Nagg_mats[grp], color = 'red', alpha = 0.9, marker = '.', zorder=10, label = 'observed')
    plt.fill_between([10,11],[-1,-1],[-1.9,-1.9], color = 'skyblue', label = 'PP Distribution')
    plt.ylim(-0.1, 1.1)
    plt.title(f'PPC on Psychometric Function, {grp}')
    plt.legend()
    plt.savefig(f'PPC_ranges_on_Psych_Funct_{grp}.png')
    plt.show()
    
    
    
    observed = post_pred_pooled.observed_data["obs"].values
    predicted = post_pred_pooled.posterior_predictive["obs"].stack(sample=("chain", "draw")).values.T  # shape: (samples, 8)
    
    # Create PPC plots for each observation
    
    for i in range(8):
        plt.figure(figsize=(1.5, 12))
        sns.kdeplot(y = (predicted[:, i]/Nagg_mats[grp][i]), label="Posterior Predictive", fill=True)
        plt.axhline(observed[i]/Nagg_mats[grp][i], linestyle="--", label="Observed")
        plt.title(f"PPC {x[i]}")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
        plt.tight_layout()
        plt.ylim(-0.1, 1.1)
        plt.xlim(0,55)
        plt.show()
        
    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(10, 12), sharey=True)
    
    for i, ax in enumerate(axes):
        values = predicted[:, i] / Nagg_mats[grp][i]
        obs_val = observed[i] / Nagg_mats[grp][i]
    
        sns.kdeplot(y=values, ax=ax, fill=True)
        ax.axhline(obs_val, linestyle="--", color="red", label="Observed")
    
        ax.set_title(f"PPC {x[i]}", fontsize=9)
        ax.set_xlim(0, 55)
        ax.set_ylim(-0.1, 1.1)
    
        if i == 0:
            ax.set_ylabel("Value")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    
        ax.set_xlabel("Density")
    
    plt.suptitle(f"Group {grp}")
    plt.savefig(f'PPC_per_stim_{grp}.png')
    plt.show()

#IC 

LOOs = {}
WAICs = {}

for grp in ['ld','ln','rd','rn']:
    LOOs[grp] = az.loo(trace_pools[grp])
    WAICs[grp] = az.waic(trace_pools[grp])
    
az.compare(trace_pools)
