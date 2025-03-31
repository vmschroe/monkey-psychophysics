#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:30:31 2025

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
import numpy as np
import arviz as az
import sys
sys.path.append("/home/vmschroe/Documents/Monkey Analysis/Github")
import FunctionsForBayes as ffb



params_prior_params = [ [], [], [], [] ]
params_prior_range = [ [], [], [], [] ]
#Parameters of prior beta distribution of gamma
params_prior_params[0] = [2,2] #[alpha,beta] 
params_prior_range[0] = [0,0.5] # [min,max]
#Parameters of prior beta distribution of lambda
params_prior_params[1] = [2,2] #[alpha,beta] 
params_prior_range[1] = [0,0.5] # [min,max]
#Parameters of prior beta distribution of b0
params_prior_params[2] = [2,2] #[alpha,beta] 
params_prior_range[2] = [-91,-0.12] # [min,max]
#Parameters of prior beta distribution of b1
params_prior_params[3] = [2,2] #[alpha,beta] 
params_prior_range[3] = [0.02,1.82] # [min,max]

##Load and clean data
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/loaddata.py').read())
## constructs 4 dataframes:
    # df_ld
    # df_ln
    # df_rd
    # df_rn


#Construct necessary functions
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/FunctionsForBayes.py').read())

dfs = [df_ld, df_ln, df_rd, df_rn]
group_names = ["(Left hand, Distracted)", "(Left hand, Not distracted)","(Right hand, Distracted)", "(Right hand, Not distracted)"]

#Plot psychometric curves produced by Bayes estimate of b0?
plot_rec_curves = True
#Plot PD of b0?
plot_post_dist = True
#Return summary of posterior sampling?
print_post_sum = True
#Keep vectors of all the posterior means and HDI's?
store_mean_HDI = True
#Plot posterior mean of b0 as a function of number of trials? Save it?
plot_post_conv = False
save_plot = False

x = [6, 12, 18, 24, 32, 38, 44, 50] #stimulus amplitudes
params_postmeans  = [ [], [], [], [] ]
params_post3hdi  = [ [], [], [], [] ]
params_post97hdi  = [ [], [], [], [] ]
PSE_hdi = [[],[],[],[]]
JND_hdi = [[],[],[],[]]

for ii in np.arange(4):
    print('----------------------------------------------------------------------')
    print('Loading data')
    
    df = dfs[ii]
    group = group_names[ii]
    # Generate sample observed data
    # ydata = ffb.sim_exp_data(sim_params, n)
    # yndata = ydata*n
    ydata, n, yndata, x = ffb.psych_vectors(df)
    nsum = sum(n)
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        gam_norm = pm.Beta("gam_norm",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_range[0][0]+(params_prior_range[0][1]-params_prior_range[0][0])*gam_norm)
        lam_norm = pm.Beta("lam_norm",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_range[1][0]+(params_prior_range[1][1]-params_prior_range[1][0])*lam_norm)
        b0_norm = pm.Beta("b0_norm",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        b0 = pm.Deterministic("b0", params_prior_range[2][0]+(params_prior_range[2][1]-params_prior_range[2][0])*b0_norm)
        b1_norm = pm.Beta("b1_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        b1 = pm.Deterministic("b1", params_prior_range[3][0]+(params_prior_range[3][1]-params_prior_range[3][0])*b1_norm)
    
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=ffb.phi_with_lapses([gam, lam, b0, b1],x), observed=yndata)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
        
    
    # Plot PDs
    if plot_post_dist:
        # Posterior plots with titles
        az.plot_posterior(trace, var_names=["gam"])
        plt.title("PD of gamma parameter, "+group)
        
        az.plot_posterior(trace, var_names=["lam"])
        plt.title("PD of lambda parameter, "+group)
        
        az.plot_posterior(trace, var_names=["b0"])
        plt.title("PD of b0 parameter, "+group)
        
        az.plot_posterior(trace, var_names=["b1"])
        plt.title("PD of b1 parameter, "+group)
        
        # Pair plot with title
        az.plot_pair(trace, var_names=["gam", "lam", "b0", "b1"], kind='kde', marginals=True)
        plt.suptitle("Joint Posteriors of [gamma, lambda, b0, and b1], "+group, fontsize=35)
        
        plt.show()
    
    # Print a summary of the posterior
    if print_post_sum:
        print(az.summary(trace, var_names=["gam", "lam", "b0", "b1"]))
        
    
    # Plot curves using posterior estimates of b0
    if plot_rec_curves:
        xfit = np.linspace(6,50,1000)
        rec_params = [0,0,0,0]
        rec_params[0] = float(az.summary(trace)["mean"]["gam"])  # Use .iloc[0] to extract the scalar value
        rec_params[1] = float(az.summary(trace)["mean"]["lam"])
        rec_params[2] = float(az.summary(trace)["mean"]["b0"])  # Use .iloc[0] to extract the scalar value
        rec_params[3] = float(az.summary(trace)["mean"]["b1"])
        
        ## Construct HDI Curves
        b0_samples = trace.posterior['b0'].values.flatten()
        b1_samples = trace.posterior['b1'].values.flatten()
        gam_samples = trace.posterior['gam'].values.flatten()
        lam_samples = trace.posterior['lam'].values.flatten()
        
        y_samples = np.array([ffb.phi_with_lapses([gam,lam,b0,b1], xfit) for gam, lam, b0, b1 in zip(gam_samples, lam_samples, b0_samples, b1_samples)])
        hdi = az.hdi(y_samples, hdi_prob=0.95)
        yrec = ffb.phi_with_lapses(rec_params,xfit)
        
        PSE_samples = np.array([ffb.PSE(gam,lam,b0,b1) for gam, lam, b0, b1 in zip(gam_samples, lam_samples, b0_samples, b1_samples)])
        JND_samples = np.array([ffb.JND(gam,lam,b0,b1) for gam, lam, b0, b1 in zip(gam_samples, lam_samples, b0_samples, b1_samples)])
        PSE_hdi[ii] = az.hdi(PSE_samples, hdi_prob=0.95)
        JND_hdi[ii] = az.hdi(JND_samples, hdi_prob=0.95)
        
        plt.plot(xfit,yrec,label=f'Recovered Curve with params \n[{rec_params[0]},{rec_params[1]},{rec_params[2]},{rec_params[3]}] ',color='green')
        plt.fill_between(xfit, hdi[:, 0], hdi[:, 1], color='gray', alpha=0.3, label='95% HDI')
        plt.scatter(x,ydata,label=f'Data: Total {int(nsum)} trials', color = 'red')
        plt.title(group)
        plt.xlabel('Stimulus Amplitude')
        plt.legend(loc='upper left', fontsize=9.5)
        plt.show()   
    
    if store_mean_HDI:
        params_postmeans[0].append(float(az.summary(trace)["mean"]["gam"]))
        params_post3hdi[0].append(float(az.summary(trace)["hdi_3%"]["gam"]))
        params_post97hdi[0].append(float(az.summary(trace)["hdi_97%"]["gam"]))
        params_postmeans[1].append(float(az.summary(trace)["mean"]["lam"]))
        params_post3hdi[1].append(float(az.summary(trace)["hdi_3%"]["lam"]))
        params_post97hdi[1].append(float(az.summary(trace)["hdi_97%"]["lam"]))
        params_postmeans[2].append(float(az.summary(trace)["mean"]["b0"]))
        params_post3hdi[2].append(float(az.summary(trace)["hdi_3%"]["b0"]))
        params_post97hdi[2].append(float(az.summary(trace)["hdi_97%"]["b0"]))
        params_postmeans[3].append(float(az.summary(trace)["mean"]["b1"]))
        params_post3hdi[3].append(float(az.summary(trace)["hdi_3%"]["b1"]))
        params_post97hdi[3].append(float(az.summary(trace)["hdi_97%"]["b1"]))
    
    
