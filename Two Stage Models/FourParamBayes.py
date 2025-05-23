#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:14:20 2025

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

#Parameters used to simulate data
sim_params = [0.01, 0.04, -6.1, 0.23] #[gam, lam, b0,b1]

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


#Vector that gives number of trials per stimAMP for the simulation, length = number of simulations
NumTrialsVec = [300]

#Plot psychometric curves produced by Bayes estimate of b0?
plot_rec_curves = True
#Plot posterior distribution of b0?
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


for num_trials in NumTrialsVec:
    print('----------------------------------------------------------------------')
    print(f'Simulation of {num_trials} trials')
    n = np.full(8, num_trials, dtype=int)
    
    # Generate sample observed data
    ydata = ffb.sim_exp_data(sim_params, n)
    yndata = ydata*n
    
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
        trace = pm.sample(500, return_inferencedata=True)
        
    
    # Plot posterior distributions
    if plot_post_dist:
        # Posterior plots with titles
        az.plot_posterior(trace, var_names=["gam"])
        plt.title("Posterior Distribution of gamma parameter")
        
        az.plot_posterior(trace, var_names=["lam"])
        plt.title("Posterior Distribution of lambda parameter")
        
        az.plot_posterior(trace, var_names=["b0"])
        plt.title("Posterior Distribution of b0 parameter")
        
        az.plot_posterior(trace, var_names=["b1"])
        plt.title("Posterior Distribution of b1 parameter")
        
        # Pair plot with title
        az.plot_pair(trace, var_names=["gam", "lam", "b0", "b1"], kind='kde')
        plt.suptitle("Joint Posteriors of [gamma, lambda, b0, and b1]", fontsize=35)
        
        plt.show()
    
    # Print a summary of the posterior
    if print_post_sum:
        print(az.summary(trace, var_names=["b0", "b1"]))
        
    
    # Plot curves using posterior estimates of b0
    if plot_rec_curves:
        xfit = np.linspace(6,50,1000)
        rec_params = sim_params.copy()
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
        ysim = ffb.phi_with_lapses(sim_params,xfit)
        yrec = ffb.phi_with_lapses(rec_params,xfit)
        
        
        
        plt.plot(xfit,ysim,label=f'Simulation Curve, params \n =[{sim_params[0]}, {sim_params[1]},{sim_params[2]}, {sim_params[3]}]',color='blue')
        plt.plot(xfit,yrec,label=f'Using posterior means of \n[{rec_params[0]},{rec_params[1]},{rec_params[2]},{rec_params[3]}] ',color='green')
        plt.fill_between(xfit, hdi[:, 0], hdi[:, 1], color='gray', alpha=0.3, label='95% HDI')
        plt.scatter(x,ydata,label=f'Simulated Data: {num_trials} trials', color = 'red')
        plt.title('Psychometric Curve (w/ lapses), Bayes estimates of [gam,lam,b0,b1]')
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
    

if plot_post_conv and store_mean_HDI:
    # Create a figure with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Plot for param[0]
    ffb.plot_param_conv(
        axes[0], NumTrialsVec, params_post3hdi[0], params_post97hdi[0], params_postmeans[0],
        sim_params[0], "param0")
    
    # Plot for param[1]
    ffb.plot_param_conv(
        axes[1], NumTrialsVec, params_post3hdi[1], params_post97hdi[1], params_postmeans[1],
        sim_params[1], "param1")
    
    # Plot for b0
    ffb.plot_param_conv(
        axes[2], NumTrialsVec, params_post3hdi[2], params_post97hdi[2], params_postmeans[2],
        sim_params[2], "b0")
    
    # Plot for b1
    ffb.plot_param_conv(
        axes[3], NumTrialsVec, params_post3hdi[3], params_post97hdi[3], params_postmeans[3],
        sim_params[3], "b1")
    
    # Adjust layout and save the figure
    #plt.tight_layout()
    if save_plot:
        output_directory = "/home/vmschroe/Documents/Monkey Analysis/Github"  # Replace with your desired directory
        os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
        output_filename = "ConvOfFourVarBayesEst"
        plt.savefig(os.path.join(output_directory, f"{output_filename}.png"), dpi=300)
        print(f"Plots saved to {output_directory} as {output_filename}.png")
    plt.show()
    