#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:17:02 2024

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
#Parameters of prior beta distribution of b0
prior_params = [2,2] #[alpha,beta] 
prior_range = [-91,-0.12] # [min,max]
#Vector that gives number of trials per stimAMP for the simulation, length = number of simulations
NumTrialsVec = [200]

#Plot psychometric curves produced by Bayes estimate of b0?
plot_rec_curves = True
#Plot posterior distribution of b0?
plot_post_dist = True
#Return summary of posterior sampling?
print_post_sum = True
#Keep vectors of all the posterior means and HDI's?
store_mean_HDI = False
#Plot posterior mean of b0 as a function of number of trials? Save it?
plot_post_conv = False
save_plot = False

x = [6, 12, 18, 24, 32, 38, 44, 50] #stimulus amplitudes
postmeans = []
post3hdi = []
post97hdi = []


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
        # b0 = pm.Normal("b0", mu=priorb0mu, sigma=2)
        b0_norm = pm.Beta("b0_norm",alpha=prior_params[0],beta=prior_params[1])
        b0 = pm.Deterministic("b0", prior_range[0]+(prior_range[1]-prior_range[0])*b0_norm)
    
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=ffb.phi_with_lapses([sim_params[0], sim_params[1], b0, sim_params[3]],x), observed=yndata)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(500, return_inferencedata=True)
        
    
    # Plot posterior distributions
    if plot_post_dist:
        az.plot_posterior(trace, var_names=["b0"])
        plt.show()    
    
    
    # Print a summary of the posterior
    if print_post_sum:
        print(az.summary(trace, var_names=["b0"]))
    
    # Plot curves using posterior estimates of b0
    if plot_rec_curves:
        rec_params = sim_params.copy()
        rec_params[2] = float(az.summary(trace)["mean"]["b0"])  # Use .iloc[0] to extract the scalar value
        lorec_params = sim_params.copy()
        lorec_params[2] = float(az.summary(trace)["hdi_3%"]["b0"])  # Use .iloc[0]
        hirec_params = sim_params.copy()
        hirec_params[2] = float(az.summary(trace)["hdi_97%"]["b0"])  # Use .iloc[0]
        xfit = np.linspace(6,50,1000)
        
        ysim = ffb.phi_with_lapses(sim_params,xfit)
        yrec = ffb.phi_with_lapses(rec_params,xfit)
        loyrec = ffb.phi_with_lapses(lorec_params,xfit)
        hiyrec = ffb.phi_with_lapses(hirec_params,xfit)
        
        
        plt.plot(xfit,ysim,label=f'Simulation Curve with b0 = {sim_params[2]}',color='blue')
        plt.plot(xfit,yrec,label=f'Using posterior mean of b0 = {rec_params[2]}',color='green')
        plt.plot(xfit,loyrec,label='Posterior 94% HDI',color='gray')
        plt.plot(xfit,hiyrec,color='gray')
        plt.scatter(x,ydata,label=f'Simulated Data: {num_trials} trials', color = 'red')
        plt.title('Psychometric Curve (w/ lapses), Bayes est of b0')
        plt.xlabel('Stimulus Amplitude')
        plt.legend()
        plt.show()   
    
    if store_mean_HDI:
        postmeans.append(float(az.summary(trace)["mean"]["b0"]))
        post3hdi.append(float(az.summary(trace)["hdi_3%"]["b0"]))
        post97hdi.append(float(az.summary(trace)["hdi_97%"]["b0"]))
    
prior_mean = prior_range[0]+(prior_range[1]-prior_range[0])*prior_params[0]/(prior_params[0]+prior_params[1])
if plot_post_conv and store_mean_HDI:
    plt.scatter(NumTrialsVec,post3hdi,label = "Posterior 94% HDI",color='green')
    plt.scatter(NumTrialsVec,post97hdi,color='green')
    plt.scatter(NumTrialsVec,postmeans, label = "Posterior Means",color='blue')
    plt.plot(NumTrialsVec,np.full(len(postmeans),sim_params[2]), label = f"Data Simulation b0 = {sim_params[2]}", color="red")
    plt.plot(NumTrialsVec,np.full(len(postmeans),prior_mean), label = f"Prior Mean = {prior_mean}", color="purple")
    plt.title('Recovering b0, with beta prior')
    plt.xlabel('Number of Trials')
    plt.ylabel('Parameter bo')
    plt.legend()
    if save_plot:
        output_directory = "/home/vmschroe/Documents/Monkey Analysis/Github"  # Replace with your desired directory
        os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
        output_filename = "ConvergenceOfOneVariableBayesEst"
        plt.savefig(os.path.join(output_directory, f"{output_filename}.png"), dpi=300)
        print(f"Plots saved to {output_directory} as {output_filename}.png")
    plt.show()