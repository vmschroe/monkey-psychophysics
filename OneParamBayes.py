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


sim_params = [0, 0, -5.9, 0.209] #[gam, lam, b0,b1]
x = [6, 12, 18, 24, 32, 38, 44, 50]
#num_trials=50000
postmeans = []
post3hdi = []
post97hdi = []
for num_trials in np.arange(10,20010,50):
    n = np.full(8, num_trials, dtype=int)
    
    
    # Generate sample observed data
    ydata = ffb.sim_exp_data(sim_params, n)
    yndata = ydata*n
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        b0 = pm.Normal("b0", mu=-5.8, sigma=2)
        
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=ffb.phi_with_lapses([sim_params[0], sim_params[1], b0, sim_params[3]],x), observed=yndata)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
    
    # Plot posterior distributions
    az.plot_posterior(trace, var_names=["b0"])
    plt.show()    
    # Print a summary of the posterior
    print('----------------------------------------------------------------------')
    print(f'num_trials = {num_trials}')
    print(az.summary(trace, var_names=["b0"]))
    postmeans.append(float(az.summary(trace)["mean"]))
    post3hdi.append(float(az.summary(trace)["hdi_3%"]))
    post97hdi.append(float(az.summary(trace)["hdi_97%"]))
    
    rec_params = sim_params.copy()
    rec_params[2] =  float(az.summary(trace)["mean"])
    lorec_params = sim_params.copy()
    lorec_params[2] =  float(az.summary(trace)["hdi_3%"])
    hirec_params = sim_params.copy()
    hirec_params[2]  =  float(az.summary(trace)["hdi_97%"])
    #######
    # Plots
    ######
    xfit = np.linspace(6,50,1000)
    
    ysim = ffb.phi_with_lapses(sim_params,xfit)
    yrec = ffb.phi_with_lapses(rec_params,xfit)
    loyrec = ffb.phi_with_lapses(lorec_params,xfit)
    hiyrec = ffb.phi_with_lapses(hirec_params,xfit)
    
    
    plt.plot(xfit,ysim,label='Original',color='blue')
    plt.plot(xfit,yrec,label='Recovered',color='green')
    plt.plot(xfit,loyrec,color='green')
    plt.plot(xfit,hiyrec,color='green')
    plt.scatter(x,ydata, color = 'red')
    plt.title(f'Recovering b0 =  {sim_params[2]} from {num_trials} trials')
    plt.xlabel(f'Recovered b0~= {rec_params[2]}')
    plt.legend()
    plt.show()    