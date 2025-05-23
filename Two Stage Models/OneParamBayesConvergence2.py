#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:19:13 2025

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
x = [6, 12, 18, 24, 32, 38, 44, 50] #stimulus amplitudes
numtrialsvec = np.arange(50,1500,50)
numsims = 4

def oneparamconv(numtrialsvec, numsims):
    plot_post_conv = True
    save_plot = False
    
    MEANpostmeans = []
    MEANpost3hdi = []
    MEANpost97hdi = []
    
    for nn in numtrialsvec:
        postmeans = []
        post3hdi = []
        post97hdi = []
        NumTrialsVec = np.full(numsims,nn)
        
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
                
            postmeans.append(float(az.summary(trace)["mean"]["b0"]))
            post3hdi.append(float(az.summary(trace)["hdi_3%"]["b0"]))
            post97hdi.append(float(az.summary(trace)["hdi_97%"]["b0"]))
        
        MEANpostmeans.append(np.mean(postmeans))
        MEANpost3hdi.append(np.mean(post3hdi))
        MEANpost97hdi.append(np.mean(post97hdi))
    if plot_post_conv:
        #prior_mean = prior_range[0]+(prior_range[1]-prior_range[0])*prior_params[0]/(prior_params[0]+prior_params[1])
        plt.scatter(numtrialsvec,MEANpost3hdi,label = "Posterior 94% HDI",color='green')
        plt.scatter(numtrialsvec,MEANpost97hdi,color='green')
        plt.scatter(numtrialsvec,MEANpostmeans, label = "Posterior Means",color='blue')
        
        plt.plot(numtrialsvec,np.full(len(numtrialsvec),sim_params[2]), label = f"Data Simulation b0 = {sim_params[2]}", color="red")
        #plt.plot(numtrialsvec,np.full(len(numtrialsvec),prior_mean), label = f"Prior Mean = {prior_mean}", color="purple")
        plt.title('Recovering b0, with beta prior')
        plt.xlabel('Number of Trials')
        plt.ylabel('Parameter b0')
        plt.legend()
        if save_plot:
            output_directory = "/home/vmschroe/Documents/Monkey Analysis/Github"  # Replace with your desired directory
            os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
            output_filename = "MeanConvOfOneVarBayesEst"
            plt.savefig(os.path.join(output_directory, f"{output_filename}.png"), dpi=300)
            print(f"Plots saved to {output_directory} as {output_filename}.png")
        plt.show()
    return MEANpostmeans, MEANpost3hdi, MEANpost97hdi

oneparamconv(numtrialsvec, numsims)