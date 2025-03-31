#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:37:00 2024

@author: vmschroe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:58:43 2024

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

test_params = [0.09, 0.06, -5.9, 0.209]
#Nrange = np.full(100,10000) #look at spread of estimates
Nrange = 8*np.linspace(15, 1000, 5) #looking at impact of N size

x = [6, 12, 18, 24, 32, 38, 44, 50]

b0_ests = []
b1_ests = []
lam_ests = []
gam_ests = []

    
def ln_nCk(n, k):
    # Ensure n and k are numpy arrays
    n = np.asarray(n)
    k = np.asarray(k)
    
    # Use gammaln to compute ln(nCk) in a numerically stable way
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

## constructing log likelihood
def phi_with_lapses(params, X):
    X = np.array(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic


def neglogL_fxn_mod(params, ny, n, x):
    try:
        
        # Compute phi_with_lapses values
        phi = phi_with_lapses(params, x)
        
        # Ensure values for log are within a valid range to prevent log(0) or log(negative)
        # Use np.maximum to prevent log of zero, assuming tiny positive values for stability
        safe_phi = np.maximum(phi, 1e-8)  # Prevent phi from being 0 or negative
        safe_one_minus_phi = np.maximum(1 - phi, 1e-8)  # Prevent 1 - phi from being 0 or negative
        
        # Compute La using log-safe values
        #La = ln_nCk(n,ny) + ny * np.log(safe_phi) + (n - ny) * np.log(safe_one_minus_phi)
        # modify function to remove log(nCr) term, doesn't depend on parameters
        La_mod = ny * np.log(safe_phi) + (n - ny) * np.log(safe_one_minus_phi)
    
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        # Handle any errors gracefully by assigning La to 0
        print(f"Error in La calculation: {e}")
        La_mod = 0

    # Calculate the negative log-likelihood (LL)
    LL_mod = -np.sum(La_mod)
    return LL_mod

##### Let's look at convexity
meshsize = 20
gam_mesh = np.linspace(0,0.25,meshsize)
lam_mesh = np.linspace(0,0.25,meshsize)
b0_mesh = np.linspace(-15,0,meshsize)
b1_mesh = np.linspace(0,15,meshsize)
param_mesh = np.array([gam_mesh, lam_mesh, b0_mesh, b1_mesh])

L4d = np.zeros((meshsize,meshsize,meshsize,meshsize))

#####







for N in Nrange:
    n = float(np.floor(N/8))*np.ones(8)
    n = n.astype(int)
    
    
    #Generate test data y
    # ny = binom.rvs(n, phi_with_lapses(test_params,x))
    # y = ny/n
    y = [0,0,0,0,1,1,1,1]
    ny = n*y
    ## OPTIMIZER
    #apply to negll without lapses
    
    def nll_nolapse_mod(params):
        gam, lam, b0,b1 = params
        return neglogL_fxn_mod([gam,lam,b0,b1],ny,n,x)
    
    bounds1 = [(-0.01,0.5),(-0.01,0.5),(-50,0),(0,40)]
    initial_guess = [0.05,0.05,-10,0.1]
    options = {'maxiter': 50}
    result = minimize( nll_nolapse_mod,  initial_guess, method='BFGS', bounds= bounds1, options=options, )
    
    gam_est= result.x[0]
    lam_est= result.x[1]
    b0_est= result.x[2]
    b1_est= result.x[3]
    gam_ests.append(gam_est)
    lam_ests.append(lam_est)
    b0_ests.append(b0_est)
    b1_ests.append(b1_est)
    
    
#plot
xfit = np.linspace(6,50,1000)
ytrue = phi_with_lapses(test_params,xfit)

for i in np.arange(0,5):
    yrec = phi_with_lapses([gam_ests[i],lam_ests[i],b0_ests[i],b1_ests[i]],xfit)
    plt.plot(xfit,yrec,label='Recovered Curve',color='blue')
plt.plot(xfit,ytrue,label='True Curve', color='red', linewidth=2.5)
plt.xlabel('Amplitude of Stimulus')
plt.title('Fitted Curve for ' + str(Nrange[i])+' Trials')
plt.ylabel('Probability of Guess "High"')
plt.legend()
plt.show()    
    
    
plt.scatter(Nrange,gam_ests)
plt.plot(Nrange,np.full(Nrange.shape ,test_params[0]),color='red')
plt.xlabel('Number of Trials')
plt.title('gamma estimates vs number of trials')
plt.ylabel('gamma estimates')
plt.show()

plt.scatter(Nrange,lam_ests)
plt.plot(Nrange,np.full(Nrange.shape, test_params[1]),color='red')
plt.xlabel('Number of Trials')
plt.title('lambda estimates vs number of trials')
plt.ylabel('lambda estimates')
plt.show()    
    
plt.scatter(Nrange,b0_ests)
plt.plot(Nrange,np.full(Nrange.shape, test_params[2]),color='red')
plt.xlabel('Number of Trials')
plt.title('b0 estimates vs number of trials')
plt.ylabel('b0 estimates')
plt.show()

plt.scatter(Nrange,b1_ests)
plt.plot(Nrange,np.full(Nrange.shape, test_params[3]),color='red')
plt.xlabel('Number of Trials')
plt.title('b1 estimates vs number of trials')
plt.ylabel('b1 estimates')
plt.show()

plt.scatter(b0_ests,b1_ests)
plt.scatter(test_params[2],test_params[3],color='red')
plt.xlabel('b0 estimates')
plt.title('b0 estimates vs b1 estimates')
plt.ylabel('b1 estimates')
plt.show()


# Create a histogram
plt.hist(gam_ests, bins=20, edgecolor='black')
# Add labels and title
plt.xlabel('gamma')
plt.ylabel('Frequency')
plt.title('Histogram of gamma estimates')
# Show the plot
plt.show()

# Create a histogram
plt.hist(lam_ests,  bins=20,edgecolor='black')
# Add labels and title
plt.xlabel('lambda')
plt.ylabel('Frequency')
plt.title('Histogram of lambda estimates')
# Show the plot
plt.show()

# Create a histogram
plt.hist(b0_ests, bins=20, edgecolor='black')
# Add labels and title
plt.xlabel('b0')
plt.ylabel('Frequency')
plt.title('Histogram of b0 estimates')
# Show the plot
plt.show()

# Create a histogram
plt.hist(b1_ests,  bins=20,edgecolor='black')
# Add labels and title
plt.xlabel('b1')
plt.ylabel('Frequency')
plt.title('Histogram of b1 estimates')
# Show the plot
plt.show()


 