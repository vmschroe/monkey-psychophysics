#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:27:45 2024

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

#######
# Constructing Necessary functions
#######

x = [6, 12, 18, 24, 32, 38, 44, 50]


## constructing log likelihood
def phi_with_lapses(params, X):
    X = np.array(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic

def neglogL_fxn_mod(params, ny, n, x): #negative log likelihood function, modified to prevent computation error
    try:
        # Compute phi_with_lapses values
        phi = phi_with_lapses(params, x)
        safe_phi = np.maximum(phi, 1e-8)  # Prevent phi from being 0 or negative
        safe_one_minus_phi = np.maximum(1 - phi, 1e-8)  # Prevent 1 - phi from being 0 or negative
        La_mod = ny * np.log(safe_phi) + (n - ny) * np.log(safe_one_minus_phi)
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        # Handle any errors gracefully by assigning La to 0
        print(f"Error in La calculation: {e}")
        La_mod = 0
    # Calculate the negative log-likelihood (LL), ignoring nCr term
    LL_mod = -np.sum(La_mod)
    return LL_mod


def paramest(n,ny):
    ## OPTIMIZER
    #apply to negll with lapses
    
    def nll_wlapse_mod(params):
        gam, lam, b0,b1 = params
        return neglogL_fxn_mod([gam,lam,b0,b1],ny,n,x)
    
    bounds1 = [(0,0.5),(0,0.5),(-15,0),(0,2)]
    initial_guess = [0.06,0.06,-8,0.3]
    options = {'maxiter': 60}
    result = minimize( nll_wlapse_mod,  initial_guess, method='L-BFGS-B', bounds= bounds1, options=options, )
    
    gam_est= result.x[0]
    lam_est= result.x[1]
    b0_est= result.x[2]
    b1_est= result.x[3]
    return gam_est, lam_est, b0_est, b1_est

def psych_vectors(df):
    x= sorted(df['stimAMP'].unique())
    n = []
    y = []
    for amp in x:
        tempn, _ = df[df['stimAMP']==amp].shape
        tempny,_ = df[ (df['stimAMP']==amp)&(df['lowORhighGUESS']==1)].shape
        tempy = tempny/tempn
        n = np.append(n,tempn)
        y = np.append(y,tempy)
        ny = (n * y).astype(int)
    return y, n, ny, x

def solve_phi_for_X(gamma, lambda_, beta0, beta1, p):
    # Calculate X using the given formula
    X = -( (beta0 - np.log((gamma - p) / (-1 + lambda_ + p))) / beta1 )
    return X

def PSE(gamma, lambda_, beta0, beta1):
    X = solve_phi_for_X(gamma, lambda_, beta0, beta1, 0.5)
    return X

def JND(gamma, lambda_, beta0, beta1):
    X25 = solve_phi_for_X(gamma, lambda_, beta0, beta1, 0.25)
    X75 = solve_phi_for_X(gamma, lambda_, beta0, beta1, 0.75)
    return 0.5*(X75-X25)