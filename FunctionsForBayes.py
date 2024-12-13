#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:15:11 2024

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

#test_params = [0.09, 0.06, -5.9, 0.209]
x = [6, 12, 18, 24, 32, 38, 44, 50]

## constructing log likelihood
def phi_with_lapses(params, X):
    """
    Psychometric function with lapses

    Parameters:
    params : [gamma, lambda_, beta0, beta1]
    X : Stimulus amplitude level

    Returns:
    probability of guess "high"

    """
    X = np.asarray(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic

def sim_exp_data(phi_params, num_trials, num_repeats=1):
    """
    Simulates experimental data using parameters for psychometric function with lapses and trial counts, allowing for repeated simulations.


    Parameters:
        phi_params (list or array): parameters for psychometric function phi. 
            should have 2 elements [gamma = 0, lambda_ = 0 , beta0, beta1] 
            or 4 elements [gamma, lambda_, beta0, beta1]
        num_trials (int, list, or array): Number of trials per amplitude level Can be a single integer or a list/array of length 1 or 8.

    Returns:
        array: y (fraction of successes) for each stim amp level, more rows for more sims
              Raises ValueError for invalid inputs.
    """
    #adjust phi
    if len(phi_params) == 2:
        test_params = np.array([0, 0, *phi_params])
    elif len(phi_params) == 4:
        test_params = np.array(phi_params)
    else:
        raise ValueError("phi_params must have 2 or 4 elements.")
    #adjust n
    if isinstance(num_trials, int):
        n = np.full(8, num_trials, dtype=int)
    elif len(num_trials) == 1:
        n = np.full(8, num_trials[0], dtype=int)
    elif len(num_trials) == 8:
        n = np.array(num_trials, dtype=int)
    else:
        raise ValueError("num_trials must have 1 or 8 elements.")
    # Perform simulations
    results = []
    for _ in range(num_repeats):
        # Generate test data y
        ny = binom.rvs(n, phi_with_lapses(test_params, x))
        y = ny / n
        results.append(y)
    return np.array(results)