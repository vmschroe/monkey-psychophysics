#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:47:28 2024

@author: vmschroe
"""
import numpy as np
from scipy.special import comb
from scipy.special import gammaln

def log_comb(n, ny):
    # Compute log of the binomial coefficient elementwise for vectors n and ny
    return gammaln(n + 1) - gammaln(ny + 1) - gammaln(n - ny + 1)
def phi_with_lapses(params, X):
    X = np.array(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic

n=[1250, 1250, 1250, 1250, 1250, 1250, 1250, 1250]
ny=[   5,   48,  149,  373,  864, 1093, 1214, 1242]
    # Calculate the combination term

# Compute phi_with_lapses values (assuming this is another function you've defined)
phi = [   0,  0,  0,  0,  0, 0, 0, 0]

# Ensure values for log are within a valid range to prevent log(0) or log(negative)
# Use np.maximum to prevent log of zero, assuming tiny positive values for stability
safe_phi = np.maximum(phi, 0.001)  # Prevent phi from being 0 or negative
safe_one_minus_phi = np.maximum(1 - phi, 0.001)#Prevent 1 - phi from being 0 or negative

# Compute La using log-safe values
La = log_comb(n,ny) + ny * np.log(safe_phi) + (n - ny) * np.log(safe_one_minus_phi)

