#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:25:38 2024

@author: vmschroe
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
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

test_params = [0, 0, -5.9, 0.209]
x = [6, 12, 18, 24, 32, 38, 44, 50]
N = 50
n = float(np.floor(N/8))*np.ones(8)
n = n.astype(int)

def phi_with_lapses(params, X):
    X = np.array(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic

#Generate test data y
ny = binom.rvs(n, phi_with_lapses(test_params,x))
y = ny/n

# #Define likelihood function
# def likelihood_fxn(params, ny, n, x):
#     Lf = binom.pmf(ny,n,phi_with_lapses(params,x))
#     L = np.prod(Lf)
#     return L

# #visualizing likelihood for b0 and b1
# b0 = np.linspace(-7,-4,100)
# b1 = np.linspace(0.15,0.3,100)
# b0, b1 = np.meshgrid(b0,b1)

# # Initialize an empty array for the likelihood values
# L = np.zeros_like(b0)

# # Compute likelihood for each (b0, b1) pair
# for i in range(b0.shape[0]):
#     for j in range(b0.shape[1]):
#         L[i, j] = likelihood_fxn([test_params[0], test_params[1], b0[i, j], b1[i, j]], ny, n, x)

# ## 3d surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(b0, b1, L, cmap='viridis')

# # Set labels and title
# ax.set_xlabel('b0')
# ax.set_ylabel('b1')
# ax.set_zlabel('L')
# ax.set_title('3D Surface Plot')

# plt.show()

# ## contour plot

# plt.contour(b0, b1, L, levels=20, cmap='viridis')

# # Set labels and title
# plt.xlabel('b0')
# plt.ylabel('b1')
# plt.title('Contour Plot')

# plt.colorbar()  # Add a colorbar to show the mapping of colors to z values
# plt.show()

## constructing log likelihood

def neglogL_fxn(params, ny, n, x):
    try:
        # Calculate the combination term
        comb_term = comb(n, ny)
        
        # Compute phi_with_lapses values (assuming this is another function you've defined)
        phi = phi_with_lapses(params, x)
        
        # Ensure values for log are within a valid range to prevent log(0) or log(negative)
        # Use np.maximum to prevent log of zero, assuming tiny positive values for stability
        safe_phi = np.maximum(phi, 1e-10)  # Prevent phi from being 0 or negative
        safe_one_minus_phi = np.maximum(1 - phi, 1e-10)  # Prevent 1 - phi from being 0 or negative
        
        # Compute La using log-safe values
        La = np.log(comb_term) + ny * np.log(safe_phi) + (n - ny) * np.log(safe_one_minus_phi)
    
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        # Handle any errors gracefully by assigning La to 0
        print(f"Error in La calculation: {e}")
        La = 0

    # Calculate the negative log-likelihood (LL)
    LL = -np.sum(La)
    return LL

# def neglogL_fxn(params, ny, n, x):
    #     try:
    #         # Calculate La and handle potential issues
    #         La = np.log(comb(n, ny)) + ny * np.log(phi_with_lapses(params, x)) + (n - ny) * np.log(1 - phi_with_lapses(params, x))
    #     except (ValueError, FloatingPointError, ZeroDivisionError) as e:
    #         # If any error occurs, assign La to 0
    #         print(f"Error in La calculation: {e}")
    #         La = 0
    
    #     # Calculate the negative log-likelihood (LL)
    #     LL = -np.sum(La)
    #     return LL
    # def neglogL_fxn(params, ny, n, x):
    #     La = np.log(comb(n,ny))+ny*np.log(phi_with_lapses(params,x))+(n-ny)*np.log(1-phi_with_lapses(params,x))
    #     LL = -np.sum(La)
    #     return LL
    
    # b0 = np.linspace(-7,-4,100)
    # b1 = np.linspace(0.15,0.3,100)
    # b0, b1 = np.meshgrid(b0,b1)
    
    # # Initialize an empty array for the likelihood values
    # lL = np.zeros_like(b0)
    
    # # Compute likelihood for each (b0, b1) pair
    # for i in range(b0.shape[0]):
    #     for j in range(b0.shape[1]):
    #         lL[i, j] = neglogL_fxn([test_params[0], test_params[1], b0[i, j], b1[i, j]], ny, n, x)
    
    # ## 3d surface
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(b0, b1, L, cmap='viridis')
    
    # # Set labels and title
    # ax.set_xlabel('b0')
    # ax.set_ylabel('b1')
    # ax.set_zlabel('L')
    # ax.set_title('3D Surface Plot logL')
    
    # plt.show()
    
    # ## contour plot
    
    # plt.contour(b0, b1, L, levels=20, cmap='viridis')
    
    # # Set labels and title
    # plt.xlabel('b0')
    # plt.ylabel('b1')
    # plt.title('Contour Plot logL')
    
    # plt.colorbar()  # Add a colorbar to show the mapping of colors to z values
    # plt.show()
    
    
    # ## look at behavior of lambda
    
    # b0 = np.linspace(-7,-4,100)
    # lambda_ = np.linspace(0,0.3,100)
    # b0, lambda_ = np.meshgrid(b0,lambda_)
    
    # # Initialize an empty array for the likelihood values
    # lL = np.zeros_like(b0)
    
    # # Compute likelihood for each (b0, lambda_) pair
    # for i in range(b0.shape[0]):
    #     for j in range(b0.shape[1]):
    #         lL[i, j] = neglogL_fxn([test_params[0], lambda_[i,j], b0[i, j], test_params[3]], ny, n, x)
    
    # ## 3d surface
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(b0, lambda_, L, cmap='viridis')
    
    # # Set labels and title
    # ax.set_xlabel('b0')
    # ax.set_ylabel('lambda_')
    # ax.set_zlabel('L')
    # ax.set_title('3D Surface Plot logL')
    
    # plt.show()
    
    # ## contour plot
    
    # plt.contour(b0, b1, L, levels=20, cmap='viridis')
    
    # # Set labels and title
    # plt.xlabel('b0')
    # plt.ylabel('lambda_')
    # plt.title('Contour Plot logL')
    
    # plt.colorbar()  # Add a colorbar to show the mapping of colors to z values
    # plt.show()

## OPTIMIZER

#apply to negll without lapses

def nll_nolapse(params):
    b0,b1 = params
    return neglogL_fxn([0,0,b0,b1],ny,n,x)

bounds = [(-50,0),(0,40)]
initial_guess = [-10,0.1]
options = {'maxiter': 50}
result = minimize( nll_nolapse, initial_guess, method='BFGS', options=options)

b0_est= result.x[0]
b1_est= result.x[1]

#plot
xfit = np.linspace(6,50,1000)
yfit = phi_with_lapses([0,0,b0_est,b1_est],xfit)
plt.scatter(x,y,label='Experimental Data', color='blue')
plt.plot(xfit,yfit,label='Fitted Curve', color='red')
plt.xlabel('Amplitude of Stimulus')
plt.title('Fitted Curve for ' + str(N)+' Trials')
plt.ylabel('Probability of Guess "High"')
#plt.legend()
plt.show()