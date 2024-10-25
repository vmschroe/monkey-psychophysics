#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:32:56 2024

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

#did I do it?
#######
# Constructing Necessary functions
#######

x = [6, 12, 18, 24, 32, 38, 44, 50]
    
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


def paramest(n,ny):
    ## OPTIMIZER
    #apply to negll without lapses
    
    def nll_nolapse_mod(params):
        gam, lam, b0,b1 = params
        return neglogL_fxn_mod([gam,lam,b0,b1],ny,n,x)
    
    bounds1 = [(0,0.5),(0,0.5),(-15,0),(0,2)]
    initial_guess = [0.06,0.06,-8,0.3]
    options = {'maxiter': 60}
    result = minimize( nll_nolapse_mod,  initial_guess, method='L-BFGS-B', bounds= bounds1, options=options, )
    
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

#######
# Bringing in Data
#######

#load in data into dataframe
df = pd.DataFrame()
directory_path = '/home/vmschroe/Documents/Monkey Analysis/Sirius_data(1)'

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        
        # Read the Excel file into a temporary DataFrame
        dftemp = pd.read_excel(file_path, engine='openpyxl')
        
        # Concatenate the temporary DataFrame with the main DataFrame
        df = pd.concat([df, dftemp], axis=0)    
df.head(10)

##

# drop aborted trials
df = df[df['outcomeID'].isin([10, 31, 32, 33, 34])]
df.shape
# drop trials with frequency manipulation
df = df[df['tact2_Left_FR'].isin([0, 200])]
df = df[df['tact2_Right_FR'].isin([0, 200])]
df.shape
# drop repeated trials
df = df[(df['repeatedTrial'] == False)]
df.shape

##

# make new column for distraction status
distracted = df['tact2_Left_AMP']*df['tact2_Right_AMP']!=0
df['distracted'] = distracted
# make new column for stimulus amplitude
stimAMP = 10*(df['tact1_Left_DUR']*df['tact2_Left_AMP']+df['tact1_Right_DUR']*df['tact2_Right_AMP'])
df['stimAMP'] = stimAMP
# make new column for stimulus side
conditions = [df['tact1_Left_DUR'] == 0.1]
choices = ['left']
# Use np.select to assign new column
df['stimSIDE'] = np.select(conditions, choices, default='right')
# make new column for AMP guess: low=0, high=1
conditions2 = [(df['selectedChoiceTarget_ID'] == 1) | (df['selectedChoiceTarget_ID'] == 2)]
choices2 = ['0']
# Use np.select to assign new column
df['lowORhighGUESS'] = np.select(conditions2, choices2, default='1')
# make new column for "correct"
df['correct'] = (df['outcomeID']==10)
# convert stimAMP and lowORhighGUESS to integers 
df['stimAMP'] = df['stimAMP'].astype(int)
df['lowORhighGUESS'] = df['lowORhighGUESS'].astype(int)


# separate into 4 dataframes based on hand/distraction
df_ld = df[(df['stimSIDE'] == 'left') & (df['distracted'] == True)]
df_ln = df[(df['stimSIDE'] == 'left') & (df['distracted'] == False)]
df_rd = df[(df['stimSIDE'] == 'right') & (df['distracted'] == True)]
df_rn = df[(df['stimSIDE'] == 'right') & (df['distracted'] == False)]

y_ld, n_ld, ny_ld, x_ld = psych_vectors(df_ld)
y_ln, n_ln, ny_ln, x_ln = psych_vectors(df_ln)
y_rd, n_rd, ny_rd, x_rd = psych_vectors(df_rd)
y_rn, n_rn, ny_rn, x_rn = psych_vectors(df_rn)

#######
# Analysis
#######
  
gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld = paramest(n_ld,ny_ld)
gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln = paramest(n_ln,ny_ln)
gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd = paramest(n_rd,ny_rd)
gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn = paramest(n_rn,ny_rn)


#######
# Compute PSE and JND
#######
def solve_phi_for_X(gamma, lambda_, beta0, beta1, p):
    # Calculate X using the given formula
    X = -( (beta0 - np.log((gamma - p) / (-1 + lambda_ + p))) / beta1 )
    return X

PSE_ld = solve_phi_for_X(gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld, 0.5)
PSE_ln = solve_phi_for_X(gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln, 0.5)
PSE_rd = solve_phi_for_X(gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd, 0.5)
PSE_rn = solve_phi_for_X(gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn, 0.5)

x25_ld = solve_phi_for_X(gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld, 0.25)
x25_ln = solve_phi_for_X(gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln, 0.25)
x25_rd = solve_phi_for_X(gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd, 0.25)
x25_rn = solve_phi_for_X(gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn, 0.25)

x75_ld = solve_phi_for_X(gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld, 0.75)
x75_ln = solve_phi_for_X(gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln, 0.75)
x75_rd = solve_phi_for_X(gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd, 0.75)
x75_rn = solve_phi_for_X(gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn, 0.75)

JND_ld = 0.5*(x75_ld-x25_ld)
JND_ln = 0.5*(x75_ln-x25_ln)
JND_rd = 0.5*(x75_rd-x25_rd)
JND_rn = 0.5*(x75_rn-x25_rn)

#######
# Plots
#######
xfit = np.linspace(6,50,1000)

yrec_ld = phi_with_lapses([gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld],xfit)
yrec_ln = phi_with_lapses([gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln],xfit)
yrec_rd = phi_with_lapses([gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd],xfit)
yrec_rn = phi_with_lapses([gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn],xfit)

plt.plot(xfit,yrec_ld,label='Distracted' ,color='blue')
plt.scatter(x,y_ld, color = 'blue')
plt.plot(xfit,yrec_ln,label='Undistracted',color='green')
plt.scatter(x,y_ln, color = 'green')
plt.vlines(PSE_ld, 0, 0.5, color='blue', label='PSE = '+ str(np.round(PSE_ld,2)), linestyles='dashed')
plt.vlines(PSE_ln, 0, 0.5, color='green', label='PSE = '+ str(np.round(PSE_ln,2)), linestyles='dashed')
plt.xlabel('Amplitude of Stimulus')
plt.title('Left Hand: Fitted with Lapses')
plt.ylabel('Probability of Guess "High"')
plt.legend()
plt.show()    


plt.plot(xfit,yrec_rd,label='Distracted',color='blue')
plt.scatter(x,y_rd, color = 'blue')
plt.plot(xfit,yrec_rn,label='Undistracted',color='green')
plt.scatter(x,y_rn, color = 'green')
plt.vlines(PSE_rd, 0, 0.5, color='blue', label='PSE = '+ str(np.round(PSE_rd,2)), linestyles='dashed')
plt.vlines(PSE_rn, 0, 0.5, color='green', label='PSE = '+ str(np.round(PSE_rn,2)), linestyles='dashed')
plt.xlabel('Amplitude of Stimulus')
plt.title('Right Hand: Fitted with Lapses')
plt.ylabel('Probability of Guess "High"')
plt.legend()
plt.show()    
    
    
