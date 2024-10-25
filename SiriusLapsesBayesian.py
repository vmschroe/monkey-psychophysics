#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:48:50 2024

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

##

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

# Define the psychometric function with lapses
def psychometric_func_with_lapses(params, X):
    X = np.array(X)
    gamma, lambda_, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gamma + (1 - gamma - lambda_) * logistic

#construct vectors
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
    return y, n, x

def likelihood_fxn(params, y, n, x):
    L = 1
    for i in range(len(x)):
        L = L *binom.pmf(y[i]*n[i], n[i], psychometric_func_with_lapses(params, x[i]))
    return L

def estparams_maxl(y,n,x,param_guess):
    y_data = y
    n_data = n
    x_data = x
    gamma_guess, lambda_guess, beta0_guess, beta1_guess = param_guess
    def neg_l(params, y, n, x):
        return - likelihood_fxn(params, y, n, x)
    negfixed = partial(neg_l, y=y_data, n=n_data, x=x_data)
    result = minimize(negfixed, param_guess)
    return result.x