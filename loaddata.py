#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:02:29 2024

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
# Bringing in Data
#######

#load in data into dataframe
directory_path = '/home/vmschroe/Documents/Monkey Analysis/Sirius_data(1)'
drop_abort_trials = True
drop_freq_manip = True
drop_repeats = True


df = pd.DataFrame()
# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        
        # Read the Excel file into a temporary DataFrame
        dftemp = pd.read_excel(file_path, engine='openpyxl')
        
        # Concatenate the temporary DataFrame with the main DataFrame
        df = pd.concat([df, dftemp], axis=0)

if drop_abort_trials == True:
    # drop aborted trials
    df = df[df['outcomeID'].isin([10, 31, 32, 33, 34])]

if drop_freq_manip == True:
    # drop trials with frequency manipulation
    df = df[df['tact2_Left_FR'].isin([0, 200])]
    df = df[df['tact2_Right_FR'].isin([0, 200])]
    
if drop_repeats == True:
    # drop repeated trials
    df = df[(df['repeatedTrial'] == False)]



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
