#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:44:43 2025

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

frames = {}
frames["ld"] = {}
frames["ln"] = {}
frames["rd"] = {}
frames["rn"] = {}
sessions = []
dates = []
dist_amps = []

Y = {}
Y["ld"] = {}
Y["ln"] = {}
Y["rd"] = {}
Y["rn"] = {}

N = {}
N["ld"] = {}
N["ln"] = {}
N["rd"] = {}
N["rn"] = {}

NY = {}
NY["ld"] = {}
NY["ln"] = {}
NY["rd"] = {}
NY["rn"] = {}
x = [6, 12, 18, 24, 32, 38, 44, 50]


def psych_vectors(df):
    n = []
    y = []
    for amp in x:
        tempn, _ = df[df['stimAMP']==amp].shape
        tempny,_ = df[ (df['stimAMP']==amp)&(df['lowORhighGUESS']==1)].shape
        tempy = tempny/tempn
        n = np.append(n,tempn)
        y = np.append(y,tempy)
        ny = (n * y).astype(int)
    return y, n, ny

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        print("--------------------------------------------------------------")
        print("File Name: ", filename)
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        sess_date = filename[8:13]
        list.append(dates, sess_date)
        sess_num = dates.count(sess_date)
        session = sess_date + "-S" + str(sess_num)
        list.append(sessions,session)
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, engine='openpyxl')
        
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
        # make new column for distraction amplitude
        distAMP = 10*((0.1-df['tact1_Left_DUR'])*df['tact2_Left_AMP']+(0.1-df['tact1_Right_DUR'])*df['tact2_Right_AMP'])
        df['distAMP'] = distAMP
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
        frames['ld'][session] = df[(df['stimSIDE'] == 'left') & (df['distracted'] == True)]
        frames['ln'][session] = df[(df['stimSIDE'] == 'left') & (df['distracted'] == False)]
        frames['rd'][session] = df[(df['stimSIDE'] == 'right') & (df['distracted'] == True)]
        frames['rn'][session] = df[(df['stimSIDE'] == 'right') & (df['distracted'] == False)]
        
        for grp in ['ld', 'ln', 'rd', 'rn']:
            y, n, ny = psych_vectors(frames[grp][session])
            Y[grp][session] = y
            N[grp][session] = n
            NY[grp][session] = ny
        
        #list.append(dist_amps, max(frames[:][session]['distAMP'].unique()))
        
dist_amps = [round(x) for x in dist_amps]
distAMPS = dict(zip(sessions, dist_amps))

NumTrials ={}
# Loop through each session and group
for session in sessions:
    sumtemp = 0
    row = {}
    for grp in ['ld', 'ln', 'rd', 'rn']:
        row[grp] = sum(N[grp][session])
        sumtemp += row[grp]
    row['sum'] = sumtemp
    row['Deviation from Split'] = max(row['ld'], row['ln'], row['rd'], row['rn'])-min(row['ld'], row['ln'], row['rd'], row['rn'])
    
    NumTrials[session] = row  # Store row in dictionary with session as key

# Convert dictionary to a DataFrame
NumTrialsdf = pd.DataFrame.from_dict(NumTrials, orient='index')

# Rename index and columns for clarity
NumTrialsdf.index.name = "Session"
NumTrialsdf.columns = ["ld", "ln", "rd", "rn", "sum", 'Deviation from Split']

# Print table
print(NumTrialsdf)




        
# Create an empty dictionary to store results
DistAmps = {}
# Loop through each session and group
for session in sessions:
    row = {}
    for grp in ['ld', 'ln', 'rd', 'rn']:
        row[grp] = max(frames[grp][session]['distAMP'].unique())
    DistAmps[session] = row  # Store row in dictionary with session as key

# Convert dictionary to a DataFrame
DistAmpsdf = pd.DataFrame.from_dict(DistAmps, orient='index')

# Rename index and columns for clarity
DistAmpsdf.index.name = "Session"
DistAmpsdf.columns = ["ld", "ln", "rd", "rn"]

# Print table
print(DistAmpsdf)