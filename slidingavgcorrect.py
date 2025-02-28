#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:19:29 2025

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
DistAMPS = {}

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
            
        ### rename trial index
        
        df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
        
        
        
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
        
        
        df['rewardDURATION'] = df['correct']*(df['trialEnd'] - df['rewardStart'])
        
        # separate into 4 dataframes based on hand/distraction
        frames['ld'][session] = df[(df['stimSIDE'] == 'left') & (df['distracted'] == True)]
        frames['ln'][session] = df[(df['stimSIDE'] == 'left') & (df['distracted'] == False)]
        frames['rd'][session] = df[(df['stimSIDE'] == 'right') & (df['distracted'] == True)]
        frames['rn'][session] = df[(df['stimSIDE'] == 'right') & (df['distracted'] == False)]
  
def slidplots(df_d, df_nd, win_sizes, plotlabel):
    for win_size in win_sizes:
        #prep reward plotting
        df_corr_d = df_d[df_d['correct']==True]
        df_corr_nd = df_nd[df_nd['correct']==True]
        
        # Compute moving average
        df_d = df_d[df_d['stimAMP'].isin(stims)]
        df_nd = df_nd[df_nd['stimAMP'].isin(stims)]
        df_d['Moving_Avg'] = df_d['correct'].rolling(window=win_size, center=True).mean()
        df_nd['Moving_Avg'] = df_nd['correct'].rolling(window=win_size, center=True).mean()
        
        irange_d = df_d[~np.isnan(df_d['Moving_Avg'])]['index']
        irange_nd = df_nd[~np.isnan(df_nd['Moving_Avg'])]['index']
        ranges = np.append(irange_d,irange_nd)

        # Plot original data and moving average
        plt.plot(df_d['index'],df_d['Moving_Avg'], label='distracted', color = 'red', linewidth=2)
        plt.plot(df_nd['index'],df_nd['Moving_Avg'], label='not distracted', color = 'blue', linewidth=2)
        plt.plot(df_corr_d['index'],np.round(df_corr_d['rewardDURATION'] * 30) / 30, label='reward', color = 'red', linewidth=1, ls='--')
        plt.plot(df_corr_nd['index'], np.round(df_corr_nd['rewardDURATION'] * 30) / 30, color = 'blue', linewidth=1, ls='--')
        plt.ylim(-0.05, 1.05)
        plt.xlim( min(ranges)-2, max(ranges)+2)
        plt.legend()
        plt.xlabel('Trial Index')
        plt.ylabel('Sliding P(correct)')
        plt.title('Window = ' + str(win_size) + ', '+ plotlabel)
        plt.show()
        
stims =[18, 24, 32, 38]
for session in sessions:
    if not (session == '04-14-S2'):
        print(session)
        df_d = frames['ld'][session]
        df_nd = frames['ln'][session]
        slidplots(df_d,df_nd, [10,15,20], 'Left Hand, '+ session + ', stimAMPs = '+str(stims))
  