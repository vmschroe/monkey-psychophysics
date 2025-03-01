#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:22:51 2025

@author: vmschroe
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def loaddfs(directory_path = '/home/vmschroe/Documents/Monkey Analysis/Sirius_data(1)', drop_abort_trials = True, drop_freq_manip = True, drop_repeats = True):
    
    frames = {}
    frames["ld"] = {}
    frames["ln"] = {}
    frames["rd"] = {}
    frames["rn"] = {}
    frames["all"] = {}
    sessions = []
    dates = []

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
            frames['all'][session] = df
    return frames, sessions
           

def slidplots(plotlabel, rew_dur, time_win = '900s', **dfs):
    
    
    
    """
    Plots sliding averages for up to 4 dataframes with optional custom labels.

    Parameters:
        plotlabel (str): Title label for the plot.
        
        rew_dur (list of 2 lists): [ trial_indices , reward_durations ]
        dfs (dict): Named dataframes with optional labels, e.g., 
                    slidplots("Example", 10, distracted=(df1, "Distracted"), non_distracted=(df2, "Not Distracted")).
    """

    colors = ['red', 'blue', 'green', 'purple']  # Define distinct colors for up to 4 datasets

    
    plt.figure(figsize=(8, 5))
    ranges = []  # Store valid index ranges

    # Iterate over input dataframes
    for i, (var_name, df_info) in enumerate(dfs.items()):
        if i >= 4:
            print("Warning: More than 4 dataframes provided. Only the first 4 will be plotted.")
            break

        # Allow input as either a dataframe or a (dataframe, label) tuple
        if isinstance(df_info, tuple):
            df, label = df_info
        else:
            df, label = df_info, var_name  # Use variable name as fallback label

       

        # Compute moving average
        # df['Moving_Avg'] = df['correct'].rolling(window=win_size, center=True).mean()
        # Convert start_time (in seconds) to a Timedelta and set as index
        df['time'] = pd.to_timedelta(df['start_time'], unit='s')
        df = df.set_index('time')
        
        # Now apply a rolling window of 100 seconds
        df['Moving_Avg'] = df['correct'].rolling(time_win, min_periods = 3, center=True).mean()

        # Collect index ranges
        irange = df[~np.isnan(df['Moving_Avg'])]['index']
        ranges.extend(irange)

        # Plot moving average
        plt.scatter(df['index'], df['Moving_Avg'], color=colors[i], linewidth=1)
        plt.plot(df['index'], df['Moving_Avg'], label=f'{label} (win={time_win})', color=colors[i], linewidth=2)

    # Plot rewards
    plt.plot(rew_dur[0], rew_dur[1], label = 'reward duration', linewidth=1, ls='--')

    # Adjust plot settings
    if ranges:
        plt.xlim(min(ranges) - 2, max(ranges) + 2)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.xlabel('Trial Index')
    plt.ylabel('Sliding P(correct)')
    plt.title(f'{plotlabel}')
    plt.show()


def reward_levels(df, thresh = 0.03, plot_steps = False):
    corr = (df['rewardDURATION']  >0.3) & (df['correct']  == True)
    
    rew = {}
    rew['index'] = df[corr]['index']
    rew['reward_dur'] = df[corr]['rewardDURATION']
    rew_df = pd.DataFrame.from_dict(rew)
    rew_df = rew_df[~((rew_df['reward_dur'] - rew_df['reward_dur'].shift(1) > 0.05) &
                      (rew_df['reward_dur'] - rew_df['reward_dur'].shift(-1) > 0.05))]
    
    for i in range(10):
        rew_df["reward_diff"] = rew_df["reward_dur"].diff().fillna(0)
        rew_df = rew_df[rew_df["reward_diff"]>-0.01]
        rew_df = rew_df.drop(columns='reward_diff')
    
    rew_df["reward_diff"] = rew_df["reward_dur"].diff().fillna(0)
    
    rew_df['group'] = (rew_df['reward_diff'] > thresh).cumsum()
    
    # Compute the mean reward_dur for each group and assign it to a new column
    rew_df['rew_level'] = rew_df.groupby('group')['reward_dur'].transform('mean')
    
    # (Optional) Remove the temporary group column
    rew_df = rew_df.drop(columns='group')
    
    if plot_steps:
        plt.plot(rew_df['index'],rew_df['rew_level'])
        plt.show()
    return [rew_df['index'],rew_df['rew_level']]


reload = False
if reload:        
    frames, sessions = loaddfs()
session = '06-07-S1'

for stims in [[6, 12, 18, 24],[18, 24, 32, 38],[32, 38, 44, 50], [6, 12, 18, 24, 32, 38, 44, 50]]:
    df = frames['all'][session]
    ld = frames['ld'][session]
    ln = frames['ln'][session]
    rd = frames['rd'][session]
    rn = frames['rn'][session]
    
    ld = ld[ld['stimAMP'].isin(stims)]
    ln = ln[ln['stimAMP'].isin(stims)]
    rd = rd[rd['stimAMP'].isin(stims)]
    rn = rn[rn['stimAMP'].isin(stims)]
    dfstim = df[df['stimAMP'].isin(stims)]
    
    left = pd.concat([ld, ln], axis=0).sort_values('index').reset_index(drop=True)
    right = pd.concat([rd, rn], axis=0).sort_values('index').reset_index(drop=True)
    dist = pd.concat([ld, rd], axis=0).sort_values('index').reset_index(drop=True)
    nodist = pd.concat([ln, rn], axis=0).sort_values('index').reset_index(drop=True)
    
    
    slidplots(session +', Stims: '+ str(stims), reward_levels(df), left_distracted = ld, left_nondistracted = ln,  right_distracted = rd, right_nondistracted = rn)
    slidplots(session +', Stims: '+ str(stims), reward_levels(df), left_hand = left, right_hand = right)
    slidplots(session +', Stims: '+ str(stims), reward_levels(df), distracted = dist, not_distracted = nodist)
    slidplots(session +', Stims: '+ str(stims), reward_levels(df), all_trials = dfstim)
    
    
        
    
    
    