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
from scipy import signal

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
            dates.append(sess_date)
            sess_num = dates.count(sess_date)
            session = sess_date + "-S" + str(sess_num)
            sessions.append(session)
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
           



def slidplots(plotlabel, time_win='400s', **dfs):
    """
    Plots sliding averages for up to 4 dataframes with optional custom labels.

    Parameters:
        plotlabel (str): Title label for the plot.
        dfs (dict): Named dataframes with optional labels, e.g., 
                    slidplots("Example", distracted=(df1, "Distracted"), non_distracted=(df2, "Not Distracted")).
    Returns:
        If one dataset is provided, returns the modified dataframe.
        If multiple datasets are provided, returns a dictionary of modified dataframes.
    """

    colors = ['red', 'blue', 'green', 'purple']  # Define distinct colors for up to 4 datasets

    plt.figure(figsize=(8, 5))
    modified_dfs = {}  # Dictionary to store modified dataframes

    for i, (var_name, df_info) in enumerate(dfs.items()):
        if i >= 4:
            print("Warning: More than 4 dataframes provided. Only the first 4 will be plotted.")
            break

        # Allow input as either a dataframe or a (dataframe, label) tuple
        if isinstance(df_info, tuple):
            df, label = df_info
        else:
            df, label = df_info, var_name  # Use variable name as fallback label

        df = df.copy()  # Ensure `df` is a copy before modification

        # Convert `start_time` to Timedelta
        df['time'] = pd.to_timedelta(df['start_time'], unit='s')

        # Compute moving average
        df['Moving_Avg'] = df.rolling(time_win, on='time', min_periods=3)['correct'].mean()

        # Plot moving average
        plt.plot(df['time'], df['Moving_Avg'], label=f'{label} (win={time_win})', color=colors[i], linewidth=2)

        # Process reward duration for the first dataset only
        if i == 0:
            rew_dur = reward_levels(df)
            df = df.merge(rew_dur[['index', 'rew_level']], on='index', how='left')
            df['rew_level'] = df['rew_level'].ffill()
            plt.plot(df['time'], df['rew_level'], label='Reward Duration', linewidth=1, linestyle='--')

        modified_dfs[var_name] = df  # Store modified dataframe

    # Adjust plot settings
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Sliding P(correct)')
    plt.title(f'{plotlabel}')
    plt.show()

    # Return a single dataframe if there was only one input, otherwise return a dictionary
    return modified_dfs 
#---if len(dfs) > 1 else list(modified_dfs.values())[0]




def reward_levels(df, thresh=0.03, plot_steps=False):
    corr = (df['rewardDURATION'] > 0.3) & (df['correct'] == True)

    # Create dictionary and ensure proper conversion to DataFrame
    rew = {
        'time': pd.to_timedelta(df.loc[corr, 'start_time'], unit='s'),
        'index': df.loc[corr, 'index'].values,
        'reward_dur': df.loc[corr, 'rewardDURATION'].values
    }
    rew_df = pd.DataFrame(rew)

    # Remove isolated jumps in reward durations
    rew_df = rew_df[
        ~((rew_df['reward_dur'] - rew_df['reward_dur'].shift(1) > 0.05) &
          (rew_df['reward_dur'] - rew_df['reward_dur'].shift(-1) > 0.05))
    ]

    # Ensure the DataFrame is a copy before modifying
    rew_df = rew_df.copy()

    # Filter out small reward changes iteratively
    for _ in range(10):
        rew_df["reward_diff"] = rew_df["reward_dur"].diff().fillna(0)
        rew_df = rew_df.loc[rew_df["reward_diff"] > -0.01].copy()

    # Final computation of reward level groups
    rew_df["reward_diff"] = rew_df["reward_dur"].diff().fillna(0)
    rew_df['group'] = (rew_df['reward_diff'] > thresh).cumsum()

    # Compute mean reward duration for each group and assign to `rew_level`
    rew_df['rew_level'] = rew_df.groupby('group')['reward_dur'].transform('mean')

    # Drop temporary group column
    rew_df = rew_df.drop(columns=['group', 'reward_diff'])

    # Optional plotting
    if plot_steps:
        plt.plot(rew_df['index'], rew_df['rew_level'])
        plt.xlabel("Trial Index")
        plt.ylabel("Reward Level")
        plt.title("Computed Reward Levels")
        plt.show()

    return rew_df



reload = False
if reload:        
    frames, sessions = loaddfs()
session = '06-07-S1'

stims = [6, 12, 18, 24, 32, 38, 44, 50]
df = frames['all'][session]
dfstim = df[df['stimAMP'].isin(stims)]
dfs = slidplots(session+', '+str(stims), df = dfstim)
dfnew = dfs['df']

df = df.merge(dfnew[['index', 'rew_level']], on='index', how='left')
df['rew_level'] = df['rew_level'].ffill().bfill()
print(df)


times = np.array(df['start_time'])
correct = np.array(df['correct'])
rewards = np.array(df['rew_level'])


        
    
    
    