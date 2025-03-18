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
import scipy.interpolate as interpolate
import scipy.stats as stats

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


def reward_levels(df, thresh=0.03, plot_steps=False, plot_title = "Computed Reward Levels"):
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
        plt.scatter(df[(df['correct'] == True)]['index'], df[(df['correct'] == True)]['rewardDURATION'], color = 'green', alpha  = 0.5, label = 'Reward Duration')
        plt.plot(rew_df['index'], rew_df['rew_level'], color = 'blue', label = 'Smoothed Reward Level')
        plt.xlabel("Trial Index")
        plt.ylabel("Reward Duration")
        plt.title(plot_title)
        plt.legend()
        if saveimg == True:
            plt.savefig(plot_title+".png", dpi=300)
        plt.show()

    return rew_df


def slidplots(plotlabel, time_win='100s', **dfs):
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
        plt.plot(df['start_time'], df['Moving_Avg'], label=f'{label} (time window = {time_win})', color=colors[i], linewidth=2)

        # Process reward duration for the first dataset only
        if i == 0:
            rew_dur = reward_levels(df)
            df = df.merge(rew_dur[['index', 'rew_level']], on='index', how='left')
            df['rew_level'] = df['rew_level'].ffill().bfill()
            plt.plot(df['start_time'], df['rew_level'], label='Reward Duration (s)', linewidth=1, linestyle='--')

        modified_dfs[var_name] = df  # Store modified dataframe

    # Adjust plot settings
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Reward Duration (s) / Sliding P(correct)')
    plt.title(f'{plotlabel}')
    if saveimg == True:
        plt.savefig(plotlabel+".png", dpi=300)
    plt.show()
    return modified_dfs if len(modified_dfs) > 1 else next(iter(modified_dfs.values()), None)
    # Return a single dataframe if there was only one input, otherwise return a dictionary
    

def cross_corr(values1, values2, indices1=None, indices2=None, is_time_series=True,
               values1_name='Series 1', values2_name='Series 2', main_title=None, 
               causal_analysis=False, causal_direction='values1_to_values2'):
    """
    Robust correlation analysis with support for both time series and trial indices.
    Supports both boolean and float arrays, and handles NaN values.
    
    Parameters:
    -----------
    values1 : array-like
        Values for first series (boolean or float)
    values2 : array-like
        Values for second series (boolean or float)
    indices1 : array-like, optional
        Indices/timestamps for first series. If None, defaults to [1, 2, 3, ...].
    indices2 : array-like, optional
        Indices/timestamps for second series. If None, indices1 will be used.
    is_time_series : bool, default=True
        If True, treats indices as timestamps and performs interpolation
        If False, treats indices as trial/experiment identifiers and aligns by index
    values1_name : str, default='Series 1'
        Name of the first value series for plotting
    values2_name : str, default='Series 2'
        Name of the second value series for plotting
    main_title : str, optional
        Main title for the entire figure. If None, a default title will be generated.
    causal_analysis : bool, default=False
        If True, focuses analysis on the specified causal direction
    causal_direction : str, default='values1_to_values2'
        Specifies the direction of causality to analyze:
        - 'values1_to_values2': values1 causes values2 (analyze negative lags)
        - 'values2_to_values1': values2 causes values1 (analyze positive lags)
    
    Returns:
    --------
    dict : Correlation analysis results
    """
    
    # Assign default indices if indices1 is not provided
    if indices1 is None:
        indices1 = np.arange(1, len(values1) + 1)
    
    # If indices2 is not provided, use indices1
    if indices2 is None:
        indices2 = indices1
        shared_indices = True
    else:
        shared_indices = False
    
    # Ensure inputs are numpy arrays
    val1 = np.array(values1, dtype=float)
    val2 = np.array(values2, dtype=float)
    idx1 = np.array(indices1)
    idx2 = np.array(indices2)
    
    # Check indices and values have matching lengths
    if len(idx1) != len(val1):
        error_msg = f"{values1_name}: Length mismatch between indices ({len(idx1)}) and values ({len(val1)})"
        print(error_msg)
        return {
            'correlation': None,
            'p_value': None,
            'error': error_msg
        }
    
    if len(idx2) != len(val2):
        error_msg = f"{values2_name}: Length mismatch between indices ({len(idx2)}) and values ({len(val2)})"
        print(error_msg)
        return {
            'correlation': None,
            'p_value': None,
            'error': error_msg
        }
    
    # Handle NaN values in the first series
    nan_mask1 = np.isnan(val1)
    if np.any(nan_mask1):
        nan_count1 = np.sum(nan_mask1)
        print(f"Removing {nan_count1} NaN values from {values1_name}")
        
        # Keep only non-NaN values and their corresponding indices
        val1 = val1[~nan_mask1]
        idx1 = idx1[~nan_mask1]
        
        if len(val1) == 0:
            error_msg = f"All values in {values1_name} are NaN"
            print(error_msg)
            return {
                'correlation': None,
                'p_value': None,
                'error': error_msg
            }
    
    # Handle NaN values in the second series
    nan_mask2 = np.isnan(val2)
    if np.any(nan_mask2):
        nan_count2 = np.sum(nan_mask2)
        print(f"Removing {nan_count2} NaN values from {values2_name}")
        
        # Keep only non-NaN values and their corresponding indices
        val2 = val2[~nan_mask2]
        idx2 = idx2[~nan_mask2]
        
        if len(val2) == 0:
            error_msg = f"All values in {values2_name} are NaN"
            print(error_msg)
            return {
                'correlation': None,
                'p_value': None,
                'error': error_msg
            }
    
    # Check if arrays are boolean and convert appropriately
    if isinstance(values1, (list, np.ndarray)) and np.issubdtype(np.array(values1).dtype, np.bool_):
        series1_type = 'boolean'
    else:
        series1_type = 'float'
        
    if isinstance(values2, (list, np.ndarray)) and np.issubdtype(np.array(values2).dtype, np.bool_):
        series2_type = 'boolean'
    else:
        series2_type = 'float'
    
    # Check for constant or zero-variation series
    if np.std(val1) == 0:
        print(f"Warning: {values1_name} has no variation!")
        return {
            'correlation': None,
            'p_value': None,
            'error': f'Constant {values1_name}'
        }
    
    if np.std(val2) == 0:
        print(f"Warning: {values2_name} has no variation!")
        return {
            'correlation': None,
            'p_value': None,
            'error': f'Constant {values2_name}'
        }
    try:
        # Different processing based on whether data is time series or trial indices
        if is_time_series:
            # Time series approach (with interpolation)
            # Create a common time grid
            min_time = max(idx1.min(), idx2.min())
            max_time = min(idx1.max(), idx2.max())
            
            # Handle case with insufficient time range
            if min_time >= max_time:
                print("Warning: Insufficient time overlap!")
                return {
                    'correlation': None,
                    'p_value': None,
                    'error': 'Insufficient time overlap'
                }
            
            # Calculate the average spacing in the original time indices for lag conversion
            if len(idx1) > 1:
                time_spacing = (idx1[-1] - idx1[0]) / (len(idx1) - 1)
            else:
                time_spacing = 1.0
                
            # Number of points for interpolation
            n_points = 200
            
            # Interpolate series
            common_indices = np.linspace(min_time, max_time, n_points)
            
            # Choose interpolation methods based on data type
            kind1 = 'nearest' if series1_type == 'boolean' else 'linear'
            kind2 = 'nearest' if series2_type == 'boolean' else 'linear'
            
            # Interpolation with error handling
            interp_func1 = interpolate.interp1d(idx1, val1, kind=kind1, fill_value='extrapolate')
            interp_func2 = interpolate.interp1d(idx2, val2, kind=kind2, fill_value='extrapolate')
            
            interpolated_series1 = interp_func1(common_indices)
            interpolated_series2 = interp_func2(common_indices)
            
            plot_title_prefix = 'Time Series'
            x_label = 'Time'
            
        else:
            # Trial indices approach (align by shared indices)
            # Find common indices
            common_indices = np.intersect1d(idx1, idx2)
            
            if len(common_indices) == 0:
                print("Warning: No common trial indices found!")
                return {
                    'correlation': None,
                    'p_value': None,
                    'error': 'No common trial indices'
                }
            
            # Extract values at common indices
            interpolated_series1 = np.array([val1[np.where(idx1 == idx)[0][0]] for idx in common_indices])
            interpolated_series2 = np.array([val2[np.where(idx2 == idx)[0][0]] for idx in common_indices])
            
            # Time spacing based on average interval between common indices
            if len(common_indices) > 1:
                time_spacing = (common_indices[-1] - common_indices[0]) / (len(common_indices) - 1)
            else:
                time_spacing = 1.0
                
            plot_title_prefix = 'Trial Indices'
            x_label = 'Trial Index'
        
        # Choose correlation method based on data types
        if series1_type == 'boolean' and series2_type == 'boolean':
            # Both boolean - use phi coefficient (equivalent to Pearson for binary)
            pearson_corr, p_value = stats.pearsonr(interpolated_series1, interpolated_series2)
            correlation_type = 'phi_coefficient'
        elif series1_type == 'boolean' or series2_type == 'boolean':
            # One boolean, one continuous - use point-biserial
            if series1_type == 'boolean':
                # Convert float values to boolean for point-biserial
                bool_series = (interpolated_series1 > 0.5)
                numeric_series = interpolated_series2
            else:
                bool_series = (interpolated_series2 > 0.5)
                numeric_series = interpolated_series1
            
            # Calculate point-biserial correlation
            pearson_corr, p_value = stats.pointbiserialr(bool_series, numeric_series)
            correlation_type = 'point_biserial'
        else:
            # Both continuous - use Pearson
            pearson_corr, p_value = stats.pearsonr(interpolated_series1, interpolated_series2)
            correlation_type = 'pearson'
        
        # Calculate cross-correlation if dealing with time series
        if is_time_series:
            # Safe cross-correlation calculation
            centered_series1 = interpolated_series1 - interpolated_series1.mean()
            centered_series2 = interpolated_series2 - interpolated_series2.mean()
            
            # Compute cross-correlation
            cross_corr = np.correlate(centered_series1, centered_series2, mode='full')
            
            # Normalize cross-correlation safely
            std1 = np.std(interpolated_series1)
            std2 = np.std(interpolated_series2)
            
            if std1 > 0 and std2 > 0:
                cross_corr = cross_corr / (std1 * std2 * len(interpolated_series1))
            
                # Find max correlation by magnitude (positive or negative)
                max_corr_index = np.argmax(np.abs(cross_corr))
                max_correlation = cross_corr[max_corr_index]
                max_abs_correlation = np.abs(max_correlation)
                lag_index = max_corr_index - len(interpolated_series1) + 1
                
                # Convert lag index to original time units
                # Calculate time span and step size in the common indices
                time_span = max_time - min_time
                time_step = time_span / (n_points - 1)
                
                # Convert lag index to time units
                lag = lag_index * time_step
                
                # Also find max positive and min negative for comprehensive reporting
                max_pos_idx = np.argmax(cross_corr)
                max_pos_corr = cross_corr[max_pos_idx]
                lag_pos_index = max_pos_idx - len(interpolated_series1) + 1
                lag_pos = lag_pos_index * time_step
                
                min_neg_idx = np.argmin(cross_corr)
                min_neg_corr = cross_corr[min_neg_idx]
                lag_neg_index = min_neg_idx - len(interpolated_series1) + 1
                lag_neg = lag_neg_index * time_step
                
                # Create lag time array for plotting in original time units
                lag_indices = np.arange(-len(interpolated_series1) + 1, len(interpolated_series1))
                lag_times = lag_indices * time_step
            else:
                max_correlation = None
                max_abs_correlation = None
                lag = None
                lag_index = None
                max_pos_corr = None
                lag_pos = None
                lag_pos_index = None
                min_neg_corr = None
                lag_neg = None
                lag_neg_index = None
                lag_times = None
        else:
            # For trial indices, we don't calculate lag-based cross-correlation
            max_correlation = None
            max_abs_correlation = None
            lag = None
            lag_index = None
            max_pos_corr = None
            lag_pos = None
            lag_pos_index = None
            min_neg_corr = None
            lag_neg = None
            lag_neg_index = None
            cross_corr = None
            lag_times = None
        
        # Create default main title if none provided
        if main_title is None:
            if is_time_series:
                main_title = f"Correlation Analysis: {values1_name} vs {values2_name} (Time Series)"
            else:
                main_title = f"Correlation Analysis: {values1_name} vs {values2_name} (Trial Indices)"
        
        # Visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Add main title for the entire figure
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.85)  # Adjust top to make room for the title
        
        # Original Series
        plt.subplot(1, 3, 1)
        plt.scatter(idx1, val1, label=f'{values1_name} ({series1_type})', alpha=0.7)
        plt.scatter(idx2, val2, label=f'{values2_name} ({series2_type})', alpha=0.7)
        plt.title(f'Original {plot_title_prefix}')
        plt.xlabel(x_label)
        plt.ylabel('Value')
        plt.legend()
        
        # Interpolated/Aligned Series
        plt.subplot(1, 3, 2)
        if is_time_series:
            plt.plot(common_indices, interpolated_series1, label=f'{values1_name} (n={len(val1)})')
            plt.plot(common_indices, interpolated_series2, label=f'{values2_name} (n={len(val2)})')
            plt.title('Interpolated Series')
        else:
            plt.plot(common_indices, interpolated_series1, 'o-', label=f'{values1_name} ({series1_type})')
            plt.plot(common_indices, interpolated_series2, 'o-', label=f'{values2_name} ({series2_type})')
            plt.title('Aligned Series')
        plt.xlabel(x_label)
        plt.ylabel('Value')
        plt.legend()
        
        # Cross-Correlation (only for time series)
        plt.subplot(1, 3, 3)
        if is_time_series and cross_corr is not None and lag_times is not None:
            if causal_analysis:
                # For causal analysis, only show relevant lags based on direction
                if causal_direction == 'values1_to_values2':
                    # values1 causes values2, so we look at negative lags
                    relevant_indices = lag_times <= 0
                    plt.plot(lag_times[relevant_indices], cross_corr[relevant_indices])
                    title_direction = f"{values1_name} → {values2_name}"
                else:
                    # values2 causes values1, so we look at positive lags
                    relevant_indices = lag_times >= 0
                    plt.plot(lag_times[relevant_indices], cross_corr[relevant_indices])
                    title_direction = f"{values2_name} → {values1_name}"
                
                plt.title(f'Causal Analysis: {title_direction}')
            else:
                # Standard bidirectional analysis
                plt.plot(lag_times, cross_corr)
                plt.title('Cross-Correlation')
            
            plt.xlabel('Lag (in original time units)')
            plt.ylabel('Correlation')
            
            # Mark the maximum correlation by magnitude on the plot
            if max_correlation is not None:
                if not causal_analysis or (causal_analysis and 
                   ((causal_direction == 'values1_to_values2' and lag <= 0) or
                    (causal_direction == 'values2_to_values1' and lag >= 0))):
                    plt.plot(lag, max_correlation, 'ro', markersize=8, 
                         label=f'Max magnitude: {max_correlation:.3f} at lag {lag:.3f}')
                    plt.axvline(x=lag, color='r', linestyle='--', alpha=0.5)
                
                # Find max correlation in causal direction
                if causal_analysis:
                    if causal_direction == 'values1_to_values2':
                        # Find max in negative lag region (values1 → values2)
                        causal_indices = np.where(lag_times <= 0)[0]
                    else:
                        # Find max in positive lag region (values2 → values1)
                        causal_indices = np.where(lag_times >= 0)[0]
                    
                    if len(causal_indices) > 0:
                        causal_max_idx = causal_indices[np.argmax(np.abs(cross_corr[causal_indices]))]
                        causal_max_corr = cross_corr[causal_max_idx]
                        causal_lag = lag_times[causal_max_idx]
                        
                        plt.plot(causal_lag, causal_max_corr, 'go', markersize=8, 
                             label=f'Max causal effect: {causal_max_corr:.3f} at lag {causal_lag:.3f}')
                        plt.axvline(x=causal_lag, color='g', linestyle='--', alpha=0.5)
                
                plt.legend()
        else:
            if is_time_series:
                plt.text(0.5, 0.5, 'Cross-correlation unavailable\ndue to zero standard deviation', 
                        horizontalalignment='center', verticalalignment='center')
                plt.title('Cross-Correlation Error')
            else:
                plt.text(0.5, 0.5, 'Cross-correlation not applicable\nfor trial indices', 
                        horizontalalignment='center', verticalalignment='center')
                plt.title('No Cross-Correlation')
        
        plt.tight_layout()
        if saveimg == True:
            plt.savefig(main_title+".png", dpi=300)
        plt.show()
        
        # Prepare return dictionary
        result = {
            'correlation_type': correlation_type,
            'correlation': pearson_corr,
            'p_value': p_value,
            'series1_type': series1_type,
            'series2_type': series2_type,
            'data_type': 'time_series' if is_time_series else 'trial_indices',
            'common_indices_count': len(common_indices),
            'shared_indices': shared_indices,
            'original_length_series1': len(values1),
            'original_length_series2': len(values2),
            'clean_length_series1': len(val1),
            'clean_length_series2': len(val2),
            'nan_count_series1': len(values1) - len(val1) if isinstance(values1, (list, np.ndarray)) else 0,
            'nan_count_series2': len(values2) - len(val2) if isinstance(values2, (list, np.ndarray)) else 0
        }
        
        # Add cross-correlation results if applicable
        if is_time_series:
            result.update({
                'max_correlation_by_magnitude': max_correlation,
                'max_abs_correlation': max_abs_correlation,
                'lag_at_max_magnitude': lag,
                'lag_index_at_max_magnitude': lag_index,
                'max_positive_correlation': max_pos_corr,
                'lag_at_max_positive': lag_pos,
                'lag_index_at_max_positive': lag_pos_index,
                'min_negative_correlation': min_neg_corr,
                'lag_at_min_negative': lag_neg,
                'lag_index_at_min_negative': lag_neg_index,
                'time_step': time_step if 'time_step' in locals() else None
            })
            
            if causal_analysis:
                if causal_direction == 'values1_to_values2':
                    # Find max in negative lag region (values1 → values2)
                    causal_indices = np.where(lag_times <= 0)[0]
                    delay_description = f"{values1_name} → {values2_name}"
                else:
                    # Find max in positive lag region (values2 → values1)
                    causal_indices = np.where(lag_times >= 0)[0]
                    delay_description = f"{values2_name} → {values1_name}"
                
                if len(causal_indices) > 0:
                    causal_max_idx = causal_indices[np.argmax(np.abs(cross_corr[causal_indices]))]
                    causal_max_corr = cross_corr[causal_max_idx]
                    causal_lag = lag_times[causal_max_idx]
                    
                    result.update({
                        'causal_analysis': True,
                        'causal_direction': causal_direction,
                        'causal_description': delay_description,
                        'max_causal_correlation': causal_max_corr,
                        'causal_lag': causal_lag,
                        'causal_lag_index': causal_max_idx - len(interpolated_series1) + 1
                    })
            
        return result
    
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return {
            'correlation': None,
            'p_value': None,
            'error': str(e)
        }

reload_data = False
all_sessions = True
stim_levels = False
saveimg = True
dep_var = 'Moving_Avg'
# indep_var = 'stimAMP'
indep_var = 'rew_level'


if reload_data:        
    frames, sessions = loaddfs()

if all_sessions:
   sess_list = sessions
else:
    sess_list = ['06-07-S1']

if stim_levels:
    stims_list = [ [6, 12, 18, 24], [18, 24, 32, 38], [32, 38, 44, 50]]
else:
    stims_list = [[6, 12, 18, 24, 32, 38, 44, 50]]


pvals = []
maxcorr = []
lags = []

for stims in stims_list:
    for session in sess_list:
        df = frames['all'][session]
        reward_levels(df, plot_steps = True, plot_title = 'Reward Duration, ' + str(session))
        # dfstim = df[df['stimAMP'].isin(stims)]
        # dfnew = slidplots(session+', Sliding Average Proportion Correct', Avg_Prop_Correct = dfstim)
        # result = cross_corr(np.array(dfnew[indep_var]), np.array(dfnew[dep_var]), indices1=np.array(dfnew['start_time']), values1_name='Reward Level', values2_name='Average Proportion Correct', main_title='Cross-correlation Analysis: Reward Duration vs Sliding Avg Proportion Correct, ' + str(session), causal_analysis=True)
        # pvals = np.append(pvals,result['p_value'])
        # maxcorr = np.append(maxcorr, result['max_causal_correlation'])
        # lags = np.append(lags, result['causal_lag'])