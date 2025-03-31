# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 20:40:17 2025

@author: schro
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d


with open("Sirius_DFS_nodrops.pkl", "rb") as f:
    DFS_nodrops = pickle.load(f)

with open("Sirius_DFS.pkl", "rb") as f:
    Sirius_DFS = pickle.load(f)
    
sessions = list(Sirius_DFS.keys())

def breaks(df):
    indices = df.index[df['outcomeDescription'] == 'eye-abort'].tolist()
    indices = np.array(indices)
    diffs = np.diff(indices)
    breaks = np.where(diffs != 1)[0] + 1  # Indices where sequences break
    
    # Get start indices and lengths
    starts = np.insert(indices[breaks], 0, indices[0])
    lengths = np.diff(np.insert(breaks, 0, 0))
    break_data = np.array(list(zip(starts, lengths)))
    return break_data

import numpy as np

def remove_repeats(arr, keep="first"):
    
    unique_seen = {}  # Dictionary to track occurrences
    
    # Flatten array while keeping original shape for restoration
    flat_arr = arr.flatten()
    
    # Iterate based on the order specified
    indices = range(len(flat_arr)) if keep == "first" else range(len(flat_arr) - 1, -1, -1)
    
    for i in indices:
        val = flat_arr[i]
        if val in unique_seen:
            flat_arr[i] = -1  # Replace duplicate with -1
        else:
            unique_seen[val] = i  # Store first or last occurrence

    return flat_arr.reshape(arr.shape)  # Restore original shape




#try breaklength> 1,2,3
def trials_before_after(df, breaklength = 3 ,wind_rad = 5):
    break_data = breaks(df)
    longbreaks = break_data[break_data[:,1]>=breaklength]
    
    #get indices before and after each break
    ind_lists = []
    for brk in longbreaks:
        ind_before = np.arange(brk[0]-wind_rad,brk[0])
        ind_after = np.arange(brk[0]+brk[1],brk[0]+brk[1]+wind_rad)
        
        ind_list = np.append(ind_before,ind_after).reshape(1, -1)
        ind_lists.append(ind_list)
        

    if ind_lists:  # Only stack if there are valid lists
        ind_lists = np.vstack(ind_lists)
    else:
        ind_lists = np.array([])  # Return an empty array if no valid data
    
    
    
    left, right = np.hsplit(ind_lists,2)
    left = np.array(remove_repeats(left, keep = 'first'))
    right = np.array(remove_repeats(right, keep = 'last'))
    newlists = np.hstack([left,right])
        
    
    return newlists

def trial_results(df_nd, df, brk_len = 3 ,win = 5):
    ind_lists = trials_before_after(df_nd, breaklength = brk_len ,wind_rad = win)
    rows = []
    for ind_list in ind_lists:
        row = df.reindex(ind_list)['correct'].tolist()
        rows.append(row)
    rows = np.array(rows)
    return rows    

def overall_avg(Sirius_DFS):
    trials = 0
    numcorr = 0
    for session in sessions:
        df = Sirius_DFS[session]
        corrs = np.array(df['correct'])
        trials += len(corrs)
        numcorr += sum(corrs == True)
    return numcorr/trials



def avg_acc(DFS_nodrops,Sirius_DFS,brk_len = 3 ,win = 5,plot_on=True):
    rows = np.empty((0, 2 * win), dtype=int)
    offsets = np.delete(np.arange(-win, win + 1), win)
    for session in sessions:
        df_nd = DFS_nodrops[session]
        df = Sirius_DFS[session]
        outcomes = trial_results(df_nd, df, brk_len = brk_len ,win = win)
        if outcomes.shape[0]!=0:
            rows = np.vstack((rows, outcomes))
    rows = np.array(rows)
    means = np.nanmean(rows, axis=0)
    counts = np.sum(~np.isnan(rows), axis=0)
    num_breaks = rows.shape[0]
    min_counts = min(counts)
    
    if plot_on:
        plt.scatter(offsets,means,label="Avg Trial Accuracy, n>"+str(min_counts))
        plt.plot(offsets,means,color='b', linestyle='dotted', linewidth=1)
        y_smooth = gaussian_filter1d(means, sigma=1)
        plt.plot(offsets,y_smooth,label="Smoothened w Gaussian filter")
        plt.axhline(y=overall_avg(Sirius_DFS), color='r', linewidth=1, label="Overall mean accuracy")

        # Add a vertical dotted line at x = 0
        plt.xlabel("Offset from Break")
        plt.ylabel("Accuracy")
        plt.axvline(x=0, color='gray', linewidth=1, label = f'Break of {brk_len} or more conscutive aborted trials')
        plt.title(f'Avg Accuracy Before and After "Breaks", n = {num_breaks}')
        plt.legend()
        plt.show()
    
    
    return [offsets, means, counts]

for leng in [1,2,3,4]:
    avg_acc(DFS_nodrops,Sirius_DFS,brk_len = leng, win = 10)


    
    







