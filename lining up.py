# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 19:32:48 2025

@author: schro
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter1d


with open("Sirius_DFS.pkl", "rb") as f:
    Sirius_DFS = pickle.load(f)

sessions = list(Sirius_DFS.keys())

def reward_levels(df, thresh=0.03, plot_steps=True):
    corr = (df['rewardDURATION'] > 0.3) & (df['correct'] == True)

    # Create dictionary and ensure proper conversion to DataFrame
    rew = {
        'time': pd.to_timedelta(df.loc[corr, 'start_time'], unit='s'),
        'index': df.loc[corr, 'index'].values,
        'reward_dur': df.loc[corr, 'rewardDURATION'].values
    }
    rew_df = pd.DataFrame(rew)


    # Ensure the DataFrame is a copy before modifying
    rew_df = rew_df.copy()

    # Final computation of reward level groups
    rew_df["reward_diff"] = rew_df["reward_dur"].diff().fillna(0)
    rew_df['group'] = (rew_df['reward_diff'] > thresh).cumsum()

    # Compute mean reward duration for each group and assign to `rew_level`
    rew_df['rew_level'] = rew_df.groupby('group')['reward_dur'].transform('mean')
    boost_indices = rew_df.groupby("group").head(1).index.tolist()
    levels = rew_df.loc[boost_indices]['rew_level'].tolist()

   
    # Optional plotting
    if plot_steps:
        plt.scatter(rew_df['index'], rew_df['reward_dur'])
        plt.plot(rew_df['index'], rew_df['rew_level'])
        plt.xlabel("Trial Index")
        plt.ylabel("Reward Level")
        plt.title(session)
        plt.show()

    return np.array([boost_indices, levels])

boost_info = {}
offsets = np.arange(-20,21)
rows = []
manual = True

for session in sessions:
    df = Sirius_DFS[session]
    boosts, levels = reward_levels(df)
    boost_info[session] = [boosts, levels]
    
        
bad = []      
for session in sessions:
    lst = boost_info[session][0]
    lst.sort()  # Sort the list first
    safe = True
    for i in range(len(lst) - 1):
        if lst[i + 1] - lst[i] < 10:
            safe = False
    if safe == False:
        bad.append(session)
        

if manual == True:                          
    boost_info['03-31-S1'] = [[  2.,  58., 185., 381., 541.] , [0.32878261, 0.32924   , 0.42881905, 0.53027273 , 0.62718182]]
    
    boost_info['04-10-S1'] = [[  5., 235.] , [0.32865116,  0.42801961]]
    
    boost_info['04-24-S1'] = [ [  0.,467.],[0.32946988,  0.42866176] ]
    
    boost_info['05-03-S1'] = [[  0., 152.,  191., 316., 349., 457., 498.] , [0.3297191 ,  0.37972   , 0.42847692, 0.47933333,0.52901818, 0.5806    , 0.631125  ]]
    
    boost_info['05-04-S1'] = [[  2., 152., 193., 310., 340., 423] , [0.32915054, 0.38065217, 0.42943939, 0.48188235, 0.52915556, 0.55125 ]]
    
    boost_info['05-09-S1'] = [[  1., 157., 199., 341., 395., 492., 522.], [0.32870588, 0.37743478, 0.42813846, 0.4708, 0.52843137, 0.5504, 0.6274]]
    
    boost_info['05-12-S1'] = [[  1., 167., 201., 317., 363., 459., 514.] , [0.33025   , 0.38070588, 0.44616667, 0.4811    , 0.52941304, 0.578   , 0.63022727]]
    
    boost_info['05-18-S1'] = [[  0., 159., 189., 278., 318., 404., 467.] , [0.330125  , 0.378625  , 0.42856364, 0.47735294, 0.52851111, 0.580625  , 0.63064286]]
    
    boost_info['05-19-S1'] = [[  6., 165., 195., 318., 342., 415., 482.] , [0.3288427 , 0.3805    , 0.428     , 0.4776    , 0.52914634, 0.5783,  0.63033333 ]]
    
    boost_info['05-26-S1'] = [[  4., 161., 190., 281., 303., 371., 440] , [0.3284382 , 0.37738462, 0.42714545, 0.477875  ,0.5287561 , 0.57953333, 0.62654545]]
    
    boost_info['05-30-S1'] = [[  1., 160., 191., 296., 328., 392., 461.] , [0.32854444, 0.38031818, 0.42926786, 0.47963158,  0.535125  , 0.5774    , 0.62786667]]
    
    boost_info['05-31-S1'] = [[  1., 171., 201., 299., 325., 394., 460.] , [0.32863218, 0.37721429, 0.42958824, 0.47795   , 0.52920513, 0.58078788 , 0.62989474]]
    
    boost_info['06-14-S1'] = [[  0., 157., 198., 290., 347., 415., 488.] , [0.32725843, 0.38052632, 0.42922917, 0.475125 , 0.52984211, 0.5812    , 0.62782353]]
    
    boost_info['06-23-S1'] = [[  1., 174., 313., 370., 438., 506., 582., 612.] , [0.38008197,0.42915094, 0.48077419, 0.529     ,  0.58013636, 0.6388   , 0.6665, 0.73222222]]
    
    boost_info['07-11-S1'] = [[  0., 142., 183., 288., 327., 412., 469.] , [0.32806977, 0.3758125 , 0.42746296, 0.47707407, 0.52769231, 0.57688889  ,0.62808   ]]
    
    boost_info['07-24-S1'] = [[  0., 113., 165., 255., 309., 392., 443.] , [ 0.334     , 0.379     , 0.4279434 , 0.47707143, 0.53066667, 0.57943333, 0.62873913]]
else:
    sessions.remove(bad)
    
    
for session in sessions:
    df = Sirius_DFS[session]
    for boost in boosts[1:]:
        indices = np.arange(boost-20, boost+21)
        row = df.reindex(indices)['correct'].tolist()
        rows.append(row)
rows = np.array(rows)        


means  = (np.nanmean(rows, axis=0))

y_smooth = gaussian_filter1d(means, sigma=3)
plt.scatter(offsets,means,label="Avg Trial Accuracy, n=215")
plt.plot(offsets,means,color='b', linestyle='dotted', linewidth=1)
plt.plot(offsets,y_smooth,label="Smoothened w Gaussian filter")

plt.axhline(y=np.mean(means), color='r', linewidth=1, label="Overall mean accuracy")

# Add a vertical dotted line at x = 5
plt.xlabel("Offset from boost")
plt.ylabel("Accuracy")
plt.axvline(x=0, color='gray', linewidth=1)
plt.title('Avg Accuracy Before and After Reward Bumps')
plt.legend()
plt.show()


