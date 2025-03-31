# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:12:49 2025

@author: schro
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


drop_abort_trials = False
drop_freq_manip = False
drop_repeats = False

directory_path = "Sirius_data"
sessions = []
dates = []

Sirius_DFS = {}

for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
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
        df['rewardDURATION'] = df['correct']*(df['trialEnd'] - df['rewardStart'])
        df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
        
        
        Sirius_DFS[session] = df
        
with open("Sirius_DFS_nodrops.pkl","wb") as f:
    pickle.dump(Sirius_DFS, f)
with open("Sirius_DFS_nodrops.pkl", "rb") as f:
    testout = pickle.load(f)
