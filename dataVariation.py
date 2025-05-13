#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:50:43 2025

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

with open("Sirius_DFS_nodrops.pkl", "rb") as f:
    datadict = pickle.load(f)
sessdict = {}


def extract_stim_dist_FR(row):
    if 'right' in row['stimSIDE']:
        return pd.Series({
            'stimFR': row['tact2_Right_FR'],
            'distFR': row['tact2_Left_FR']
        })
    elif 'left' in row['stimSIDE']:
        return pd.Series({
            'stimFR': row['tact2_Left_FR'],
            'distFR': row['tact2_Right_FR']
        })
    else:
        return pd.Series({'stimFR': None, 'distFR': None})


for sess in datadict.keys():
    
    df = datadict[sess]
    df = df[df['outcomeID']!=14]
    df = df[df['outcomeID']!=140]
    
    
    condlist = sorted(df['trialCond'].unique())
    
    condsum = []
    criteria = ['tact1_Left_FR', 'tact1_Right_FR', 'tact2_Left_FR', 'tact2_Right_FR',
                'distracted', 'stimAMP', 'distAMP', 'stimSIDE']
    
    for cond in condlist:
        row = {'trialCond': cond}
        df_cond = df[df['trialCond'] == cond]
        for crit in criteria:
            row[crit] = df_cond[crit].unique().tolist()
        condsum.append(row)
    
    condsum_df = pd.DataFrame(condsum)
    
    
    condsum_df[['stimFR', 'distFR']] = condsum_df.apply(extract_stim_dist_FR, axis=1)
    condsum_df.drop(['tact1_Left_FR', 'tact1_Right_FR', 'tact2_Left_FR', 'tact2_Right_FR'], axis=1)
    
    distAMPS = sorted({item for sublist in condsum_df['distAMP'] for item in sublist})
    distFRS = sorted({item for sublist in condsum_df['distFR'] for item in sublist})
    stimFRS = sorted({item for sublist in condsum_df['stimFR'] for item in sublist})
    
    distAMPS = [round(x) for x in distAMPS if x]
    distFRS = [x for x in distFRS if x]
    stimFRS = [x for x in stimFRS if x]
    
    sessdict[sess] = [distAMPS,distFRS,stimFRS]

