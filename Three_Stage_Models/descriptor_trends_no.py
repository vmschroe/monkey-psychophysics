# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 20:29:42 2025

@author: schro
"""

import datetime
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import arviz as az
import matplotlib.pyplot as plt

#looking for trends in descriptors related to weekdays or familiarity with dist level

with open('H3sL_descrip_post_ests.pkl', "rb") as f:
    post_ests = pickle.load(f)  
    
with open("session_summary.pkl", "rb") as f:
    sess_sum = pickle.load(f)  
    
grp = 'ld'
desc = 'PSE'
for desc in descriptors:
    x = []
    y = []
    y_err_min = []
    y_err_max = []
    for grp in ['ld', 'rd']:
        subdf = sess_sum[['NumTrials_'+grp, 'Dist_AMP','weekday']]
        subdf['mean'] = post_ests[grp][desc]['mean']
        subdf['lHDI'] = post_ests[grp][desc]['lHDI']
        subdf['hHDI'] = post_ests[grp][desc]['hHDI']
        subdf['AMP_seen_before'] = subdf.groupby('Dist_AMP').cumcount()
        subdf = subdf[subdf['NumTrials_'+grp]>60]
        
        leng = len(subdf['Dist_AMP'])
        jitter = np.random.uniform(-1, 1, size=leng)
        subdf['AMP_j'] = subdf['AMP_seen_before']+ 0.125*jitter
        
        x.append(np.array(subdf['AMP_j']))
        y.append(np.array(subdf['mean']))
        
        
        y_err_min.append(np.array(subdf['lHDI']))
        y_err_max.append(np.array(subdf['hHDI']))
    
    # Asymmetric error bars
    
    
    x = np.array(x).reshape([1,-1])[0]
    y = np.array(y).reshape([1,-1])[0]
    y_err_min = np.array(y_err_min).reshape([1,-1])[0]
    y_err_max = np.array(y_err_max).reshape([1,-1])[0]
    
    y_err = [y_err_min, y_err_max]
    
    #initiate linear regression model
    model = LinearRegression()

    #define predictor and response variables
    X = [subdf['AMP_seen_before'],subdf['AMP_seen_before']]
    X = np.array(X).reshape([-1,1])

    #fit regression model
    model.fit(X, y)

    #calculate R-squared of regression model
    r_squared = model.score(X, y)
    
    
    plt.scatter(x, y, alpha = 0.6)
    plt.title(grp+' '+desc)
    plt.show()
    
    print(desc)
    print('intercept = '+ str(model.intercept_))
    print('coef = '+ str(model.coef_))
    print('r^2 = ' + str(r_squared))



with open("/home/vmschroe/Documents/Monkey Analysis/H3sL_traces.pkl", "rb") as f:
    traces = pickle.load(f)  
    
left = 'ld'
right = 'rd'

for param in ['gamma_h', 'gamma_l', 'beta0', 'beta1','PSE','JND']:

#param = 'PSE'
    l_means = np.array(az.summary(traces[left], var_names=param)['mean'])
    r_means = np.array(az.summary(traces[right], var_names=param)['mean'])
    
    lstd = np.std(l_means)
    rstd = np.std(r_means)
    
    hstd = 0.5*(lstd+rstd)
    sstd = 0.5*np.mean(np.abs(l_means-r_means))
    
    
    if hstd<sstd:
        grp_by = "hand"
        rat = hstd/sstd
    else:
        grp_by = "session"
        rat = sstd/hstd
    
    print('------')
    print(param)
    print("Hand deviation: "+ str(hstd))
    print("Session deviation: "+ str(sstd))
    print("For "+param+", group by: "+grp_by)
    print("Confidence: "+ str(1-rat))
    
    x_vals = np.arange(len(l_means))
    
    
    plt.scatter(x_vals, l_means,c='black')
    plt.scatter(x_vals, r_means,c='blue')
    plt.title(left+right+param)
    plt.show()
