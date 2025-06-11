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



