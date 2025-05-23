#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:24:30 2024

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

##Load and clean data
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/loaddata.py').read())

#Construct necessary functions
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/FunctionsForMLE.py').read())


#######
# Analysis
#######

y_ld, n_ld, ny_ld, x_ld = psych_vectors(df_ld)
y_ln, n_ln, ny_ln, x_ln = psych_vectors(df_ln)
y_rd, n_rd, ny_rd, x_rd = psych_vectors(df_rd)
y_rn, n_rn, ny_rn, x_rn = psych_vectors(df_rn)


gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld = paramest(n_ld,ny_ld)
gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln = paramest(n_ln,ny_ln)
gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd = paramest(n_rd,ny_rd)
gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn = paramest(n_rn,ny_rn)


#######
# Compute PSE and JND
#######


PSE_ld = PSE(gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld)
PSE_ln = PSE(gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln)
PSE_rd = PSE(gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd)
PSE_rn = PSE(gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn)

JND_ld = JND(gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld)
JND_ln = JND(gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln)
JND_rd = JND(gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd)
JND_rn = JND(gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn)


#######
# Plots
#######
xfit = np.linspace(6,50,1000)

yrec_ld = phi_with_lapses([gam_est_ld, lam_est_ld, b0_est_ld, b1_est_ld],xfit)
yrec_ln = phi_with_lapses([gam_est_ln, lam_est_ln, b0_est_ln, b1_est_ln],xfit)
yrec_rd = phi_with_lapses([gam_est_rd, lam_est_rd, b0_est_rd, b1_est_rd],xfit)
yrec_rn = phi_with_lapses([gam_est_rn, lam_est_rn, b0_est_rn, b1_est_rn],xfit)

plt.plot(xfit,yrec_ld,label='Distracted' ,color='blue')
plt.scatter(x,y_ld, color = 'blue')
plt.plot(xfit,yrec_ln,label='Undistracted',color='green')
plt.scatter(x,y_ln, color = 'green')
plt.vlines(PSE_ld, 0, 0.5, color='blue', label='PSE = '+ str(np.round(PSE_ld,2)), linestyles='dashed')
plt.vlines(PSE_ln, 0, 0.5, color='green', label='PSE = '+ str(np.round(PSE_ln,2)), linestyles='dashed')
plt.xlabel('Amplitude of Stimulus')
plt.title('Left Hand: Fitted with Lapses')
plt.ylabel('Probability of Guess "High"')
plt.legend()
plt.savefig('plot_left_lapses.png')
plt.show()    


plt.plot(xfit,yrec_rd,label='Distracted',color='blue')
plt.scatter(x,y_rd, color = 'blue')
plt.plot(xfit,yrec_rn,label='Undistracted',color='green')
plt.scatter(x,y_rn, color = 'green')
plt.vlines(PSE_rd, 0, 0.5, color='blue', label='PSE = '+ str(np.round(PSE_rd,2)), linestyles='dashed')
plt.vlines(PSE_rn, 0, 0.5, color='green', label='PSE = '+ str(np.round(PSE_rn,2)), linestyles='dashed')
plt.xlabel('Amplitude of Stimulus')
plt.title('Right Hand: Fitted with Lapses')
plt.ylabel('Probability of Guess "High"')
plt.legend()
plt.savefig('plot_right_lapses.png')
plt.show()    
    
    