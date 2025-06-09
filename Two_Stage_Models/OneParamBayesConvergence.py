#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:17:02 2024

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
import pymc as pm
import numpy as np
import arviz as az
import sys
sys.path.append("/home/vmschroe/Documents/Monkey Analysis/Github")
import FunctionsForBayes as ffb




plt.scatter(NumTrialsVec,postmeans, label = "Posterior Means",color='blue')
plt.plot(NumTrialsVec,np.full(len(postmeans),sim_params[2]), label = f"Data Simulation b0 = {sim_params[2]}", color="red")
plt.title('b0 posterior means')
plt.xlabel('Number of Trials')
plt.ylabel('Parameter bo')
plt.legend()
plt.show()

b0errs = np.abs(float(sim_params[2]) - np.array(postmeans))
plt.scatter(NumTrialsVec,b0errs,color='blue')
plt.plot(NumTrialsVec,(NumTrialsVec)**(-1/2), label = "abs(error) = n^(-1/2)", color="red")
#plt.plot(NumTrialsVec,(NumTrialsVec)**(-0.3), label = "abs(error) = n^(-0.3)", color="purple")
plt.plot(NumTrialsVec,1/(NumTrialsVec), label = "abs(error) = n^(-1)", color="green")
#plt.plot(NumTrialsVec,np.exp(c)*(NumTrialsVec)**(m), label = "abs(error) = e^c * n^(m)", color="green")
plt.title('n vs error')
plt.xlabel('Number of Trials')
plt.ylabel('abs error in bo')
plt.legend()
plt.show()

# b0errs = np.abs(float(sim_params[2]) - np.array(postmeans))
# plt.plot(NumTrialsVec,(NumTrialsVec)**(-1/2)-b0errs, label = "abs(error) = n^(-1/2)", color="red")
# plt.plot(NumTrialsVec,np.exp(c)*(NumTrialsVec)**(m)-b0errs, label = "abs(error) = e^c * n^(m)", color="green")
# plt.title('Residuals')
# plt.xlabel('Number of Trials')
# plt.ylabel('residuals')
# plt.legend()
# plt.show()


plt.scatter(np.log(NumTrialsVec),2*np.log(b0errs),color='blue')
plt.plot(np.log(NumTrialsVec),-0.5*np.log(NumTrialsVec),label = "log(abs(error)) = -0.5log(abs(n))", color="red")
#plt.plot(np.log(NumTrialsVec),-0.3*np.log(NumTrialsVec),label = "log(abs(error)) = -0.3log(abs(n))", color="purple")
plt.plot(np.log(NumTrialsVec),-np.log(NumTrialsVec),label = "log(abs(error)) = -log(abs(n))", color="green")
# plt.plot(np.log(NumTrialsVec), m*np.log(NumTrialsVec)+c, color="green")
plt.title('log n vs log err')
plt.xlabel('log Number of Trials')
plt.ylabel('log error')
plt.legend()
plt.show()

# # Data
# x = np.array(np.log(NumTrialsVec)[b0errs !=0])
# y = np.array(np.log(b0errs)[b0errs !=0])

# # Fit a line (y = mx + c)
# m, c = np.polyfit(x, y, 1)  # 1 specifies a linear fit (degree=1)
