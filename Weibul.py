#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:52 2025

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
import seaborn as sns
import pickle


with open("psych_vecs_all.pkl", "rb") as f:
    psych_vecs_all = pickle.load(f)
    
grps = ['ld', 'ln', 'rd', 'rn']
sessions = list(psych_vecs_all['NY']['ld'].keys())

#combine sessions
NY = {}
N = {}
Y = {}

for grp in grps:
    NY[grp] = sum(psych_vecs_all['NY'][grp].values())
    N[grp] = sum(psych_vecs_all['N'][grp].values())
    Y[grp] = NY[grp] / N[grp]
    
def phi_W(params,x):
    [gam,lam,L,k] = params
    x = np.asarray(x)
    weib = 1 - np.exp(-(x/L)^k)
    return gam + (1 - gam - lam) * weib

def phi_inv_W(params, p):
    [gam,lam,L,k] = params
    p = np.asarray(p)
    larg = (1-gam-lam)/(1-p-lam)
    return L* (np.log(larg))**(1/k)

def PSE_W(params):
    return phi_inv_W(params, 0.5)

def JND_W(params):
    x25 = phi_inv_W(params, 0.25)
    x75 = phi_inv_W(params, 0.75)
    return 0.5*(x75-x25)