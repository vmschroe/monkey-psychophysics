#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:59:08 2025

@author: vmschroe
"""
import scipy.stats
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
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
import arviz as az
import sys
import FunctionsForGeneralized as ffg
import seaborn as sns
import pickle
import numpy as np

# Simulating for higherarchical weibul


# params = [gam,lam,L,k]
params_fixed = np.array([0.01, 0.05, 35, 1.7])
hyper_fixed = np.array([[1,1],[1,1],[4.5,0.18],[1,1]])
hyper_dict = {
    'gam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'beta': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'lam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'beta': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'L': {
        'distribution': 'gamma',
        'C': 1,
        'hps': {
            'alpha': {
                'W_dist': 'gamma',
                'params': [1.75,0.5]
            },
            'beta': {
                'dist': 'gamma',
                'params': [2.2,12]
            }
        }
    },
    'k': {
        'distribution': 'gamma',
        'C': 1,
        'hps': {
            'alpha': {
                'W_dist': 'gamma',
                'params': [2.0,1.6]
            },
            'beta': {
                'dist': 'gamma',
                'params': [2.5,9.4]
            }
        }
    }
}
hyper_deets = [hyper_dict['gam'], hyper_dict['lam'], hyper_dict['L'], hyper_dict['k']]
x = np.array([6, 12, 18, 24, 32, 38, 44, 50])



def sim_exp_data(params, n=40):
    
    num_repeats = 1
    # Perform simulations
    results = []
    for _ in range(num_repeats):
        # Generate test data y
        ny = binom.rvs(n, phi_W(params, x))
        y = ny / n
        results.append(y)
    return np.array(results)

def sim_all_data(n=40):
    vector = np.array([40.0] * 8)
    psych_vecs_sim = {
        'Y': {'all' : {i: vector.copy() for i in range(1, 46)}},
        'N': {'all' : {i: vector.copy() for i in range(1, 46)}},
        'NY': {'all' : {i: vector.copy() for i in range(1, 46)}}
        }
    
    Ls = []
    
    for key in psych_vecs_sim['N']['all'].keys():
        sess_params = params_fixed.copy()
        L = scipy.stats.gamma.rvs(a = hyper_fixed[2][0], scale = 1/hyper_fixed[2][1])
        Ls.append(L)
        sess_params[2] = L
        y = sim_exp_data(sess_params, n)
        psych_vecs_sim['Y']['all'][key] = y
        psych_vecs_sim['NY']['all'][key] = n*y
    return psych_vecs_sim, Ls
    
def phi_W(params,x):
    [gam,lam,L,k] = params
    x = np.asarray(x)
    weib = 1 - np.exp(-(x/L)**k)
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

psych_vecs_sim = sim_all_data()[0]
    
def data_analysis(psych_vecs_df, grp_name = " ", printsum = True):
    
    print('----------------------------------------------------------------------')
    print('Loading data')
    
    sessions = sorted(psych_vecs_df['NY']['all'].keys())
    n_sessions = len(sessions)
   
    NY_data = np.array([psych_vecs_df['NY']['all'][k] for k in sessions])
    N_data = np.array([psych_vecs_df['N']['all'][k] for k in sessions])

    nsum = sum(sum(N_data))
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        #gam
        gam = params_fixed[0]
        #lam
        lam = params_fixed[1]
        #L hyperpriors
        W_aL = pm.Gamma("W_aL",alpha=hyper_deets[2]['hps']['alpha']['params'][0],beta=hyper_deets[2]['hps']['alpha']['params'][1])
        alpha_L = pm.Deterministic("alpha_L", W_aL+1)
        beta_L = pm.Gamma("beta_L",alpha=hyper_deets[2]['hps']['beta']['params'][0],beta=hyper_deets[2]['hps']['beta']['params'][1])
        #L session-specific
        W_L_session = pm.Gamma("W_L_session", alpha=alpha_L, beta=beta_L, shape=n_sessions)
        L_session = pm.Deterministic("L_session", hyper_deets[2]['C']*W_L_session)
        #k
        k = params_fixed[3]
        # Session-specific PSE and JND
        pse_session = pm.Deterministic("pse_session", 
                                     PSE_W([gam, lam, L_session, k]))
        jnd_session = pm.Deterministic("jnd_session", 
                                     JND_W([gam, lam, L_session, k]))
        # Likelihood for each session
        for i in range(n_sessions):
            # Calculate probability for this session using its own L value
            p_i = phi_W([gam, lam, L_session[i], k], x)
            # Define likelihood for this session's data
            pm.Binomial(f"obs_{i}", n=N_data[i], p=p_i, observed=NY_data[i])
           
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True, progressbar=True)
    if printsum == True:
        print("Summary of parameter estimates:" + grp_name)
        print("Sample size:", nsum, "total trials")
        
        # Print group-level parameters
        print("\nGroup-level parameters:")
        print(az.summary(trace, var_names=["W_aL", "alpha_L", "beta_L"]))
        
        # Print session-specific parameters
        print("\nSession-specific L values:")
        session_summary = az.summary(trace, var_names=["L_session"])
        
        # Map indices to session IDs for clearer output
        for i, session_id in enumerate(sessions):
            print(f"Session {session_id}: L = {session_summary.loc['L_session[{i}]', 'mean']:.2f} " +
                  f"(95% CI: {session_summary.loc['L_session[{i}]', 'hdi_3%']:.2f} - " +
                  f"{session_summary.loc['L_session[{i}]', 'hdi_97%']:.2f})")
    return trace



trace = data_analysis(psych_vecs_sim)


L_ests = trace.posterior["L"].stack(sample=("chain", "draw")).values.mean(axis=1)
plt.scatter(Ls,L_ests)