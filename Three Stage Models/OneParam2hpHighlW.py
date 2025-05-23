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
from ipywidgets import interact
import pandas as pd
import os
import math
from scipy.stats import binom

import sys
import FunctionsForGeneralized as ffg
import seaborn as sns
import pickle
import tqdm
import time

np.random.seed(12345)
# Simulating for higherarchical weibul
# params = [gam,lam,L,k]
#### NOTE: For a GAMMA distribution Gamma[alpha,beta]:
    #       alpha = shape, beta = rate
    #       scipy.stats.gamma : a = alpha = shape, scale = 1/beta = 1/rate
    #       pm.Gamma : alpha = shape, beta = rate
    
#### NOTE: For a BETA distribution Beta[alpha,beta]:
    #       alpha, beta are both shape parameters
    #       scipy.stats.beta : a = alpha, b = beta
    #       pm.Beta : alpha, beta
params_fixed = np.array([0.01, 0.05, 35, 1.7])
hyper_fixed = np.array([[1,1],[1,1],[4.5,0.18],[1,1]])
hyper_dict = {
    'gam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha_prior': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'lam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha_prior': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'L': {
        'distribution': 'gamma',
        'C': 1,
        'hps': {
            'shape_prior': {
                'W_dist': 'gamma',
                'params': [1.02,0.1]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [2.2,12]
            }
        }
    },
    'k': {
        'distribution': 'gamma',
        'C': 1,
        'hps': {
            'shape_prior': {
                'W_dist': 'gamma',
                'params': [2.0,1.6]
            },
            'rate_prior': {
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
        ny = binom.rvs(n, ffg.phi_W(params, x))
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

psych_vecs_sim, Ls = sim_all_data()
    
def data_analysis(psych_vecs_df, grp_name = " ", num_post_samps = 1000):
    
    start_time = time.time()
    
    print('----------------------------------------------------------------------')
    print('Loading data')
    print('-- please wait --')
    
    sessions = sorted(psych_vecs_df['NY']['all'].keys())
    n_sessions = len(sessions)
   
    NY_data = np.array([psych_vecs_df['NY']['all'][k] for k in sessions])
    N_data = np.array([psych_vecs_df['N']['all'][k] for k in sessions])

    nsum = sum(sum(N_data))
    
    print('Data is loaded')
    load_timestamp = time.time()
    load_duration = load_timestamp - start_time
    print(f"Data loading completed in {load_duration:.2f} seconds ({load_duration/60:.2f} minutes)")
    print('----------------------------------------------------------------------')
    print('Constructing priors and parameters')
    print('-- please wait --')
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        #gam
        gam = params_fixed[0]
        #lam
        lam = params_fixed[1]
        #L hyperpriors
        W_aL = pm.Gamma("W_aL",alpha=hyper_deets[2]['hps']['shape_prior']['params'][0],beta=hyper_deets[2]['hps']['shape_prior']['params'][1])
        alpha_L = pm.Deterministic("alpha_L", W_aL+1)
        beta_L = pm.Gamma("beta_L",alpha=hyper_deets[2]['hps']['rate_prior']['params'][0],beta=hyper_deets[2]['hps']['rate_prior']['params'][1])
        #L session-specific
        W_L_session = pm.Gamma("W_L_session", alpha=alpha_L, beta=beta_L, shape=n_sessions)
        L_session = pm.Deterministic("L_session", hyper_deets[2]['C']*W_L_session)
        #k
        k = params_fixed[3]
        # Session-specific PSE and JND
        pse_session = pm.Deterministic("pse_session", 
                                     ffg.PSE_W([gam, lam, L_session, k]))
        jnd_session = pm.Deterministic("jnd_session", 
                                     ffg.JND_W([gam, lam, L_session, k]))
        
        print('Priors and parameters are constructed')
        PP_timestamp = time.time()
        PP_duration = PP_timestamp - load_timestamp
        print(f"Priors and parameters completed in {PP_duration:.2f} seconds ({PP_duration/60:.2f} minutes)")
        print('----------------------------------------------------------------------')
        print('Constructing likelihoods')
        print('-- please wait --')
        # Likelihood for each session
        # for i in range(n_sessions):
        #     # Calculate probability for this session using its own L value
        #     p_i = ffg.phi_W([gam, lam, L_session[i], k], x)
        #     # Define likelihood for this session's data
        #     pm.Binomial(f"obs_{i}", n=N_data[i], p=p_i, observed=NY_data[i])
        #     print(f" Session {sessions[i]} likelihood defined ( {i} of {n_sessions} )")
        # Reshape data for vectorized operations
        x_expanded = np.tile(x, (n_sessions, 1))  # Shape: (n_sessions, n_stimulus_levels)
        
        # Create probabilities for all sessions at once
        session_probs = ffg.phi_W([gam, lam, L_session[:, None], k], x_expanded)
        
        # Single vectorized likelihood
        pm.Binomial("obs", n=N_data, p=session_probs, observed=NY_data)
        
        print('Likelihoods are constructed')
        Lik_timestamp = time.time()
        Lik_duration = Lik_timestamp - PP_timestamp
        print(f"Likelihoods completed in {Lik_duration:.2f} seconds ({Lik_duration/60:.2f} minutes)")
        print('----------------------------------------------------------------------')
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        print(f'Drawing {num_post_samps} samples from Posterior')
        print('-- please wait (A LONG TIME) --')
        trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True)
        print('Posterior samples are drawn')
        post_timestamp = time.time()
        post_duration = post_timestamp - Lik_timestamp
        print(f"Posterior samples completed in {post_duration:.2f} seconds ({post_duration/60:.2f} minutes)")
        total_duration = post_timestamp - start_time
        print(f"Data analysis took {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print('----------------------------------------------------------------------')
    return trace



trace = data_analysis(psych_vecs_sim, num_post_samps = 1000)

L_ests = trace.posterior["L_session"].stack(sample=("chain", "draw")).values.mean(axis=1)

plt.scatter(Ls,L_ests)
plt.show()

# plt.savefig('LTrueVEsts.png')


az.plot_pair(trace, var_names=["L_session"], kind='kde', marginals=True)
plt.show()

az.plot_pair(trace, var_names=["alpha_L", "beta_L"], kind='kde', marginals=True)
plt.show()






# prior of L
appr_size = 50000
WLS = scipy.stats.gamma.rvs(a = hyper_deets[2]['hps']['shape_prior']['params'][0], scale = 1/hyper_deets[2]['hps']['shape_prior']['params'][1], size = appr_size)
BLS = scipy.stats.gamma.rvs(a = hyper_deets[2]['hps']['rate_prior']['params'][0], scale = 1/hyper_deets[2]['hps']['rate_prior']['params'][1], size = appr_size)
LPS = scipy.stats.gamma.rvs(a = WLS+1, scale = 1/BLS)

H = plt.hist(LPS,density=True,bins = 2000);
plt.show()
Lpart = H[1]
Lgrid = (Lpart[:-1] + Lpart[1:]) / 2

#posterior of L for sessions
L_post_samps = trace.posterior["L_session"].stack(sample=("chain", "draw")).values


for sess in [15,37,41]:
    #plot
    plt.hist(L_post_samps[sess], density = True, label = 'Histogram of samples from posterior')
    plt.plot(Lgrid, H[0], label = 'Approximation of Prior')
    plt.vlines(x=Ls[sess], ymin=0, ymax=0.03, color='r', linestyle='--', label='True L for session')
    plt.xlim(min(Ls)-10, max(Ls)+10)
    plt.ylim(0,0.03)
    plt.xlabel('L value')
    plt.ylabel('density')
    plt.title(f'Prior, Posterior, and true value, Session {sess}')
    plt.legend()
    plt.show()
    
#Hyperparameter alpha
alpha_post_samps = trace.posterior["alpha_L"].stack(sample=("chain", "draw")).values
mi = min(alpha_post_samps)
ma = max(alpha_post_samps)
plt.hist(alpha_post_samps, density = True, bins = 20, label = 'Histogram of samples from posterior')
prior_pdf_alpha = scipy.stats.gamma.pdf(np.linspace(0,ma,100), a = hyper_deets[2]['hps']['shape_prior']['params'][0], scale = 1/hyper_deets[2]['hps']['shape_prior']['params'][1])
plt.plot(np.linspace(0,ma,100)+1, prior_pdf_alpha, label = 'Prior')
plt.vlines(x=hyper_fixed[2][0], ymin=0, ymax=0.05, color='r', linestyle='--', label='True alpha_L for simulation')
plt.xlim(0, ma+1)
plt.ylim(0,0.05)
plt.xlabel('alpha_L value')
plt.ylabel('density')
plt.title('Prior, Posterior, and true value, of hyperparameter alpha_L')
plt.legend()
plt.show()

#L vs estimates
plt.scatter(Ls,L_ests)
plt.xlabel('True L values used in sim')
plt.ylabel('Estimated L value for corresponding session')
plt.title('L parameter')
plt.legend()
plt.show()
