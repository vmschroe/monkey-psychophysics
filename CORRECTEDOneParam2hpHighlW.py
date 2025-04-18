#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:41:46 2025

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
import sys
from datetime import datetime
import subprocess

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
hyper_fixed = np.array([[1,1],[1,1],[40,1.6],[1,1]])
params_prior_params = np.array([[2, 5], [2, 5], [4.5, 0.1], [2, 0.6]])
params_prior_scale = np.array([0.25, 0.25, 1, 1])
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
    vector = np.int64(np.array([40.0] * 8))
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
        #print(sess_params)
        y = sim_exp_data(sess_params, n)
        psych_vecs_sim['Y']['all'][key] = y
        psych_vecs_sim['NY']['all'][key] = n*y
    return psych_vecs_sim, Ls

def data_analysis(psych_vecs_df, grp_name = " ", num_post_samps = 1000):
    
    start_time = time.time()
    
    print('----------------------------------------------------------------------')
    print('Loading data')
    print('-- please wait --')
    
    sessions = sorted(psych_vecs_df['NY']['all'].keys())
    n_sessions = len(sessions)
   
    NY_data = np.array([psych_vecs_df['NY']['all'][k][0] for k in sessions])
    N_data = np.array([psych_vecs_df['N']['all'][k] for k in sessions])
    print("N_data shape:", N_data.shape)
    print("NY_data shape:", NY_data.shape)

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
        # Reshape data for vectorized operations
        x_expanded = np.tile(x, (n_sessions, 1))  # Shape: (n_sessions, n_stimulus_levels)
        
        # Create probabilities for all sessions at once
        session_probs = ffg.phi_W([gam, lam, L_session[:, None], k], x_expanded)
        print("x_expanded shape:", x_expanded.shape)
        print("session_probs shape:", session_probs.shape)
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
        try:
            trace = pm.sample(num_post_samps, return_inferencedata=True, progressbar=True)
            print('Posterior samples are drawn')
            post_timestamp = time.time()
            post_duration = post_timestamp - Lik_timestamp
            print(f"Posterior samples completed in {post_duration:.2f} seconds ({post_duration/60:.2f} minutes)")
            total_duration = post_timestamp - start_time
            print(f"Data analysis took {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            print('----------------------------------------------------------------------')
            return trace
        except Exception as e:
            print(f"Error during sampling: {e}")
            err_timestamp = time.time()
            total_duration = err_timestamp - start_time
        
    



# Create a timestamp for the output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "sim_1O2HP_HighWeib_output_" + timestamp
os.makedirs(output_dir, exist_ok=True)

# Set up logging
log_path = os.path.join(output_dir, "sim_1O2HP_HighWeib_log.txt")
log_file = open(log_path, "w")
original_stdout = sys.stdout
sys.stdout = log_file

try:
    print(f"Simulation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run your simulation and analysis
    psych_vecs_sim, Ls = sim_all_data()
    trace = data_analysis(psych_vecs_sim, num_post_samps = 1000)
    
    
    plt.figure(figsize=(10, 6))
    L_posterior = np.mean(trace.posterior["L_session"].values, axis=(0, 1))
    plt.scatter(Ls, L_posterior)
    plt.plot([min(Ls), max(Ls)], [min(Ls), max(Ls)], 'k--')
    plt.xlabel('True L values')
    plt.ylabel('Estimated L values (posterior mean)')
    plt.title('Recovery of L parameter')
    plot_path = os.path.join(output_dir, "L_recovery_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    # Save the results
    results_path = os.path.join(output_dir, "sim_1O2HP_HighWeib_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'psych_vecs_sim': psych_vecs_sim,
            'Ls': Ls,
            'trace': trace
        }, f)
    
    print(f"Results saved to {results_path}")
    print(f"Simulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc(file=log_file)

finally:
    # Restore original stdout
    sys.stdout = original_stdout
    log_file.close()
    print(f"Simulation complete. Log saved to {log_path}")
    
    # Push results to GitHub
    try:
        # Stage new files
        subprocess.run(["git", "add", output_dir], check=True)
        
        # Commit changes
        commit_message = f"Add simulation results from {timestamp}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push to remote repository
        subprocess.run(["git", "push"], check=True)
        
        print("Successfully pushed results to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}")
        
        
        

