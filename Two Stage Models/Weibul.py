<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:52 2025

@author: vmschroe
"""

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
# from ipywidgets import interact
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
sys.path.append("/home/vmschroe/Documents/Monkey Analysis/Github")
import FunctionsForBayes as ffb
import seaborn as sns
import pickle

params_prior_params = [ [], [], [], [] ]
params_prior_scale = [1,1,1,1]
#Parameters of prior beta distribution of W_gamma
params_prior_params[0] = [2,5] #[alpha,beta] 
params_prior_scale[0] = 0.25 # param = scale * W
#Parameters of prior beta distribution of W_lambda
params_prior_params[1] = [2,5] #[alpha,beta] 
params_prior_scale[1] = 0.25 # 
#Parameters of prior gamma distribution of W_L
params_prior_params[2] = [4.5,0.1] #[alpha,beta] 
params_prior_scale[2] = 1 # 
#Parameters of prior gamma distribution of W_k
params_prior_params[3] = [2,0.6] #[alpha,beta] 
params_prior_scale[3] = 1 # 
plot_all_posteriors = False

x = [6, 12, 18, 24, 32, 38, 44, 50]

with open("psych_vecs_all.pkl", "rb") as f:
    psych_vecs_all = pickle.load(f)
    
grps = ['ld', 'ln', 'rd', 'rn']
sessions = list(psych_vecs_all['NY']['ld'].keys())

#combine sessions
NY = {}
N = {}
Y = {}
psych_vector = {}
labels = {
    'ld': 'Left hand, Distracted',
    'ln': 'Left hand, Not distracted',
    'rd': 'Right hand, Distracted',
    'rn': 'Right hand, Not distracted'
    }

for grp in grps:
    NY[grp] = sum(psych_vecs_all['NY'][grp].values())
    N[grp] = sum(psych_vecs_all['N'][grp].values())
    Y[grp] = NY[grp] / N[grp]
    psych_vector[grp] = [ N[grp], NY[grp]]
    
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

def data_analysis(psych_vec, grp_name = " ", printsum = True):
    print('----------------------------------------------------------------------')
    print('Loading data')
   
    n, yndata = psych_vec
    nsum = sum(n)
    ydata = yndata / n
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        W_gam = pm.Beta("W_gam",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_scale[0]*W_gam)
        W_lam = pm.Beta("W_lam",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_scale[1]*W_lam)
        W_L = pm.Gamma("W_L",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        L = pm.Deterministic("L", params_prior_scale[2]*W_L)
        W_k = pm.Gamma("k_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        k = pm.Deterministic("k", params_prior_scale[3]*W_k)
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", PSE_W([gam, lam, L, k]))
        jnd = pm.Deterministic("jnd", JND_W([gam, lam, L, k]))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=phi_W([gam, lam, L, k],x), observed=yndata)
        import numpy as np
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
    if printsum ==True:
        print("Summary of parameter estimates:"+grp_name)
        print("Sample size:", nsum, "total trials")
        print(az.summary(trace, var_names=["gam", "lam", "L", "k", "pse", "jnd"]))    
    return trace
  
def plot_post(trace, grp_name = " "):
    az.plot_pair(trace, var_names=["gam", "lam", "L", "k"], kind='kde', marginals=True)
    plt.suptitle("Joint Posteriors of Parameters, "+grp_name, fontsize=35)
    plt.show()
    
def plot_rec_curves(traces, ydatas, labels, plot_title="Recovered Curves", w_hdi = True, saveimg = False):
    gam_samples, lam_samples, L_samples, k_samples = [], [], [], []
    rec_params, y_samps, hdi, yrec = [], [], [], []
    col = [['red','darkred'], ['blue','navy'], ['limegreen','darkgreen'], ['darkorchid','indigo'], ['hotpink','deeppink'], ['red','darkred'], ['blue','navy'], ['limegreen','darkgreen'], ['darkorchid','indigo'], ['hotpink','deeppink']]
    xfit = np.linspace(6,50,1000)
    plt.figure(figsize=(8, 5))  # Create a new figure for each attribute
    for i, trace in enumerate(traces):
        # Extract posterior samples
        gam_samples.append(trace.posterior['gam'].values.flatten())
        lam_samples.append(trace.posterior['lam'].values.flatten())
        L_samples.append(trace.posterior['L'].values.flatten())
        k_samples.append(trace.posterior['k'].values.flatten()) 
        # Compute mean parameter estimates
        rec_params.append([float(az.summary(trace)["mean"][param]) for param in ["gam", "lam", "L", "k"]])
        # Compute the reconstructed curve samples
        y_samps.append(np.array([
            phi_W([gam, lam, L, k], xfit) 
            for gam, lam, L, k in zip(gam_samples[i], lam_samples[i], L_samples[i], k_samples[i])
            ]))
        # Compute 95% HDI
        hdi.append(az.hdi(y_samps[i], hdi_prob=0.95))

        # Compute the mean reconstructed curve
        yrec.append(phi_W(rec_params[i], xfit))
        
        # Plot reconstructed curve
        plt.plot(xfit, yrec[i], label=labels[i], color=col[i % len(col)][0])
        
        # Scatter plot of the actual data
        plt.scatter(x, ydatas[i], color=col[i % len(col)][1], label=f"Data {labels[i]}")
        
        # Plot HDI interval
        if w_hdi:
            plt.fill_between(xfit, hdi[i][:, 0], hdi[i][:, 1], color=col[i % len(col)][0], alpha=0.2)

         
    plt.title(plot_title)
    plt.xlabel('Stimulus Amplitude')
    plt.legend(frameon=False, fontsize=9.5)
    if saveimg == True:
        plt.savefig(plot_title+".png", dpi=300)
    plt.show()
    

def plot_attr_dist_with_hdi(traces, labels, attributes=["gam", "lam", "pse", "jnd"], saveimg=False):
    col = ['red', 'blue', 'limegreen', 'darkorchid', 'hotpink', 'red', 'blue', 'limegreen', 'darkorchid', 'hotpink']
    for att in attributes:
        plt.figure(figsize=(8, 5))  # Create a new figure for each attribute
        for i, trace in enumerate(traces):
            if att not in trace.posterior:
                print(f"Warning: '{att}' not found in trace {i}. Skipping.")
                continue
            
            param_sample = trace.posterior[att].values.flatten()
            sns.kdeplot(param_sample, label=labels[i], color=col[i % len(col)], fill=True, alpha=0.4)

            # Compute and plot HDI
            hdi = az.hdi(param_sample, hdi_prob=0.95)
            print(f"{labels[i]} ({att}) HDI: {hdi}")  # Debugging output
            plt.axvline(hdi[0], color=col[i % len(col)], linestyle="--", label=f"{labels[i]} 95% HDI")
            plt.axvline(hdi[1], color=col[i % len(col)], linestyle="--")

        # Add legend and labels
        plt.legend(frameon=False)
        plt.title(f"Posterior Distribution of {att}")
        plt.xlabel(att)
        
        if saveimg:
            filename = f"{att}_{'_'.join(labels)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()

traces_all = {}

for i, grp in enumerate(grps):
    traces_all[grp] = data_analysis(psych_vector[grp], grp_name = grp)
    plot_post(traces_all[grp], grp_name = labels[grp])
    

plot_rec_curves([traces_all['ld'], traces_all['ln']], [Y['ld'],Y['ln']], [labels['ld'],labels['ln']])
plot_attr_dist_with_hdi([traces_all['ld'], traces_all['ln']], [labels['ld'],labels['ln']])
plot_rec_curves([traces_all['rd'], traces_all['rn']], [Y['rd'],Y['rn']], [labels['rd'],labels['rn']])
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:52 2025

@author: vmschroe
"""

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
# from ipywidgets import interact
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
sys.path.append("/home/vmschroe/Documents/Monkey Analysis/Github")
import FunctionsForBayes as ffb
import seaborn as sns
import pickle

params_prior_params = [ [], [], [], [] ]
params_prior_scale = [1,1,1,1]
#Parameters of prior beta distribution of W_gamma
params_prior_params[0] = [2,5] #[alpha,beta] 
params_prior_scale[0] = 0.25 # param = scale * W
#Parameters of prior beta distribution of W_lambda
params_prior_params[1] = [2,5] #[alpha,beta] 
params_prior_scale[1] = 0.25 # 
#Parameters of prior gamma distribution of W_L
params_prior_params[2] = [4.5,0.1] #[alpha,beta] 
params_prior_scale[2] = 1 # 
#Parameters of prior gamma distribution of W_k
params_prior_params[3] = [2,0.6] #[alpha,beta] 
params_prior_scale[3] = 1 # 
plot_all_posteriors = False

x = [6, 12, 18, 24, 32, 38, 44, 50]

with open("psych_vecs_all.pkl", "rb") as f:
    psych_vecs_all = pickle.load(f)
    
grps = ['ld', 'ln', 'rd', 'rn']
sessions = list(psych_vecs_all['NY']['ld'].keys())

#combine sessions
NY = {}
N = {}
Y = {}
psych_vector = {}
labels = {
    'ld': 'Left hand, Distracted',
    'ln': 'Left hand, Not distracted',
    'rd': 'Right hand, Distracted',
    'rn': 'Right hand, Not distracted'
    }

for grp in grps:
    NY[grp] = sum(psych_vecs_all['NY'][grp].values())
    N[grp] = sum(psych_vecs_all['N'][grp].values())
    Y[grp] = NY[grp] / N[grp]
    psych_vector[grp] = [ N[grp], NY[grp]]
    
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

def data_analysis(psych_vec, grp_name = " ", printsum = True):
    print('----------------------------------------------------------------------')
    print('Loading data')
   
    n, yndata = psych_vec
    nsum = sum(n)
    ydata = yndata / n
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        W_gam = pm.Beta("W_gam",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_scale[0]*W_gam)
        W_lam = pm.Beta("W_lam",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_scale[1]*W_lam)
        W_L = pm.Gamma("W_L",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        L = pm.Deterministic("L", params_prior_scale[2]*W_L)
        W_k = pm.Gamma("k_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        k = pm.Deterministic("k", params_prior_scale[3]*W_k)
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", PSE_W([gam, lam, L, k]))
        jnd = pm.Deterministic("jnd", JND_W([gam, lam, L, k]))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=phi_W([gam, lam, L, k],x), observed=yndata)
        import numpy as np
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
    if printsum ==True:
        print("Summary of parameter estimates:"+grp_name)
        print("Sample size:", nsum, "total trials")
        print(az.summary(trace, var_names=["gam", "lam", "L", "k", "pse", "jnd"]))    
    return trace
  
def plot_post(trace, grp_name = " "):
    az.plot_pair(trace, var_names=["gam", "lam", "L", "k"], kind='kde', marginals=True)
    plt.suptitle("Joint Posteriors of Parameters, "+grp_name, fontsize=35)
    plt.show()
    
def plot_rec_curves(traces, ydatas, labels, plot_title="Recovered Curves", w_hdi = True, saveimg = False):
    gam_samples, lam_samples, L_samples, k_samples = [], [], [], []
    rec_params, y_samps, hdi, yrec = [], [], [], []
    col = [['red','darkred'], ['blue','navy'], ['limegreen','darkgreen'], ['darkorchid','indigo'], ['hotpink','deeppink'], ['red','darkred'], ['blue','navy'], ['limegreen','darkgreen'], ['darkorchid','indigo'], ['hotpink','deeppink']]
    xfit = np.linspace(6,50,1000)
    plt.figure(figsize=(8, 5))  # Create a new figure for each attribute
    for i, trace in enumerate(traces):
        # Extract posterior samples
        gam_samples.append(trace.posterior['gam'].values.flatten())
        lam_samples.append(trace.posterior['lam'].values.flatten())
        L_samples.append(trace.posterior['L'].values.flatten())
        k_samples.append(trace.posterior['k'].values.flatten()) 
        # Compute mean parameter estimates
        rec_params.append([float(az.summary(trace)["mean"][param]) for param in ["gam", "lam", "L", "k"]])
        # Compute the reconstructed curve samples
        y_samps.append(np.array([
            phi_W([gam, lam, L, k], xfit) 
            for gam, lam, L, k in zip(gam_samples[i], lam_samples[i], L_samples[i], k_samples[i])
            ]))
        # Compute 95% HDI
        hdi.append(az.hdi(y_samps[i], hdi_prob=0.95))

        # Compute the mean reconstructed curve
        yrec.append(phi_W(rec_params[i], xfit))
        
        # Plot reconstructed curve
        plt.plot(xfit, yrec[i], label=labels[i], color=col[i % len(col)][0])
        
        # Scatter plot of the actual data
        plt.scatter(x, ydatas[i], color=col[i % len(col)][1], label=f"Data {labels[i]}")
        
        # Plot HDI interval
        if w_hdi:
            plt.fill_between(xfit, hdi[i][:, 0], hdi[i][:, 1], color=col[i % len(col)][0], alpha=0.2)

         
    plt.title(plot_title)
    plt.xlabel('Stimulus Amplitude')
    plt.legend(frameon=False, fontsize=9.5)
    if saveimg == True:
        plt.savefig(plot_title+".png", dpi=300)
    plt.show()
    

def plot_attr_dist_with_hdi(traces, labels, attributes=["gam", "lam", "pse", "jnd"], saveimg=False):
    col = ['red', 'blue', 'limegreen', 'darkorchid', 'hotpink', 'red', 'blue', 'limegreen', 'darkorchid', 'hotpink']
    for att in attributes:
        plt.figure(figsize=(8, 5))  # Create a new figure for each attribute
        for i, trace in enumerate(traces):
            if att not in trace.posterior:
                print(f"Warning: '{att}' not found in trace {i}. Skipping.")
                continue
            
            param_sample = trace.posterior[att].values.flatten()
            sns.kdeplot(param_sample, label=labels[i], color=col[i % len(col)], fill=True, alpha=0.4)

            # Compute and plot HDI
            hdi = az.hdi(param_sample, hdi_prob=0.95)
            print(f"{labels[i]} ({att}) HDI: {hdi}")  # Debugging output
            plt.axvline(hdi[0], color=col[i % len(col)], linestyle="--", label=f"{labels[i]} 95% HDI")
            plt.axvline(hdi[1], color=col[i % len(col)], linestyle="--")

        # Add legend and labels
        plt.legend(frameon=False)
        plt.title(f"Posterior Distribution of {att}")
        plt.xlabel(att)
        
        if saveimg:
            filename = f"{att}_{'_'.join(labels)}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()

traces_all = {}

for i, grp in enumerate(grps):
    traces_all[grp] = data_analysis(psych_vector[grp], grp_name = grp)
    plot_post(traces_all[grp], grp_name = labels[grp])
    

plot_rec_curves([traces_all['ld'], traces_all['ln']], [Y['ld'],Y['ln']], [labels['ld'],labels['ln']])
plot_attr_dist_with_hdi([traces_all['ld'], traces_all['ln']], [labels['ld'],labels['ln']])
plot_rec_curves([traces_all['rd'], traces_all['rn']], [Y['rd'],Y['rn']], [labels['rd'],labels['rn']])
>>>>>>> 93055b0ab5a537b235aeab2bc3d478646c1a64d5
plot_attr_dist_with_hdi([traces_all['rd'], traces_all['rn']], [labels['rd'],labels['rn']])