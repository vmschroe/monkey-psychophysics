#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:23:06 2025

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
from scipy.special import erf
import jax


params_prior_params = [ [], [], [], [] ]
params_prior_scale = [1,1,1,1]
#Parameters of prior beta distribution of W_gamma
params_prior_params[0] = [2,5] #[alpha,beta] 
params_prior_scale[0] = 0.25 # param = scale * W
#Parameters of prior beta distribution of W_lambda
params_prior_params[1] = [2,5] #[alpha,beta] 
params_prior_scale[1] = 0.25 # [min,max]
#Parameters of prior gamma distribution of W_mu
params_prior_params[2] = [5.5,0.16] #[alpha,beta] 
params_prior_scale[2] = 1 # [min,max]
#Parameters of prior gamma distribution of W_sigma
params_prior_params[3] = [2.08,0.054] #[alpha,beta] 
params_prior_scale[3] = 1 # [min,max]

##Load and clean data
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/loaddata.py').read())
## constructs 4 dataframes:
    # df_ld
    # df_ln
    # df_rd
    # df_rn


#Construct necessary functions
exec(open('/home/vmschroe/Documents/Monkey Analysis/Github/FunctionsForBayes.py').read())

dfs = [df_ld, df_ln, df_rd, df_rn]
group_names = ["Left hand, Distracted", "Left hand, Not distracted","Right hand, Distracted", "Right hand, Not distracted"]


x = [6, 12, 18, 24, 32, 38, 44, 50] #stimulus amplitudes


def phi(params, x):
    gam, lam, mu, sigma = params
    x_arr = np.array(x)
    
    # Check if any input is a PyMC variable by looking for common attributes
    try:
        has_pymc_var = any(hasattr(p, 'eval') or hasattr(p, 'tag') or hasattr(p, 'logp') for p in params)
    except:
        # If we can't check attributes (e.g., with JAX arrays), assume it's not a PyMC variable
        has_pymc_var = False
    
    if has_pymc_var:
        # Use pm.math functions for PyMC variables
        return gam + (1 - gam - lam) * 0.5 * (1 + pm.math.erf((x_arr - mu) / (sigma * pm.math.sqrt(2))))
    else:
        # Use scipy/numpy for regular numeric inputs
        return gam + (1 - gam - lam) * 0.5 * (1 + erf((x_arr - mu) / (sigma * np.sqrt(2))))


def solve_phi_g_for_x(params, p):
    gam, lam, mu, sigma = params
    
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive.")
    
    # Compute the inverse
    z = (2 * (p - gam) / (1 - gam - lam)) - 1
    
    if z < -1 or z > 1:
        raise ValueError(f"No solution: {p} is outside the valid range of phi.")

    x = mu + sigma * np.sqrt(2) * pm.math.erfinv(z)
    return x

def PSE_g(params):
    return solve_phi_g_for_x(params, 0.5)

def JND_g(params):
    x25 = solve_phi_g_for_x(params, 0.25)
    x75 = solve_phi_g_for_x(params, 0.75)
    return 0.5*(x75-x25)

def bayes_data_analysis(df, grp_name, plot_posts):
    print('----------------------------------------------------------------------')
    print('Loading data')
    
    group = grp_name
    # Generate sample observed data
    # ydata = ffb.sim_exp_data(sim_params, n)
    # yndata = ydata*n
    ydata, n, yndata, x = ffb.psych_vectors(df)
    nsum = sum(n)
    
    # Define the model
    with pm.Model() as model:
        # Define priors for the parameters
        W_gam = pm.Beta("W_gam",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_scale[0]*W_gam)
        W_lam = pm.Beta("W_lam",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_scale[1]*W_lam)
        W_mu = pm.Gamma("W_mu",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        mu = pm.Deterministic("mu", params_prior_scale[2]*W_mu)
        W_sigma = pm.Gamma("sigma_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        sigma = pm.Deterministic("sigma", params_prior_scale[3]*W_sigma)
    
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", PSE_g([gam, lam, mu, sigma]))
        jnd = pm.Deterministic("jnd", JND_g([gam, lam, mu, sigma]))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=phi([gam, lam, mu, sigma],x), observed=yndata)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
        
    # Extract parameter samples
    mu_samples = trace.posterior['mu'].values.flatten()
    sigma_samples = trace.posterior['sigma'].values.flatten()
    gam_samples = trace.posterior['gam'].values.flatten()
    lam_samples = trace.posterior['lam'].values.flatten()
    
    # Extract PSE and JND samples directly from trace
    PSE_samples = trace.posterior['pse'].values.flatten()
    JND_samples = trace.posterior['jnd'].values.flatten()
    
    if plot_posts:
        az.plot_pair(trace, var_names=["gam", "lam", "mu", "sigma"], kind='kde', marginals=True)
        plt.suptitle("Joint Posteriors of Parameters, "+group, fontsize=35)
        plt.show()
        
    return trace, [gam_samples, lam_samples, mu_samples, sigma_samples], PSE_samples, JND_samples, ydata
    

    
def plot_curves_tog(trace1, param_samps1, ydata1, label1, trace2, param_samps2, ydata2, label2, w_hdi, plot_title):
    [gam_samples1, lam_samples1, mu_samples1, sigma_samples1] = param_samps1
    [gam_samples2, lam_samples2, mu_samples2, sigma_samples2] = param_samps2
    
    xfit = np.linspace(6,50,1000)
    rec_params1 = [float(az.summary(trace1)["mean"][param]) for param in ["gam", "lam", "mu", "sigma"]]
    rec_params2 = [float(az.summary(trace2)["mean"][param]) for param in ["gam", "lam", "mu", "sigma"]]
    
   # Convert tensor variables to numpy arrays
    def convert_to_numpy(tensor_var):
        if hasattr(tensor_var, 'eval'):
            return tensor_var.eval()
        return np.array(tensor_var)
    
    # Convert all samples to numpy arrays
    gam_samples1 = convert_to_numpy(gam_samples1)
    lam_samples1 = convert_to_numpy(lam_samples1)
    mu_samples1 = convert_to_numpy(mu_samples1)
    sigma_samples1 = convert_to_numpy(sigma_samples1)
    
    gam_samples2 = convert_to_numpy(gam_samples2)
    lam_samples2 = convert_to_numpy(lam_samples2)
    mu_samples2 = convert_to_numpy(mu_samples2)
    sigma_samples2 = convert_to_numpy(sigma_samples2)
    
    # Calculate predicted values for all parameter samples
    y_samps1 = np.array([phi([g, l, m, s], xfit) 
                         for g, l, m, s in zip(gam_samples1, lam_samples1, 
                                             mu_samples1, sigma_samples1)])
    
    y_samps2 = np.array([phi([g, l, m, s], xfit) 
                         for g, l, m, s in zip(gam_samples2, lam_samples2, 
                                             mu_samples2, sigma_samples2)])
    
    
    # Calculate mean curves
    yrec1 = phi(rec_params1, xfit)
    yrec2 = phi(rec_params2, xfit)
    
    plt.figure(figsize=(10, 6))
    
    # Plot HDI bands if requested
    if w_hdi:
        hdi1 = az.hdi(np.float64(y_samps1), hdi_prob=0.95)
        hdi2 = az.hdi(np.float64(y_samps2), hdi_prob=0.95)
        plt.fill_between(xfit, hdi1[:, 0], hdi1[:, 1], color='salmon', alpha=0.3, label=f'{label1} 95% HDI')
        plt.fill_between(xfit, hdi2[:, 0], hdi2[:, 1], color='skyblue', alpha=0.3, label=f'{label2} 95% HDI')
    
    # Plot mean curves and data points
    plt.plot(xfit, yrec1, label=label1, color='red')
    plt.scatter(x, ydata1, color='darkred', alpha=0.6)
    plt.plot(xfit, yrec2, label=label2, color='blue')
    plt.scatter(x, ydata2, color='navy', alpha=0.6)
    
    plt.title(plot_title, fontsize=12)
    plt.xlabel('Stimulus Amplitude', fontsize=10)
    plt.ylabel('Response Probability', fontsize=10)
    plt.legend(frameon=False, fontsize=9.5)
    plt.grid(True, alpha=0.3)


# def plot_attr_dist_with_hdi(samples1, label1, samples2, label2, plot_title, x_label):
#     color1 = "red"
#     color2 = "blue"
#     # Plot the KDE for both distributions
#     sns.kdeplot(samples1, label=label1, color=color1, fill=True, alpha=0.4)
#     sns.kdeplot(samples2, label=label2, color=color2, fill=True, alpha=0.4)

#     # Compute HDIs
#     hdi_1 = az.hdi(samples1, hdi_prob=0.95)
#     hdi_2 = az.hdi(samples2, hdi_prob=0.95)

#     # Print HDI values for debugging
#     print(f"{label1} HDI: {hdi_1}")
#     print(f"{label2} HDI: {hdi_2}")

#     # Plot HDI for first distribution
#     plt.axvline(hdi_1[0], color=color1, linestyle="--", label=f"{label1} 95% HDI")
#     plt.axvline(hdi_1[1], color=color1, linestyle="--")

#     # Plot HDI for second distribution
#     plt.axvline(hdi_2[0], color=color2, linestyle="--", label=f"{label2} 95% HDI")
#     plt.axvline(hdi_2[1], color=color2, linestyle="--")

#     # Add legend and title
#     plt.legend(frameon=False)
#     plt.title(plot_title)
#     plt.xlabel(x_label)
#     plt.savefig(plot_title+".png", dpi=300, bbox_inches='tight')
#     plt.show()

def plot_attr_dist_with_hdi(samples1, label1, samples2, label2, plot_title, x_label):
    color1 = "red"
    color2 = "blue"
    
    # Convert samples to numpy arrays if they aren't already
    samples1 = np.array(samples1).astype(float)
    samples2 = np.array(samples2).astype(float)
    
    # Plot the KDE for both distributions
    sns.kdeplot(samples1, label=label1, color=color1, fill=True, alpha=0.4)
    sns.kdeplot(samples2, label=label2, color=color2, fill=True, alpha=0.4)

    # Compute HDIs
    hdi_1 = az.hdi(samples1, hdi_prob=0.95)
    hdi_2 = az.hdi(samples2, hdi_prob=0.95)

    # Print HDI values for debugging
    print(f"{label1} HDI: {hdi_1}")
    print(f"{label2} HDI: {hdi_2}")

    # Plot HDI for first distribution
    plt.axvline(hdi_1[0], color=color1, linestyle="--", label=f"{label1} 95% HDI")
    plt.axvline(hdi_1[1], color=color1, linestyle="--")

    # Plot HDI for second distribution
    plt.axvline(hdi_2[0], color=color2, linestyle="--", label=f"{label2} 95% HDI")
    plt.axvline(hdi_2[1], color=color2, linestyle="--")

    # Add legend and title
    plt.legend(frameon=False)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.savefig(plot_title+".png", dpi=300, bbox_inches='tight')
    plt.show()




trace_ld, param_samps_ld, PSE_samples_ld, JND_samples_ld, ydata_ld = bayes_data_analysis(dfs[0], group_names[0], True)
trace_ln, param_samps_ln, PSE_samples_ln, JND_samples_ln, ydata_ln = bayes_data_analysis(dfs[1], group_names[1], True)
trace_rd, param_samps_rd, PSE_samples_rd, JND_samples_rd, ydata_rd = bayes_data_analysis(dfs[2], group_names[2], True)
trace_rn, param_samps_rn, PSE_samples_rn, JND_samples_rn, ydata_rn = bayes_data_analysis(dfs[3], group_names[3], True)


############

plot_curves_tog(trace_ld, param_samps_ld, ydata_ld, group_names[0], trace_ln, param_samps_ln, ydata_ln, group_names[1], True, "Left Hand")
plot_curves_tog(trace_rd, param_samps_rd, ydata_rd, group_names[2], trace_rn, param_samps_rn, ydata_rn, group_names[3], True, "Right Hand")


plot_attr_dist_with_hdi(PSE_samples_ld, "Distracted", PSE_samples_ln, "Not Distracted", "Left Hand PSE", "PSE")
plot_attr_dist_with_hdi(PSE_samples_rd, "Distracted", PSE_samples_rn, "Not Distracted", "Right Hand PSE", "PSE")
plot_attr_dist_with_hdi(JND_samples_ld, "Distracted", JND_samples_ln, "Not Distracted", "Left Hand JND", "JND")
plot_attr_dist_with_hdi(JND_samples_rd, "Distracted", JND_samples_rn, "Not Distracted", "Right Hand JND", "JND")

plot_attr_dist_with_hdi(param_samps_ld[0], "Distracted", param_samps_ln[0], "Not Distracted", "Left Hand Gamma (guess rate)", "gamma")
plot_attr_dist_with_hdi(param_samps_rd[0], "Distracted", param_samps_rn[0], "Not Distracted", "Right Hand Gamma (guess rate)", "gamma")
plot_attr_dist_with_hdi(param_samps_ld[1], "Distracted", param_samps_ln[1], "Not Distracted", "Left Hand Lambda (lapse rate)", "lambda")
plot_attr_dist_with_hdi(param_samps_rd[1], "Distracted", param_samps_rn[1], "Not Distracted", "Right Hand Lambda (lapse rate)", "lambda")

################
az.plot_trace(trace_ld, var_names=["gam", "lam", "mu", "sigma"])


print("Left Hand, Distracted")
print(az.summary(trace_ld, var_names=["gam", "lam", "mu", "sigma"]))
print("Left Hand, Not Distracted")
print(az.summary(trace_ln, var_names=["gam", "lam", "mu", "sigma"]))
print("Right Hand, Distracted")
print(az.summary(trace_rd, var_names=["gam", "lam", "mu", "sigma"]))
print("Right Hand, Not Distracted")
print(az.summary(trace_rn, var_names=["gam", "lam", "mu", "sigma"]))

print("-----------------PSE------------")
print("Left Hand, Not Distracted")
print(az.summary(PSE_samples_ln))
print("Left Hand, Distracted")
print(az.summary(PSE_samples_ld))
print("Right Hand, Not Distracted")
print(az.summary(PSE_samples_rn))
print("Right Hand, Distracted")
print(az.summary(PSE_samples_rd))

print("-----------------JND------------")
print("Left Hand, Not Distracted")
print(az.summary(JND_samples_ln))
print("Left Hand, Distracted")
print(az.summary(JND_samples_ld))
print("Right Hand, Not Distracted")
print(az.summary(JND_samples_rn))
print("Right Hand, Distracted")
print(az.summary(JND_samples_rd))



