#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:59:28 2025

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



params_prior_params = [ [], [], [], [] ]
params_prior_scale = [1,1,1,1]
#Parameters of prior beta distribution of W_gamma
params_prior_params[0] = [2,5] #[alpha,beta] 
params_prior_scale[0] = 0.25 # param = scale * W
#Parameters of prior beta distribution of W_lambda
params_prior_params[1] = [2,5] #[alpha,beta] 
params_prior_scale[1] = 0.25 # [min,max]
#Parameters of prior gamma distribution of W_b0
params_prior_params[2] = [4,1] #[alpha,beta] 
params_prior_scale[2] = -1 # [min,max]
#Parameters of prior gamma distribution of W_b1
params_prior_params[3] = [1.25,2.4] #[alpha,beta] 
params_prior_scale[3] = 1 # [min,max]
plot_all_posteriors = False

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
        W_b0 = pm.Gamma("W_b0",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        b0 = pm.Deterministic("b0", params_prior_scale[2]*W_b0)
        W_b1 = pm.Gamma("b1_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        b1 = pm.Deterministic("b1", params_prior_scale[3]*W_b1)
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", ffb.PSE(gam, lam, b0, b1))
        jnd = pm.Deterministic("jnd", ffb.JND(gam, lam, b0, b1))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=n, p=ffb.phi_with_lapses([gam, lam, b0, b1],x), observed=yndata)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace = pm.sample(1000, return_inferencedata=True)
        
    b0_samples = trace.posterior['b0'].values.flatten()
    b1_samples = trace.posterior['b1'].values.flatten()
    gam_samples = trace.posterior['gam'].values.flatten()
    lam_samples = trace.posterior['lam'].values.flatten()
    PSE_samples = trace.posterior['pse'].values.flatten()
    JND_samples = trace.posterior['jnd'].values.flatten()
        
    if plot_posts:
        az.plot_pair(trace, var_names=["gam", "lam", "b0", "b1"], kind='kde', marginals=True)
        plt.suptitle("Joint Posteriors of Parameters, "+group, fontsize=35)
        plt.show()
        
    return trace, [gam_samples, lam_samples, b0_samples, b1_samples], PSE_samples, JND_samples, ydata
    

    
def plot_curves_tog(trace1, param_samps1, ydata1, label1, trace2, param_samps2, ydata2, label2, w_hdi, plot_title):
    [gam_samples1, lam_samples1, b0_samples1, b1_samples1] = param_samps1
    [gam_samples2, lam_samples2, b0_samples2, b1_samples2] = param_samps2
    
    xfit = np.linspace(6,50,1000)
    rec_params1 = [float(az.summary(trace1)["mean"][param]) for param in ["gam", "lam", "b0", "b1"]]
    rec_params2 = [float(az.summary(trace2)["mean"][param]) for param in ["gam", "lam", "b0", "b1"]]
    y_samps1 = np.array([ffb.phi_with_lapses([gam,lam,b0,b1], xfit) for gam, lam, b0, b1 in zip(gam_samples1, lam_samples1, b0_samples1, b1_samples1)])    
    y_samps2 = np.array([ffb.phi_with_lapses([gam,lam,b0,b1], xfit) for gam, lam, b0, b1 in zip(gam_samples2, lam_samples2, b0_samples2, b1_samples2)])    
    hdi1 = az.hdi(y_samps1, hdi_prob=0.95)
    yrec1 = ffb.phi_with_lapses(rec_params1,xfit)
    hdi2 = az.hdi(y_samps2, hdi_prob=0.95)
    yrec2 = ffb.phi_with_lapses(rec_params2,xfit)
        
    plt.plot(xfit,yrec1,label=label1,color='red')
    plt.scatter(x,ydata1,label='Data', color = 'darkred')
    plt.plot(xfit,yrec2,label=label2,color='blue')
    plt.scatter(x,ydata2,label='Data', color = 'navy')
    if w_hdi:
        plt.fill_between(xfit, hdi1[:, 0], hdi1[:, 1], color='salmon', alpha=0.3, label='95% HDI')
        plt.fill_between(xfit, hdi2[:, 0], hdi2[:, 1], color='skyblue', alpha=0.3, label='95% HDI')
    plt.title(plot_title)
    plt.xlabel('Stimulus Amplitude')
    plt.legend(frameon=False, fontsize=9.5)
    plt.savefig(plot_title+".png", dpi=300)
    plt.show()   


def plot_attr_dist_with_hdi(samples1, label1, samples2, label2, plot_title, x_label):
    color1 = "red"
    color2 = "blue"
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






trace_ld, param_samps_ld, PSE_samples_ld, JND_samples_ld, ydata_ld = bayes_data_analysis(dfs[0], group_names[0], plot_all_posteriors)
trace_ln, param_samps_ln, PSE_samples_ln, JND_samples_ln, ydata_ln = bayes_data_analysis(dfs[1], group_names[1], plot_all_posteriors)
trace_rd, param_samps_rd, PSE_samples_rd, JND_samples_rd, ydata_rd = bayes_data_analysis(dfs[2], group_names[2], plot_all_posteriors)
trace_rn, param_samps_rn, PSE_samples_rn, JND_samples_rn, ydata_rn = bayes_data_analysis(dfs[3], group_names[3], plot_all_posteriors)


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
az.plot_trace(trace_ld, var_names=["gam", "lam", "b0", "b1"])


print("Left Hand, Distracted")
print(az.summary(trace_ld, var_names=["gam", "lam", "b0", "b1", "pse", "jnd"]))
print("Left Hand, Not Distracted")
print(az.summary(trace_ln, var_names=["gam", "lam", "b0", "b1", "pse", "jnd"]))
print("Right Hand, Distracted")
print(az.summary(trace_rd, var_names=["gam", "lam", "b0", "b1", "pse", "jnd"]))
print("Right Hand, Not Distracted")
print(az.summary(trace_rn, var_names=["gam", "lam", "b0", "b1", "pse", "jnd"]))

