# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:05:04 2025

@author: schro
"""


#TESTTESTTESTTEST
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.stats import binom
import sys
import FunctionsForGeneralized as ffg
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm
import pytensor.tensor as pt  # Use pytensor instead of aesara
from arviz.stats.density_utils import kde
from High3sLogFuncs import HighLogAnalysis

with open("psych_vecs_all.pkl", "rb") as f:
    data = pickle.load(f)  
    
with open("session_summary.pkl", "rb") as f:
    sess_sum = pickle.load(f)  
    

Cagg_mats = []
Nagg_mats = []
data_dict = {}

for grp in ['ld','ln','rd','rn']:
    data_dict[grp] = {}
    C = np.array([*data['NY'][grp].values()])
    data_dict[grp]['C_mat'] = C
    Cagg_mats.append(sum(C))
    
    N = np.array([*data['N'][grp].values()])
    data_dict[grp]['N_mat'] = N
    Nagg_mats.append(sum(N))

data_dict['agg'] = {}
data_dict['agg']['C_mat'] = np.array(Cagg_mats)
data_dict['agg']['N_mat'] = np.array(Nagg_mats)

trace_agg = HighLogAnalysis(data_dict['agg'])
traces = {}
traces['agg'] = trace_agg

grps = ['ld','ln','rd','rn']

for grp in grps:
    traces[grp] = HighLogAnalysis(data_dict[grp])


with open('H3sL_traces.pkl', 'wb') as file:
    pickle.dump(traces, file)



for grp in grps:
    #print(az.summary(traces[grp]))
    az.plot_posterior(traces[grp],var_names = 'PSE',coords={'PSE_dim_0': [1]})


#forestplots
for descrip in ['gamma_h','gamma_l','PSE','JND']:
    # Create a figure with two side-by-side subplots
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6,12), sharey=True)
    
    # Plot forest on left axis
    az.plot_forest(
        [traces['ld'], traces['ln']],
        var_names=descrip,
        combined=True,
        colors=['red', 'blue'],
        model_names=['distracted', 'not distracted'],
        ax=ax_left
    )
    ax_left.set_title("Left Hand", fontsize=14)
    
    # Plot forest on right axis
    az.plot_forest(
        [traces['rd'], traces['rn']],
        var_names=descrip,
        combined=True,
        colors=['red', 'blue'],
        model_names=['distracted', 'not distracted'],
        ax=ax_right
    )
    ax_right.set_title("Right Hand", fontsize=14)
    
    # --- Sync x-axis limits ---
    # Get current limits from both
    xlim_left = ax_left.get_xlim()
    xlim_right = ax_right.get_xlim()
    
    # Compute the global min/max
    x_min = min(xlim_left[0], xlim_right[0])
    x_max = max(xlim_left[1], xlim_right[1])
    
    # Apply to both
    ax_left.set_xlim(x_min, x_max)
    ax_right.set_xlim(x_min, x_max)

    
    # Add a shared title above both plots
    fig.suptitle(f"Posterior {descrip} by Session", fontsize=18)
    plt.tight_layout()
    plt.savefig(descrip+'3HighLog.png')

    # Adjust spacing
    plt.show()
    

# #for descrip in ['gamma_h','gamma_l','PSE','JND']:
# descrip = 'PSE'

# ld_all = traces['agg'].posterior['PSE'].stack(sample=("chain", "draw")).values
# ln_all = traces['agg'].posterior['PSE'].stack(sample=("chain", "draw")).values



# # _all: 1D array of posterior samples
# x_ld, y_ld = kde(ld_all)
# x_ln, y_ln = kde(ln_all)


# plt.plot(x_ld, y_ld, alpha = .80, color='red')
# plt.plot(x_ln, y_ln, alpha = .80, color='blue')

# #plt.title(r"Hierarchal Model Recovery of $\beta_0$")
# #plt.legend()
# #plt.savefig('ParamBeta0RecovHighLog.png')
# plt.show()
sessions = np.array(sess_sum.index)
xfit = np.linspace(6,50,1000)

def get_curve_HDR(traces,grp,sess):
    sums_d = [0,0,0,0]
    rec_params = [0,0,0,0]
    for i, varn in enumerate(['gamma_h','gamma_l','beta0','beta1']):
        sums_d[i] = az.summary(traces[grp], var_names = varn, coords={varn+'_dim_0': sess})
        rec_params[i] = sums_d[i]["mean"]
    rec_params = np.array(rec_params)
    
    ## Construct HDI Curves
    b0_samples = traces[grp].posterior['beta0'][:,:,sess].values.flatten()
    b1_samples = traces[grp].posterior['beta1'][:,:,sess].values.flatten()
    gam_samples = traces[grp].posterior['gamma_h'][:,:,sess].values.flatten()
    lam_samples = traces[grp].posterior['gamma_l'][:,:,sess].values.flatten()
    
    y_samples = np.array([ffg.phi_L([gam,lam,b0,b1], xfit) for gam, lam, b0, b1 in zip(gam_samples, lam_samples, b0_samples, b1_samples)])
    hdi = az.hdi(y_samples, hdi_prob=0.95)
    yrec = ffg.phi_L(rec_params,xfit)
    return yrec, hdi
#'','','rd','rn'

x_stim = [6, 12, 18, 24, 32, 38, 44, 50]


for sess in [2,39]:
    dat_ld = data_dict['ld']['C_mat'][sess]/data_dict['ld']['N_mat'][sess]
    dat_ln = data_dict['ln']['C_mat'][sess]/data_dict['ln']['N_mat'][sess]
    dat_rd = data_dict['rd']['C_mat'][sess]/data_dict['rd']['N_mat'][sess]
    dat_rn = data_dict['ln']['C_mat'][sess]/data_dict['ln']['N_mat'][sess]
    
    yrec_ld, hdi_ld = get_curve_HDR(traces,'ld',sess)
    yrec_ln, hdi_ln = get_curve_HDR(traces,'ln',sess)
    yrec_rd, hdi_rd = get_curve_HDR(traces,'rd',sess)
    yrec_rn, hdi_rn = get_curve_HDR(traces,'rn',sess)
    
    
    
    # Create a 1-row, 2-column figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # --- LEFT HAND PLOT ---
    ax_left.plot(xfit, yrec_ld, color='red', label='distracted')
    ax_left.plot(xfit, yrec_ln, color='blue', label='not distracted')
    ax_left.fill_between(xfit, hdi_ld[:, 0], hdi_ld[:, 1], color='red', alpha=0.3)
    ax_left.fill_between(xfit, hdi_ln[:, 0], hdi_ln[:, 1], color='blue', alpha=0.3)
    ax_left.scatter(x_stim, dat_ld, color='red', label='data')
    ax_left.scatter(x_stim, dat_ln, color='blue')
    ax_left.set_title('Left Hand')
    ax_left.set_xlabel('Stimulus Amplitude')
    ax_left.set_ylabel('Proportion "Right"')
    ax_left.legend()
    
    # --- RIGHT HAND PLOT ---
    ax_right.plot(xfit, yrec_rd, color='red', label='distracted')
    ax_right.plot(xfit, yrec_rn, color='blue', label='not distracted')
    ax_right.fill_between(xfit, hdi_rd[:, 0], hdi_rd[:, 1], color='red', alpha=0.3)
    ax_right.fill_between(xfit, hdi_rn[:, 0], hdi_rn[:, 1], color='blue', alpha=0.3)
    ax_right.scatter(x_stim, dat_rd, color='red', label='data')
    ax_right.scatter(x_stim, dat_rn, color='blue')
    ax_right.set_title('Right Hand')
    ax_right.set_xlabel('Stimulus Amplitude')
    ax_right.legend()
    
    # Adjust layout and add a global title
    fig.suptitle(f"Psychometric Function Fits, Session {sessions[sess][0:5]}", fontsize=16)
    plt.savefig('Sessfit'+sessions[sess][0:2]+sessions[sess][3:5]+'HighLog.png')
    plt.show()
    
    

    



