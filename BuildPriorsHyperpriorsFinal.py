# -*- coding: utf-8 -*-
"""
Created on Sat May 17 22:18:18 2025

@author: schro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import beta
import arviz as az
import ast
import pandas as pd
from scipy.stats import truncnorm

def find_gamma_params(lik_range, c, alpha_range=(1.00, 100), beta_range=(0.0001, 100), resolution=200):
    [l,u] = lik_range

    alpha_vals = np.linspace(*alpha_range, resolution)
    beta_vals = np.linspace(*beta_range, resolution)
    A, B = np.meshgrid(alpha_vals, beta_vals)
    
    # Flatten for vectorized computation
    alpha_flat = A.ravel()
    beta_flat = B.ravel()
    
    # Calculate scale = 1 / rate
    scale_flat = 1 / beta_flat
    
    # Compute CDF values
    F_l = gamma.cdf(l, a=alpha_flat, scale=scale_flat)
    F_u = gamma.cdf(u, a=alpha_flat, scale=scale_flat)
    
    mu = alpha_flat/beta_flat
    sig = np.sqrt(alpha_flat/beta_flat**2)

    # Apply the three conditions
    cond1 = ((mu<=u) & (mu>=l))
    cond2 = ((u-mu)/sig >=0.25)
    cond3 = (abs(c-mu)<=2*sig)
    cond4 = (F_u<1)
    cond5 = (F_l>0)
    
    print(sum(cond1 & cond4))
    print(sum(cond2 & cond4))
    print(sum(cond3 & cond4))
    print(sum(cond4))
    print(sum(cond5 & cond4))


    # Combine all conditions
    valid = cond1 & cond2 & cond3 & cond4 & cond5

    valid_alpha = alpha_flat[valid]
    valid_beta = beta_flat[valid]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(valid_alpha, valid_beta, s=10, alpha=0.7, color='blue')
    plt.xlabel("Alpha (shape)")
    plt.ylabel("Beta (rate)")
    plt.title("Feasible (alpha, beta) pairs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # print(valid_alpha,valid_beta)

    return list(zip(valid_alpha, valid_beta))

def find_beta_params(lik_range, c, alpha_range=(1.0001, 60.), beta_range=(1.0001, 60.), resolution = 400):
    
    [l,u] = lik_range
    alpha_vals = np.linspace(*alpha_range, resolution)
    beta_vals = np.linspace(*beta_range, resolution)
    A, B = np.meshgrid(alpha_vals, beta_vals)
    
    # Flatten for vectorized computation
    alpha_flat = A.ravel()
    beta_flat = B.ravel()
    mu = (alpha_flat)/(alpha_flat+beta_flat)
    mode = (alpha_flat-1)/(alpha_flat+beta_flat-2)
    std = np.sqrt(alpha_flat*beta_flat/( (alpha_flat+beta_flat)**2 * (alpha_flat+beta_flat+1) ) )
    
    F_u = beta.cdf(u, a=alpha_flat, b=beta_flat)
    
    cond1 = ((std>=0.01) & (std<0.15))
    cond2 = (( abs(c-mu)/std<1 )|( abs(c-mode)/std<1 ))
    cond3 = ((F_u<0.95) & (F_u>0.5))
    cond4 = ((mu<u)|(mode<u))
    cond5 = ((mu>=l)|(mode<=l))
    
    print(sum(cond1))
    print(sum(cond2))
    print(sum(cond3))
    print(sum(cond4))
    print(sum(cond5))
    
    
    valid = cond1 & cond2 & cond3 & cond4 & cond5
    valid_alpha = alpha_flat[valid]
    valid_beta = beta_flat[valid]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(valid_alpha, valid_beta, s=10, alpha=0.7, color='blue')
    plt.xlabel("Alpha (shape)")
    plt.ylabel("Beta (rate)")
    plt.title("Feasible (alpha, beta) pairs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # print(valid_alpha,valid_beta)

    return list(zip(valid_alpha,valid_beta))

def find_tnorm_hps(lik_range, c, resolution = 200):
    
    [l,u] = lik_range
    mu_vals = np.linspace(l,u, resolution)
    sig_vals = np.linspace(0,(u-l), resolution)
    MU, SIG = np.meshgrid(mu_vals, sig_vals)
    
    # Flatten for vectorized computation
    mu_flat = MU.ravel()
    sig_flat = SIG.ravel()
    zn1 = mu_flat-sig_flat
    zp1 = mu_flat+sig_flat

    cond1 = ((zn1<=c)&(zp1>=c))
    cond2 = (zn1>=l)
    cond3 = (zp1<=u)
    cond4 = (l>mu_flat-3*sig_flat)
    cond5 = (u<mu_flat+3*sig_flat)

    
    print(sum(cond1))
    print(sum(cond2))
    print(sum(cond3))
    print(sum(cond4))
    print(sum(cond5))
    
    
    valid = cond1 & cond2 & cond3 & cond4 & cond5
    valid_mu = mu_flat[valid]
    valid_sig = sig_flat[valid]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(valid_mu, valid_sig, s=10, alpha=0.7, color='blue')
    plt.xlabel("Mu")
    plt.ylabel("Sigma")
    plt.title("Feasible (mu,sigma) pairs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # print(valid_alpha,valid_beta)

    return list(zip(valid_mu,valid_sig))


ximins = np.full(8,0.)
ximaxs = np.full(8,0.)
xicens = np.full(8,0.)

#lapses

range_gam = np.array([0.0001,0.4])
cent_gam = 0.16
attempt1gam = find_beta_params(range_gam, cent_gam, resolution=200)
attempt2gam = find_beta_params(range_gam, cent_gam, alpha_range=(1.0001, 10), beta_range=(1.0001, 30), resolution=400)
pairsgam = np.array(attempt2gam)

ximins[0] = min(pairsgam[:,0])
ximins[1] = min(pairsgam[:,1])
ximaxs[0] = max(pairsgam[:,0])
ximaxs[1] = max(pairsgam[:,1])
xicens[0] = np.median(pairsgam[:,0])
xicens[1] = np.median(pairsgam[:,1])

ximins[2] = ximins[0]
ximins[3] = ximins[1]
ximaxs[2] = ximaxs[0]
ximaxs[3] = ximaxs[1]
xicens[2] = xicens[0]
xicens[3] = xicens[1]

#beta0

range_negbeta0 = np.array([0.7,90])
cent_negbeta0 = 2.9
attempt1bet0 = find_gamma_params(range_negbeta0, cent_negbeta0, alpha_range=(1.00, 100), beta_range=(0.0001, 100), resolution=400)
attempt2bet0 = find_gamma_params(range_negbeta0, cent_negbeta0, alpha_range=(1.00, 10), beta_range=(0.0001, 1), resolution=400)
pairsbet0 = np.array(attempt2bet0)

ximins[4] = min(pairsbet0[:,0])
ximins[5] = min(pairsbet0[:,1])
ximaxs[4] = max(pairsbet0[:,0])
ximaxs[5] = max(pairsbet0[:,1])
xicens[4] = np.median(pairsbet0[:,0])
xicens[5] = np.median(pairsbet0[:,1])

#beta1

range_beta1 = np.array([0.02,1.9])
cent_beta1 = 0.104
attempt1bet1 = find_gamma_params(range_beta1, cent_beta1, alpha_range=(1.00, 100), beta_range=(0.0001, 100), resolution=400)
attempt2bet1 = find_gamma_params(range_beta1, cent_beta1, alpha_range=(1.00, 10), beta_range=(0.0001, 40), resolution=400)
pairsbet1 = np.array(attempt2bet1)

ximins[6] = min(pairsbet1[:,0])
ximins[7] = min(pairsbet1[:,1])
ximaxs[6] = max(pairsbet1[:,0])
ximaxs[7] = max(pairsbet1[:,1])
xicens[6] = np.median(pairsbet1[:,0])
xicens[7] = np.median(pairsbet1[:,1])

#hyperpriors
mu_xi = np.full(8,0.)
sig_xi = np.full(8,0.)

for i in np.arange(8):
    hppairs = find_tnorm_hps(lik_range = np.array([ximins[i],ximaxs[i]]), c = xicens[i])
    musigpairs = np.array(hppairs)
    mu_xi[i] = np.median(musigpairs[:,0])
    sig_xi[i] = np.median(musigpairs[:,1])
    
print(mu_xi)
print(sig_xi)

hhps_mvtn = np.vstack([mu_xi,sig_xi])

np.save('hhps_prior_mv_trunc_norm.npy', hhps_mvtn)

# Load the array from the .npy file
#loaded_arr = np.load('hhps_prior_mv_trunc_norm.npy')
