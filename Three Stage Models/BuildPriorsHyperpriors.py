#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:16:25 2025

@author: vmschroe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import arviz as az
import ast
import pandas as pd
from scipy.stats import truncnorm


def find_gamma_params(gen_range, lik_range, m, alpha_range=(1.00, 100), beta_range=(0.00, 100), resolution=200):
    [l,u] = lik_range
    [L,U] = gen_range
    m = m-L
    l = l-L
    u = u-L
    U = U-L
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
    F_U = gamma.cdf(U, a=alpha_flat, scale=scale_flat)

    # Apply the three conditions
    cond1 = (0.5 <= (F_u - F_l)) & ((F_u - F_l) <= 0.95)
    cond2 = (0.75 <= F_U) & (F_U <= 0.99)
    lhs = alpha_flat - 2*np.sqrt(alpha_flat) - 1
    rhs = alpha_flat - 2*np.sqrt(alpha_flat)
    cond3 = (lhs <= m * beta_flat) & (m * beta_flat < rhs)

    # Combine all conditions
    valid = cond1 & cond2 & cond3

    valid_alpha = alpha_flat[valid]
    valid_beta = beta_flat[valid]
    
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond1 & cond2

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: center close to mean and mode")  
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond1 & cond3

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: F_U in [0.75,0.999]")
            print(f"F_U is between {min(F_U[valid]):.2f} and {max(F_U[valid]):.2f}")
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond1

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: F_U in [0.75,0.999]")
            print(f"F_U is between {min(F_U[valid]):.2f} and {max(F_U[valid]):.2f}")
            print("AND")
            print("Dropped condition: center close to mean and mode")  
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond2 & cond3

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: F_u - F_l in [0.4,0.99]")
            print(f"F_u - F_l is between {min(F_u[valid]-F_l[valid]):.2f} and {max(F_u[valid]-F_l[valid]):.2f}")
    
    if len(valid_alpha) == 0:
        print("yikes!")

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

def extract_min_max_alpha_per_beta(pairs, beta_precision=4):
    """
    For each unique beta (rounded to precision), find the min and max alpha values.

    Args:
        pairs (list of tuples): (alpha, beta) pairs
        beta_precision (int): Decimal precision for beta grouping

    Returns:
        list of tuples: [(min_alpha, beta), (max_alpha, beta), ...]
    """
    from collections import defaultdict

    # Group alphas by rounded beta
    beta_groups = defaultdict(list)
    for alpha, beta in pairs:
        beta_rounded = round(beta, beta_precision)
        beta_groups[beta_rounded].append(alpha)

    # Extract min and max alpha for each beta
    result = []
    for beta_val, alpha_list in beta_groups.items():
        min_alpha = min(alpha_list)
        max_alpha = max(alpha_list)
        result.append((min_alpha, beta_val))
        result.append((max_alpha, beta_val))

    return sorted(result, key=lambda x: x[1])  # Sort by beta for readability


def plot_gamma_pdfs(alpha_beta_pairs, x_range=(0, 30), num_points=500):
    """
    Plot Gamma PDFs for each (alpha, beta) pair on the same plot.

    Args:
        alpha_beta_pairs (list of tuples): List of (alpha, beta) pairs.
        x_range (tuple): Range of x values to evaluate PDFs.
        num_points (int): Number of points to compute in x_range.
    """
    x = np.linspace(*x_range, num_points)
    plt.figure(figsize=(10, 6))

    for alpha, beta in alpha_beta_pairs:
        scale = 1 / beta  # since scipy uses scale = 1/rate
        y = gamma.pdf(x, a=alpha, scale=scale)
        plt.plot(x, y, label=f'α={alpha:.2f}, β={beta:.2f}', alpha=0.5)

    plt.title("Gamma PDFs for Different (α, β) Pairs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def gamma_get_params(gen_range, lik_range, m):
    print("General Idea")
    find_gamma_params(gen_range, lik_range, m)

    a_range = input("Please enter alpha_range = ( ? , ? ): ")
    b_range = input("Please enter beta_range = ( ? , ? ): ")

    # Convert string input to actual tuple
    alpha_range = ast.literal_eval(a_range)
    beta_range = ast.literal_eval(b_range)

    pair = find_gamma_params(gen_range, lik_range, m, alpha_range=alpha_range, beta_range=beta_range, resolution=200)
    pairs = np.array(pair)
    
    
    plot_pairs = extract_min_max_alpha_per_beta(pair)
    plot_gamma_pdfs(plot_pairs)
    
    alpha_m = np.mean(pairs[:,0])
    beta_m =  np.mean(pairs[:,1])
    means = [alpha_m, beta_m]
    
    alpha_lik = [min(pairs[:,0]), max(pairs[:,0])]
    beta_lik = [min(pairs[:,1]), max(pairs[:,1])]
    lik_ranges = [alpha_lik, beta_lik]
    
    alpha_gen = [1.00, 2*alpha_lik[1]]
    beta_gen = [0.00, 2*beta_lik[1]]
    gen_ranges = [alpha_gen, beta_gen]
    
    return  means, lik_ranges, gen_ranges


def trunc_norm_get_ranges(gen_range, lik_range, m, res = 100, strict = False, use_U = True):
    L, U = gen_range
    l, u = lik_range
    mu = np.linspace(l, u, num = res)
    l1 = abs(m-mu)
    l2 = 0.5*abs(u-mu)
    l3 = 0.5*abs(l-mu)
    sig_lower = np.max(np.stack([l1, l2, l3]), axis=0)
    
    u1 = abs(u-mu)
    u2 = abs(l-mu)
    u3 = 0.5*abs(L-mu)
    u4 = 0.5*abs(U-mu)
    
    if use_U == True:
        u_mat = np.stack([u1, u2, u3, u4])
    else:
        u_mat = np.stack([u1, u2, u3])
    if strict == True:
        sig_upper = np.min(u_mat, axis=0)
    else:
        sig_upper = np.max(u_mat, axis=0)
        
    if not np.all(sig_upper>sig_lower):
        print("Warning: not all mu's produced valid sigmas! Returning valid results only.")
        return np.stack([mu, sig_lower, sig_upper])[:,sig_upper>sig_lower]
    return np.stack([mu, sig_lower, sig_upper])


def trunc_norm_hypers(gen_range, lik_range, m, param_name = "param"):
    L, U = gen_range
    [mu, sig_lower, sig_upper] = trunc_norm_get_ranges(gen_range, lik_range, m, use_U = False)
    # Area under the shaded region
    area = np.trapz(sig_upper - sig_lower, mu)
    # x-coordinate of center of mass
    mean_mu = np.trapz(mu * (sig_upper - sig_lower), mu) / area
    # y-coordinate of center of mass
    mean_sigma = 0.5 * np.trapz(sig_upper**2 - sig_lower**2, mu) / area
    
    std_mu = np.std(mu)
    mean_sigma = mean_sigma
    l_mu = mean_mu-std_mu
    u_mu = mean_mu+std_mu
    
    u_sigma = max(np.interp(l_mu, mu, sig_upper),np.interp(u_mu, mu, sig_upper))
    l_sigma = max(np.interp(l_mu, mu, sig_lower),np.interp(u_mu, mu, sig_lower))
    
    std_sigma = min(mean_sigma, 0.5*abs(u_sigma-mean_sigma)+0.5*abs(l_sigma-mean_sigma))
    
    
    plt.plot(mu, sig_lower, label='Lower Bound')
    plt.plot(mu, sig_upper, label='Upper Bound')
    plt.vlines([mean_mu-std_mu, mean_mu+std_mu,], min(sig_lower), max(sig_upper), linestyles='dashed')
    plt.hlines([mean_sigma-std_sigma, mean_sigma+std_sigma,], min(mu), max(mu), linestyles='dashed')
    plt.scatter(mean_mu, mean_sigma, label = f"Center = ( {mean_mu:.2f} , {mean_sigma:.2f} )")
    plt.fill_between(mu, sig_lower, sig_upper, color='gray', alpha=0.3)
    
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.legend()
    plt.title(f'Ranges for hyperparameters of {param_name}')
    plt.grid(True)
    plt.show()
    print("-----------------------------------")
    print(f"{param_name} is distributed according to a truncated normal:")
    print(f"{param_name} ~ N[ mu , sigma ] for param > {L} ")
    print(f"-----mu ~ N[ {mean_mu:.2f} , {std_mu:.2f} ] for mu > {L} ")
    print(f"-----sigma ~ N[ {mean_sigma:.2f} , {std_sigma:.2f} ] for sigma > 0 ")
    print("-----------------------------------")
    d = {'hyper_param': ["mu_" + param_name, "sigma_" + param_name], 'mean_hp': [mean_mu, mean_sigma], 'std_hp': [std_mu, std_sigma], 'lower_truncation': [L, 0]}
    df = pd.DataFrame(data=d)
    return(df)

def my_tr_norm(mean,std,L,n):
    samps = truncnorm.rvs(a = (L - mean) / std, b = 1000000, loc = mean, scale = std, size = n)
    return(samps)


def find_beta_params(gen_range, lik_range, m, alpha_range=(1.0001, 38), beta_range=(1.0001, 60), resolution = 200):
    gen_range = np.array(gen_range)
    lik_range = np.array(lik_range)
    
    [L,U] = gen_range
    C = 1/(U-L)
    scaled_lik = C*(lik_range - L)
    [l,u] = scaled_lik
    alpha_vals = np.linspace(*alpha_range, resolution)
    beta_vals = np.linspace(*beta_range, resolution)
    A, B = np.meshgrid(alpha_vals, beta_vals)
    print(L,l,m,u,U)
    # Flatten for vectorized computation
    alpha_flat = A.ravel()
    beta_flat = B.ravel()
    
    mode = (alpha_flat-1)/(alpha_flat+beta_flat-2)
    std = np.sqrt(alpha_flat*beta_flat/( (alpha_flat+beta_flat)**2 * (alpha_flat+beta_flat+1) ) )
    
    cond0 = (abs(mode-m)<1*std) & (mode<u) & (mode>l)
    cond1 = (abs(mode-l)>1*std)
    cond2 = (abs(mode-l)<3*std)
    cond3 = (abs(mode-u)>1*std)
    cond4 = (abs(mode-u)<3*std)
    score = cond1.astype(int)+cond2.astype(int)+cond3.astype(int)+cond4.astype(int)
    
    valid = cond0 & (score==4)
    valid_alpha = alpha_flat[valid]
    valid_beta = beta_flat[valid]
    
    if len(valid_alpha) == 0:
        valid = cond0 & (score>=3)
        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        print("relaxed conditions to 4/5")
    if len(valid_alpha) == 0:
        valid = cond0 & (score>=2)
        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        print("relaxed conditions to 3/5")
    if len(valid_alpha) == 0:
        valid = cond0 & (score>=1)
        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        print("relaxed conditions to 2/5")
    if len(valid_alpha) == 0:
        valid = cond0
        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        print("relaxed conditions to 1/5")
    if len(valid_alpha) == 0:
        print("yikes!")

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


# 4*lapse ~ Gamma(alpha_lapse, beta_lapse)
#          hyper_param   mean_hp    std_hp  lower_truncation
# 0     mu_alpha_lapse  2.216435  0.404387                 1
# 1  sigma_alpha_lapse  0.809104  0.274531                 0

#         hyper_param   mean_hp    std_hp  lower_truncation
# 0     mu_beta_lapse  8.459625  3.117947                 1
# 1  sigma_beta_lapse  6.240195  2.119245                 0

# beta_lapse is distributed according to a truncated normal:
# beta_lapse ~ N[ mu , sigma ] for param > 1 
# -----mu ~ N[ 8.46 , 3.12 ] for mu > 1 
# -----sigma ~ N[ 6.24 , 2.12 ] for sigma > 0 


opt = np.array(find_beta_params([0,0.25], [0,0.1], 0.04, alpha_range = (1.0001,5), beta_range = (1.0001,20)))
np.mean(opt[:,1])
tnr = trunc_norm_get_ranges(gen_range = [1,5], lik_range = [min(opt[:,0]), max(opt[:,0])], m = np.mean(opt[:,0]), res = 100, strict = True, use_U = False)

trunc_norm_hypers(gen_range = [1,20], lik_range = [min(opt[:,1]), max(opt[:,1])], m = np.mean(opt[:,1]), param_name = "beta_lapse")
    
params_dict = {}

#For parameter beta0
param_name = "beta1"    
gen = [0,2]
lik = [0,1]
m = 0.1

# trunc_norm_hypers(gen, lik, m, param_name = param_name)

#alpha = shape, beta = rate

shape_name = 'shape_'+param_name
rate_name = 'rate_'+param_name
print("-----------------------------------")
print(f"Parameter {param_name} - {gen[0]} has Gamma distribution:")
print(f"{param_name} ~ Gamma[ alpha = {shape_name} , beta = {rate_name} ]")
print("-----------------------------------")
means_hp, lik_ranges_hp, gen_ranges_hp = gamma_get_params(gen, lik, m)

hp_info = []

for i, hp_name in enumerate([shape_name, rate_name]):
    df = trunc_norm_hypers(gen_ranges_hp[i], lik_ranges_hp[i], means_hp[i], param_name = hp_name)
    hp_info.append(df)
    
combined_df = pd.concat(hp_info, ignore_index=True)

params_dict[param_name] = combined_df


K = 1000
hp_samps = [[],[],[],[]]
for i, hp in enumerate(params_dict[param_name]['hyper_param']):
    hp_samps[i] = my_tr_norm(params_dict[param_name]['mean_hp'][i],params_dict[param_name]['std_hp'][i],params_dict[param_name]['lower_truncation'][i],n = K)
    
alpha_samps = my_tr_norm(hp_samps[0],hp_samps[1],1.00001, n=K)
beta_samps = my_tr_norm(hp_samps[2],hp_samps[3],0.00001, n=K)

param_samps = gamma.rvs(a = alpha_samps, scale = 1/beta_samps, size = 1000)