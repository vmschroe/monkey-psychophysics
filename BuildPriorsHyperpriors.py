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



def find_gamma_params(gen_range, lik_range, m, alpha_range=(1.001, 30), beta_range=(0.001, 30), resolution=200):
    [l,u] = lik_range
    [L,U] = gen_range
    m = m-L
    l = l-L
    u = u-L
    U = U-L
    print(type(alpha_range))
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
        valid = cond1 & cond3

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: F_U in [0.75,0.99]")
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond2 & cond3

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: F_u - F_l in [0.5,0.95]")        
    if len(valid_alpha) == 0:
        print("Oops! Gotta relax the conditions")
        valid = cond1 & cond2

        valid_alpha = alpha_flat[valid]
        valid_beta = beta_flat[valid]
        if len(valid_alpha) != 0:
            print("Dropped condition: center close to mean and mode")  
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


def babysit_get_params(gen_range, lik_range, m):
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
    
    alpha_gen = [1.001, 2*alpha_lik[1]]
    beta_gen = [0.001, 2*beta_lik[1]]
    gen_ranges = [alpha_gen, beta_gen]
    
    return  means, lik_ranges, gen_ranges

babysit_get_params([0,30],[0,15],2.8)


