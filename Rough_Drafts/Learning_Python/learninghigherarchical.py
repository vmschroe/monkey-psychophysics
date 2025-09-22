# -*- coding: utf-8 -*-
# """
# Created on Fri Mar 14 20:53:06 2025

# @author: schro
# """

import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt

def simulate_coin_flips(num_coins = 10, num_flips = 20, xi = 0.7):
    
    # Step 2: Assign each of the 10 coins to be fair (p=0.5) or unfair (p=0.25)
    is_fair = np.random.rand(num_coins) < xi  # Boolean array: True -> fair, False -> unfair
    print(is_fair)
    # Step 3: Flip each coin 20 times
    thetas = np.where(is_fair, 0.5, 0.25)  # Assign probabilities based on fairness
    heads_counts = np.random.binomial(n=num_flips, p=thetas)  # Simulate 20 flips per coin

    return heads_counts


#  Step 1: Simulate Observed Data
#np.random.seed(45)  # For reproducibility
num_coins = 20  # Ensure this matches in both data and model
num_flips = 40
observed_heads = simulate_coin_flips(num_coins=num_coins, num_flips=num_flips)

#  Step 2: Bayesian Model
with pm.Model() as model:
    
    # Prior: xi ~ Uniform(0.5, 1)
    xi = pm.Uniform("xi", lower=0.5, upper=1)
    
    # Likelihood of a coin being fair (Bernoulli)
    is_fair = pm.Bernoulli("is_fair", p=xi, shape=num_coins)  # Matches `num_coins`
    
    # Likelihood of observing heads (Binomial)
    p_heads = pm.math.switch(is_fair, 0.5, 0.25)  # Fair (p=0.5), Unfair (p=0.25)
    observed = pm.Binomial("observed", n=num_flips, p=p_heads, observed=observed_heads)

    #  Step 3: MCMC Sampling
    trace = pm.sample(2000, return_inferencedata=True, cores=2)

#  Step 4: Plot Posterior Distribution
#az.plot_posterior(trace, var_names=["xi"])
plt.show()

#  Step 5: Print Summary Statistics
#print(az.summary(trace, var_names=["xi"]))
print(trace.posterior['is_fair'].stack(sample=("chain", "draw")).values.mean(axis=1))

