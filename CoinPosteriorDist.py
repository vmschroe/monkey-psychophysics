#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:50:14 2024

@author: vmschroe
"""

import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

#Prior distribution parameters
alph = 250
bet = 250

# Generate sample observed data
N=100
p = 0.3
data = np.random.binomial(N, p, size=None)
 
# Define the model
with pm.Model() as model:
    # Define priors for the parameters
    theta = pm.Beta("theta", alpha=alph , beta=bet )
    
    # Define the likelihood
    likelihood = pm.Binomial("obs", n=N, p=theta, observed=data)
    
    # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
    trace = pm.sample(2000, return_inferencedata=True)
    
    
# # Plot prior distribution
x = np.linspace(0, 1, 100)
prior_vals = beta.pdf(x, alph, bet)  # Evaluate the Beta prior PDF
prior_mean = beta.mean(alph, bet)
hdi_bounds = az.hdi(beta.rvs(alph, bet, size=100000), hdi_prob=0.95)

# # Plot likelihood
theta_vals = np.linspace(0, 1, 100)
likelihood_vals = [binom.pmf(data, N, theta_val) for theta_val in theta_vals]  # Use binomial PMF
# # Find the mode of the likelihood (the theta with the highest likelihood value)
mode_index = np.argmax(likelihood_vals)
mode_theta = theta_vals[mode_index]
mode_likelihood = likelihood_vals[mode_index]

# Create the figure and axis for the subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot prior distribution
axs[0].plot(x, prior_vals, label="Prior", color="blue")
axs[0].plot(prior_mean, np.max(prior_vals), color="green", label=f"Prior Mean = {prior_mean:.2f}")
axs[0].axhline(y=-0.5, xmin=hdi_bounds[0], xmax=hdi_bounds[1], color="black", label=f"95% HDI = [{hdi_bounds[0]:.2f}, {hdi_bounds[1]:.2f}]")
axs[0].set_xlabel("Theta")
axs[0].set_ylabel("Density")
axs[0].set_title("Prior Distribution of Theta")
axs[0].legend()

# Plot likelihood
axs[1].plot(theta_vals, likelihood_vals, label="Likelihood")
axs[1].plot(mode_theta, mode_likelihood, "go", label=f"Mode = {mode_theta:.2f}")  # Add a point marker at the mode
axs[1].set_xlabel("Theta")
axs[1].set_ylabel("Likelihood")
axs[1].set_title(f"Likelihood Function of Theta (given Data: {N} trials, {data} heads)")
axs[1].legend()

# Plot posterior distribution
az.plot_posterior(trace, var_names=["theta"], ax=axs[2])
axs[2].set_title("Posterior Distribution of Theta")
axs[2].set_xlabel("Theta")
axs[2].set_ylabel("Density")

# Adjust layout to avoid overlap and display the plots
plt.tight_layout()
plt.show()


# Print a summary of the posterior
print(az.summary(trace, var_names=["theta"]))
