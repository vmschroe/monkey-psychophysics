import pymc as pm
import numpy as np
import arviz as az

# Generate sample observed data
data = np.random.normal(loc=5, scale=2, size=100)

# Define the model
with pm.Model() as model:
    # Define priors for the parameters
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Define the likelihood
    likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    
    # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
    trace = pm.sample(500, return_inferencedata=True)

# Plot posterior distributions
az.plot_posterior(trace, var_names=["mu"])
az.plot_posterior(trace, var_names=["sigma"])

# Print a summary of the posterior
print(az.summary(trace, var_names=["mu", "sigma"]))
