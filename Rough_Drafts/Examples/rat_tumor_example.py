# -*- coding: utf-8 -*-
"""
Hierarchical Binomial Model: Rat Tumor Example

Created on Tue Jul 15 13:46:24 2025

@author: schro
"""
#%% 1 - packages

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from scipy.special import gammaln

#%% 2

#%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

#%% 3 - data load

# rat data (BDA3, p. 102)
# fmt: off
y = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  5,  2,
    5,  3,  2,  7,  7,  3,  3,  2,  9, 10,  4,  4,  4,  4,  4,  4,  4,
    10,  4,  4,  4,  5, 11, 12,  5,  5,  6,  5,  6,  6,  6,  6, 16, 15,
    15,  9,  4
])
n = np.array([
    20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 18, 18, 17, 20, 20, 20,
    20, 19, 19, 18, 18, 25, 24, 23, 20, 20, 20, 20, 20, 20, 10, 49, 19,
    46, 27, 17, 49, 47, 20, 20, 13, 48, 50, 20, 20, 20, 20, 20, 20, 20,
    48, 19, 19, 19, 22, 46, 49, 20, 20, 23, 19, 22, 20, 20, 20, 52, 46,
    47, 24, 14
])
# fmt: on

N = len(n)

#%% 4.1 - construct functions
## DIRECTLY COMPUTED SOLUTION
# Compute on log scale because products turn to sums
def log_likelihood(alpha, beta, y, n):
    LL = 0

    # Summing over data
    for Y, N in zip(y, n):
        LL += (
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + gammaln(alpha + Y)
            + gammaln(beta + N - Y)
            - gammaln(alpha + beta + N)
        )

    return LL


def log_prior(A, B):
    return -5 / 2 * np.log(A + B)


def trans_to_beta(x, y):
    return np.exp(y) / (np.exp(x) + 1)


def trans_to_alpha(x, y):
    return np.exp(x) * trans_to_beta(x, y)

#%% 4.2 construct dataframe

# Create space for the parameterization in which we wish to plot
X, Z = np.meshgrid(np.arange(-2.3, -1.3, 0.01), np.arange(1, 5, 0.01))
param_space = np.c_[X.ravel(), Z.ravel()]
df = pd.DataFrame(param_space, columns=["X", "Z"])

# Transform the space back to alpha beta to compute the log-posterior
df["alpha"] = trans_to_alpha(df.X, df.Z)
df["beta"] = trans_to_beta(df.X, df.Z)

df["log_posterior"] = log_prior(df.alpha, df.beta) + log_likelihood(df.alpha, df.beta, y, n)
df["log_jacobian"] = np.log(df.alpha) + np.log(df.beta)

df["transformed"] = df.log_posterior + df.log_jacobian
df["exp_trans"] = np.exp(df.transformed - df.transformed.max())

# This will ensure the density is normalized
df["normed_exp_trans"] = df.exp_trans / df.exp_trans.sum()


surface = df.set_index(["X", "Z"]).exp_trans.unstack().values.T



#%% 5 - posterior plot

fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(X, Z, surface)
ax.set_xlabel(r"$\log(\alpha/\beta)$", fontsize=16)
ax.set_ylabel(r"$\log(\alpha+\beta)$", fontsize=16)

ix_z, ix_x = np.unravel_index(np.argmax(surface, axis=None), surface.shape)
ax.scatter([X[0, ix_x]], [Z[ix_z, 0]], color="red")

text = r"$({a},{b})$".format(a=np.round(X[0, ix_x], 2), b=np.round(Z[ix_z, 0], 2))

ax.annotate(
    text,
    xy=(X[0, ix_x], Z[ix_z, 0]),
    xytext=(-1.6, 3.5),
    ha="center",
    fontsize=16,
    color="white",
    arrowprops={"facecolor": "white"},
);

#%% 6 - Estimated mean of alpha
(df.alpha * df.normed_exp_trans).sum().round(3)


#%% 7 - Estimated mean of beta
(df.beta * df.normed_exp_trans).sum().round(3)


#%% 8 - COMPUTED USING PYMC

coords = {
    "obs_id": np.arange(N),
    "param": ["alpha", "beta"],
}

#%% 9 - model

def logp_ab(value):
    """prior density"""
    return pt.log(pt.pow(pt.sum(value), -5 / 2))


with pm.Model(coords=coords) as model:
    # Uninformative prior for alpha and beta
    n_val = pm.ConstantData("n_val", n)
    ab = pm.HalfNormal("ab", sigma=10, dims="param")
    pm.Potential("p(a, b)", logp_ab(ab))

    X = pm.Deterministic("X", pt.log(ab[0] / ab[1]))
    Z = pm.Deterministic("Z", pt.log(pt.sum(ab)))

    theta = pm.Beta("theta", alpha=ab[0], beta=ab[1], dims="obs_id")

    p = pm.Binomial("y", p=theta, observed=y, n=n_val)
    trace = pm.sample(draws=2000, tune=2000, target_accept=0.95)

#%% 10 - check trace (good)

az.plot_trace(trace, var_names=["ab", "X", "Z"], compact=False);


#%% 11 - plot joint posterior

az.plot_pair(trace, var_names=["X", "Z"], kind="kde");

#%% 12 - plot marginal posteriors

az.plot_posterior(trace, var_names=["ab"]);

#%% 13 - posterior means

# estimate the means from the samples
trace.posterior["ab"].mean(("chain", "draw"))


