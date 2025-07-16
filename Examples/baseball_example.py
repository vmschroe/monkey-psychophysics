# -*- coding: utf-8 -*-
"""
baesball example : partial pooling first attempt
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
#%% 2

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#%% 3

data = pd.read_csv(pm.get_data("efron-morris-75-data.tsv"), sep="\t")
at_bats, hits = data[["At-Bats", "Hits"]].to_numpy().T

#%% 4

N = len(hits)
player_names = data["FirstName"] + " " + data["LastName"]
coords = {"player_names": player_names.tolist()}

with pm.Model(coords=coords) as baseball_model:
    phi = pm.Uniform("phi", lower=0.0, upper=1.0)

    kappa_log = pm.Exponential("kappa_log", lam=1.5)
    kappa = pm.Deterministic("kappa", pt.exp(kappa_log))

    theta = pm.Beta("theta", alpha=phi * kappa, beta=(1.0 - phi) * kappa, dims="player_names")
    y = pm.Binomial("y", n=at_bats, p=theta, dims="player_names", observed=hits)

#%% 5
with baseball_model:
    theta_new = pm.Beta("theta_new", alpha=phi * kappa, beta=(1.0 - phi) * kappa)
    y_new = pm.Binomial("y_new", n=4, p=theta_new, observed=0)

#%% 6

pm.model_to_graphviz(baseball_model)


#%% 7
with baseball_model:
    idata = pm.sample(2000, tune=2000, chains=2, target_accept=0.95)

    # check convergence diagnostics
    assert all(az.rhat(idata) < 1.03)
#%% 8 yeah
az.plot_trace(idata, var_names=["phi", "kappa"]);

#%% 9
az.plot_forest(idata, var_names="theta");

#%% 10
az.plot_trace(idata, var_names=["theta_new"]);

#%% 11
