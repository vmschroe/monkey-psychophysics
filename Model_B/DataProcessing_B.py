#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use DataProcessing_A and nonunified as a guide
Created on Tue Feb  3 17:59:51 2026

@author: vmschroe
"""


import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import pickle
import ast
import xarray as xr
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import norm

#%%

#load raw data
with open("Sirius_Data.pkl", "rb") as f:
    DataDict = pickle.load(f)  

# if doesn't load, os.getcwd()
#%% 
#########%%
## Format Experimental Data


Raw_DataFrame = DataDict['data']
stims = np.array(Raw_DataFrame['stim_amp'])

#normalize stim amps
x_old = np.unique(stims)
x_mu = np.mean(x_old)
x_sig = np.std(x_old)
stims_normed = ((stims-x_mu)/x_sig)

#construct covarate matrix
cov_mat = np.vstack([np.full_like(stims_normed, 1.0), stims_normed]).T

#hand/distraction group indices
grp_idx = np.array(Raw_DataFrame['group_idx'])
names_grp_idx = {'left_uni':0, 'left_bi':1, 'right_uni':2, 'right_bi':3}

#session date indices

sess_idx = np.array(Raw_DataFrame['sess_idx'])
dates_sess_idx = dict(zip( DataDict['session_summary'].index.tolist() , DataDict['session_summary']['sess_idx'].tolist() ))
sess_sum = DataDict['session_summary']

#binary responses
resp = np.array(Raw_DataFrame['response'])

#construct dictionary
ReadyData_Sirius_B = {'cov_mat':cov_mat,
                    'grp_idx':grp_idx, 
                    'sess_idx': sess_idx, 
                    'resp': resp, 
                    'names_grp_idx': names_grp_idx, 
                    'dates_sess_idx': dates_sess_idx, 
                    'session_summary': sess_sum}

#%% save processed data (choose active directory Model_B\\Data_B)

with open("ReadyData_Sirius_B.pkl","wb") as f:
    pickle.dump(ReadyData_Sirius_B, f)

#%% Generate Synthetic Dataset

# fix hyperparameters for synth dataset

pfixB= np.zeros((2,2,2,2,2))               
##fix parameters for each group
gam_h_mu =  np.array([0.03, 0.06, 0.06, 0.03]).reshape(2,2)
pfixB[:,:,0,0,0] = gam_h_mu
gam_l_mu =  np.array([0.005, 0.01, 0.005, 0.01]).reshape(2,2)
pfixB[:,:,0,1,0] = gam_l_mu
beta0_mu = np.array([-1, 1, -1, 1]).reshape(2,2)
pfixB[:,:,1,0,0] = beta0_mu
beta1_mu = np.array([3, 3, 6, 6]).reshape(2,2)
pfixB[:,:,1,1,0] = beta1_mu

gam_h_sig =  np.array([0.001, 0.002, 0.003, 0.005]).reshape(2,2)
pfixB[:,:,0,0,1] = gam_h_sig
gam_l_sig =  np.array([0.0005, 0.0001, 0.0004, 0.0002]).reshape(2,2)
pfixB[:,:,0,1,1] = gam_l_sig
beta0_sig = np.array([1, 0.5, 0.25, 0.75]).reshape(2,2)
pfixB[:,:,1,0,1] = beta0_sig
beta1_sig = np.array([0.006, 0.004, 0.008, 0.002]).reshape(2,2)
pfixB[:,:,1,1,1] = beta1_sig
#%% 
params_fixed_B = xr.DataArray(pfixB,
                            dims = ("hand", "manual", "par_type", "par_idx", "hp_type"),
                            coords = {"manual":["uni", "bi"], 
                                      "hand":["left", "right"], 
                                      "par_type":["gamma", "beta"], 
                                      "par_idx":["h0", "l1"], 
                                      "hp_type":["mu", "sig"]})
#%%
# -------------------------
# 1) Helper: sample (n_sessions, 2, 2) parameter arrays
# -------------------------
def sample_session_params(params_fixed_B, n_sessions=42):
    # index maps
    hand_levels = ["left", "right"]
    manual_levels = ["uni", "bi"]

    # allocate: (sess, hand, manual)
    beta0 = np.empty((n_sessions, 2, 2))
    beta1 = np.empty((n_sessions, 2, 2))
    gam_h = np.empty((n_sessions, 2, 2))
    gam_l = np.empty((n_sessions, 2, 2))

    for hi, h in enumerate(hand_levels):
        for mi, m in enumerate(manual_levels):
            # ---- betas: Normal
            mu_b0  = float(params_fixed_B.loc[h, m, "beta",  "h0", "mu"])
            sig_b0 = float(params_fixed_B.loc[h, m, "beta",  "h0", "sig"])
            beta0[:, hi, mi] = norm.rvs(loc=mu_b0, scale=sig_b0, size=n_sessions)

            mu_b1  = float(params_fixed_B.loc[h, m, "beta",  "l1", "mu"])
            sig_b1 = float(params_fixed_B.loc[h, m, "beta",  "l1", "sig"])
            beta1[:, hi, mi] = norm.rvs(loc=mu_b1, scale=sig_b1, size=n_sessions)

            # ---- gammas: Beta on [0, 0.25] via your "multiply by 4" trick
            # gamma_h
            mu_gh  = 4.0 * float(params_fixed_B.loc[h, m, "gamma", "h0", "mu"])
            sig_gh = 4.0 * float(params_fixed_B.loc[h, m, "gamma", "h0", "sig"])
            nu_gh  = (mu_gh * (1.0 - mu_gh) / (sig_gh**2)) - 1.0  # <-- fixed *
            a_gh   = mu_gh * nu_gh
            b_gh   = (1.0 - mu_gh) * nu_gh
            gam_h[:, hi, mi] = 0.25 * beta.rvs(a=a_gh, b=b_gh, size=n_sessions)

            # gamma_l
            mu_gl  = 4.0 * float(params_fixed_B.loc[h, m, "gamma", "l1", "mu"])
            sig_gl = 4.0 * float(params_fixed_B.loc[h, m, "gamma", "l1", "sig"])
            nu_gl  = (mu_gl * (1.0 - mu_gl) / (sig_gl**2)) - 1.0  # <-- fixed *
            a_gl   = mu_gl * nu_gl
            b_gl   = (1.0 - mu_gl) * nu_gl
            gam_l[:, hi, mi] = 0.25 * beta.rvs(a=a_gl, b=b_gl, size=n_sessions)

    return {"beta0": beta0, "beta1": beta1, "gam_h": gam_h, "gam_l": gam_l}

# -------------------------
# 2) Generate trial responses using sess_idx and grp_idx
# -------------------------
def synth_generator_hier_sessions(params_fixed_B, cov_mat, grp_idx, sess_idx, n_sessions=42):
    """
    cov_mat: (N,2)
    grp_idx: (N,) in {0,1,2,3} with ordering:
        0 left_uni, 1 left_bi, 2 right_uni, 3 right_bi
    sess_idx: (N,) in {0,...,n_sessions-1}
    """
    N = cov_mat.shape[0]
    assert cov_mat.shape[1] == 2
    assert len(grp_idx) == N and len(sess_idx) == N

    # sample session-level params
    sp = sample_session_params(params_fixed_B, n_sessions)
    beta0_sess = sp["beta0"]
    beta1_sess = sp["beta1"]
    gam_h_sess = sp["gam_h"]
    gam_l_sess = sp["gam_l"]

    # map grp_idx -> (hand_idx, manual_idx)
    # 0 left_uni, 1 left_bi, 2 right_uni, 3 right_bi
    hand_i   = grp_idx // 2          # 0 for left, 1 for right
    manual_i = grp_idx % 2           # 0 for uni,  1 for bi

    # pull per-trial parameters via advanced indexing
    b0 = beta0_sess[sess_idx, hand_i, manual_i]
    b1 = beta1_sess[sess_idx, hand_i, manual_i]
    gh = gam_h_sess[sess_idx, hand_i, manual_i]
    gl = gam_l_sess[sess_idx, hand_i, manual_i]

    # linear predictor and psychometric p
    log_arg = cov_mat[:, 0] * b0 + cov_mat[:, 1] * b1
    psi = gh + (1.0 - gh - gl) / (1.0 + np.exp(-log_arg))

    synth_resp = binom.rvs(n=1, p=psi, size=N)
    return synth_resp, sp

# ---- usage:
# synth_resp, sess_params = synth_generator_hier_sessions(params_fixed_B, cov_mat, grp_idx, sess_idx, n_sessions=42)
# sess_params["beta0"].shape  # (42, 2, 2)


#%% 
synth_resp, sess_params = synth_generator_hier_sessions(params_fixed_B, cov_mat, grp_idx, sess_idx, n_sessions=42)


#%%


def format_sess_params_xr(sess_params):
    n_sessions = sess_params["beta0"].shape[0]

    # allocate array:
    # dims: session, hand, manual, par_type, par_idx
    arr = np.zeros((n_sessions, 2, 2, 2, 2))

    # par_type index: 0=gamma, 1=beta
    # par_idx index:  0=h0,    1=l1

    # gamma
    arr[:, :, :, 0, 0] = sess_params["gam_h"]
    arr[:, :, :, 0, 1] = sess_params["gam_l"]

    # beta
    arr[:, :, :, 1, 0] = sess_params["beta0"]
    arr[:, :, :, 1, 1] = sess_params["beta1"]

    sess_params_xr = xr.DataArray(
        arr,
        dims=("session", "hand", "manual", "par_type", "par_idx"),
        coords={
            "session": np.arange(n_sessions),
            "hand": ["left", "right"],
            "manual": ["uni", "bi"],
            "par_type": ["gamma", "beta"],
            "par_idx": ["h0", "l1"],
        },
    )

    return sess_params_xr


sess_params_xr = format_sess_params_xr(sess_params)


#%% Sanity checks

# -------------------------
# 1) Session-level parameter sanity checks: mean/sd vs hyperparameters
# -------------------------
hand_levels = ["left", "right"]
manual_levels = ["uni", "bi"]

def print_param_check(name, arr, par_type, par_idx):
    # arr shape: (n_sessions, 2, 2)
    print(f"\n=== {name} (session draws) ===")
    for hi, h in enumerate(hand_levels):
        for mi, m in enumerate(manual_levels):
            mu_target  = float(params_fixed_B.loc[h, m, par_type, par_idx, "mu"])
            sig_target = float(params_fixed_B.loc[h, m, par_type, par_idx, "sig"])
            vals = arr[:, hi, mi]
            print(
                f"{h:>5} {m:>3} | target mu={mu_target:.6g}, sig={sig_target:.6g} "
                f"|| draw mu={vals.mean():.6g}, sig={vals.std(ddof=1):.6g} "
                f"(min={vals.min():.6g}, max={vals.max():.6g})"
            )

# betas: targets are direct
print_param_check("beta0", sess_params["beta0"], "beta", "h0")
print_param_check("beta1", sess_params["beta1"], "beta", "l1")

# gammas: targets are on [0, 0.25], so direct comparison is fine
print_param_check("gamma_h", sess_params["gam_h"], "gamma", "h0")
print_param_check("gamma_l", sess_params["gam_l"], "gamma", "l1")


# -------------------------
# 2) Recompute psi (per trial) and check it is valid
# -------------------------
# map grp_idx -> (hand_idx, manual_idx)
hand_i   = grp_idx // 2
manual_i = grp_idx % 2

b0 = sess_params["beta0"][sess_idx, hand_i, manual_i]
b1 = sess_params["beta1"][sess_idx, hand_i, manual_i]
gh = sess_params["gam_h"][sess_idx, hand_i, manual_i]
gl = sess_params["gam_l"][sess_idx, hand_i, manual_i]

log_arg = cov_mat[:, 0] * b0 + cov_mat[:, 1] * b1
psi = gh + (1.0 - gh - gl) / (1.0 + np.exp(-log_arg))

print("\n=== Trial-level psi checks ===")
print("psi finite fraction:", np.isfinite(psi).mean())
print("psi min/max:", float(np.min(psi)), float(np.max(psi)))
print("Any psi<0?", bool(np.any(psi < 0)))
print("Any psi>1?", bool(np.any(psi > 1)))

# If you want a stricter bound check with tolerance:
tol = 1e-12
print("Any psi outside [0,1] beyond tol?", bool(np.any((psi < -tol) | (psi > 1+tol))))

# -------------------------
# 3) Response rate sanity checks by group and by session
# -------------------------
group_names = np.array(["left_uni", "left_bi", "right_uni", "right_bi"])

print("\n=== Response rates by group ===")
for g in range(4):
    mask = (grp_idx == g)
    print(f"{group_names[g]:>9} | N={mask.sum():5d}  mean(y)={synth_resp[mask].mean():.4f}  mean(psi)={psi[mask].mean():.4f}")

print("\n=== Response rates by session (summary) ===")
sess_means = np.array([synth_resp[sess_idx == s].mean() for s in range(42)])
print("min/median/max session mean(y):", float(sess_means.min()), float(np.median(sess_means)), float(sess_means.max()))
print("sessions with no trials (should be none):", np.where(np.isnan(sess_means))[0])


#%% 




ReadyData_Synth_B = {'cov_mat':cov_mat,
                    'grp_idx':grp_idx, 
                    'sess_idx': sess_idx, 
                    'resp': synth_resp, 
                    'names_grp_idx': names_grp_idx, 
                    'dates_sess_idx': dates_sess_idx, 
                    'session_summary': sess_sum,
                    'hparams_fixed_B': params_fixed_B,
                    'synth_session_params': sess_params_xr}


with open("ReadyData_Synth_B.pkl","wb") as f:
    pickle.dump(ReadyData_Synth_B, f)
