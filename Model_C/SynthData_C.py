#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 2026

Synthetic data for Model C, shared by DataProcessing_C.py and later recovery
studies.

True hyperparameters are derived from Model B's fit to the real data
(Results_B.pkl), so the synthetic data resemble the real data. Model B fits
uni and bi trials as separate groups and has no b2, so per hand:
    - mu_b0, sig_b0: Model B's unimanual hyperparameters (b0 at x_dist = 0);
    - mu_b1, sig_b1: average of Model B's uni and bi hyperparameters;
    - mu_b2: slope through the origin of the per-session bimanual shift in b0
      (b0_bi - b0_uni, posterior means) on the session's x_dist, which is
      what b2 * x_dist describes in Model C;
    - sig_b2: sd across sessions of (b0_bi - b0_uni) / x_dist;
    - mu_gam_h, mu_gam_l (logit scale): logit of 4x the mean lapse rate over
      all uni and bi session estimates, since Model C shares lapse rates
      between uni and bi trials;
    - sig_gam_h, sig_gam_l: average of Model B's uni and bi hyperparameters.
These true values only generate data; Model C's priors don't use them.

@author: vmschroe
"""

import numpy as np
import pandas as pd

GROUPS = ['left', 'right']
SESSION_PARAMS = ['b0', 'b1', 'b2', 'gam_h', 'gam_l']
HYPER_NAMES = ['mu_b0', 'sig_b0', 'mu_b1', 'sig_b1', 'mu_b2', 'sig_b2',
               'mu_gam_h', 'sig_gam_h', 'mu_gam_l', 'sig_gam_l']


def logit(p):
    return np.log(p / (1 - p))


def true_hyper_from_B(means_B, sessions, sess_x_dist):
    """True Model C hyperparameters from Model B's real-data fit.

    means_B: posterior means, Results_B.pkl['az_summary_trace']['mean'].
    sessions: session labels in session order; sess_x_dist: their x_dist.
    Returns a DataFrame, rows HYPER_NAMES, columns GROUPS; gamma ones on the logit scale."""
    def hyper(var, beta, grp):
        return means_B[f"{var}[{beta}, {grp}]"]

    def per_session(var, grp, beta=None):
        prefix = f"{beta}, " if beta else ""
        return np.array([means_B[f"{var}[{prefix}{grp}, {s}]"] for s in sessions])

    true_hyper = {}
    for hand in GROUPS:
        uni, bi = f"{hand}_uni", f"{hand}_bi"
        shift = per_session('beta_vec', bi, 'b0') - per_session('beta_vec', uni, 'b0')
        # Build_Model_B.py indexes the gamma hyperparameters with the 'betas' dim: b0 -> gam_h, b1 -> gam_l
        true_hyper[hand] = {
            'mu_b0': hyper('mu_betas', 'b0', uni),
            'sig_b0': hyper('sig_betas', 'b0', uni),
            'mu_b1': np.mean([hyper('mu_betas', 'b1', g) for g in (uni, bi)]),
            'sig_b1': np.mean([hyper('sig_betas', 'b1', g) for g in (uni, bi)]),
            'mu_b2': np.sum(shift * sess_x_dist) / np.sum(sess_x_dist**2),
            'sig_b2': np.std(shift / sess_x_dist, ddof=1),
            'mu_gam_h': logit(4 * np.mean([per_session('gam_h', g) for g in (uni, bi)])),
            'sig_gam_h': np.mean([hyper('sig_gams', 'b0', g) for g in (uni, bi)]),
            'mu_gam_l': logit(4 * np.mean([per_session('gam_l', g) for g in (uni, bi)])),
            'sig_gam_l': np.mean([hyper('sig_gams', 'b1', g) for g in (uni, bi)]),
        }
    return pd.DataFrame(true_hyper).loc[HYPER_NAMES, GROUPS]


def sample_session_params(true_hyper, n_sessions, rng):
    """Session-level parameters drawn from the hyperparameters, as Model C assumes.
    Returns {param: array (n_groups, n_sessions)} for SESSION_PARAMS."""
    shape = (len(GROUPS), n_sessions)
    hp = {name: true_hyper.loc[name, GROUPS].to_numpy()[:, None] for name in HYPER_NAMES}
    return {
        'b0': hp['mu_b0'] + hp['sig_b0'] * rng.standard_normal(shape),
        'b1': hp['mu_b1'] + hp['sig_b1'] * rng.standard_normal(shape),
        'b2': hp['mu_b2'] + hp['sig_b2'] * rng.standard_normal(shape),
        'gam_h': 0.25 / (1 + np.exp(-(hp['mu_gam_h'] + hp['sig_gam_h'] * rng.standard_normal(shape)))),
        'gam_l': 0.25 / (1 + np.exp(-(hp['mu_gam_l'] + hp['sig_gam_l'] * rng.standard_normal(shape)))),
    }


def response_probs(session_params, cov_mat, grp_idx, sess_idx):
    """Model C's psychometric function for every trial."""
    par = {p: session_params[p][grp_idx, sess_idx] for p in SESSION_PARAMS}
    logistic_arg = par['b0'] * cov_mat[:, 0] + par['b1'] * cov_mat[:, 1] + par['b2'] * cov_mat[:, 2]
    return par['gam_h'] + (1 - par['gam_h'] - par['gam_l']) / (1 + np.exp(-logistic_arg))


def simulate_resp(session_params, cov_mat, grp_idx, sess_idx, rng):
    return rng.binomial(1, response_probs(session_params, cov_mat, grp_idx, sess_idx))
