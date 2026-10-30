#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

Model D: revises Model C so that the lapse rates depend on hand and condition,
pooled over sessions as in Model A, and the PSE shift isn't forced to 0 for a
weak distractor. See Model_D_Spec.txt for the reasoning behind the priors.

    psi = gam_h[c] + (1 - gam_h[c] - gam_l[c]) * invlogit( b0 + b1*x_stim + delta*x_bi )

with x_bi = 1 on bimanual trials and 0 on unimanual ones. Groups are the two
hands; b0, b1 and delta vary by session, with hyperparameters per hand.

The intended form is b0 + b1*x_stim + b2*x_dist + b3*x_bi, with all betas per
session. Each session has a single distractor amplitude x_dist, so its data can
only inform the sum delta = b3 + b2*x_dist. Integrating out that split, with b2
and b3 independent and normal across sessions, gives the same model:

    delta ~ N( mu_b3 + mu_b2 * x_dist , sqrt( sig_b3^2 + sig_b2^2 * x_dist^2 ) )

Expects x_stim, cond_idx, grp_idx, sess_idx, sess_x_dist, obs_data and sessions
to be defined (see DataProcessing_D.py).

@author: vmschroe
"""

import numpy as np
import pymc as pm


coords = {
    'groups': ['left', 'right'],
    'betas': ["b0", "b1"],
    'lapses': ["gam_h", "gam_l"],
    'conditions': ["uni", "bi"],
    'trials': range(len(obs_data)),
    'sessions': sessions}


with pm.Model(coords=coords) as model_D:
    x_stim_mut = pm.Data("x_stim", x_stim, dims=("trials",))
    cond_idx_mut = pm.Data("cond_idx", cond_idx, dims=("trials",))     # 0 = uni, 1 = bi, so also x_bi
    grp_idx_mut = pm.Data("grp_idx", grp_idx, dims=("trials",))
    sess_idx_mut = pm.Data("sess_idx", sess_idx, dims=("trials",))
    sess_x_dist_mut = pm.Data("sess_x_dist", sess_x_dist, dims=("sessions",))
    obs_data_mut = pm.Data("obs_data", obs_data, dims=("trials",))

    #HYPERPRIORS
    #Beta vector (mu_betas and sig_betas of b1 are on the log scale)
    #   Beta_0 ~N[mu_0,sig_0]
    #       mu_0 ~ N[ mu = 0 , sigma = 1 ]   (x = 0 is amplitude 28, the category boundary)
    #       sig_0 ~ HalfNormal[sig=1]
    #   log(Beta_1) ~N[mu_1,sig_1]   (lognormal, so slopes stay positive)
    #       mu_1 ~ N[ mu = log(4) , sigma = 0.5 ]   (typical slope: median 4, 95% in [1.5, 10.7])
    #       sig_1 ~ HalfNormal[sig=0.3]
    # REPARAMETRIZATION Normal: X ~ Normal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = sig*Z + mu
    # REPARAMETRIZATION HalfNormal: X ~ HalfNormal[ sig ]
    #   Z ~ HalfNormal[1]
    #   X = sig * Z
    #   (not abs(sig * Z) with Z ~ Normal[0,1]: then the sign of Z is unidentified and chains
    #   settle on opposite signs, giving false r_hat warnings)
    z_mu_b0 = pm.Normal('z_mu_b0', mu = 0, sigma = 1, dims = ('groups',))
    z_mu_b1 = pm.Normal('z_mu_b1', mu = 0, sigma = 1, dims = ('groups',))
    mu_betas = pm.Deterministic('mu_betas', pm.math.stack([ 1 * z_mu_b0 , np.log(4) + 0.5 * z_mu_b1 ], axis=0), dims = ("betas", "groups"))
    z_sig_b0 = pm.HalfNormal('z_sig_b0', sigma = 1, dims = ('groups',))
    z_sig_b1 = pm.HalfNormal('z_sig_b1', sigma = 1, dims = ('groups',))
    sig_betas = pm.Deterministic('sig_betas', pm.math.stack([ 1 * z_sig_b0 , 0.3 * z_sig_b1 ], axis=0), dims = ("betas", "groups"))
    z_betas = pm.Normal('z_betas', mu = 0, sigma = 1, dims = ("betas", "groups", 'sessions'))
    betas_latent = sig_betas[..., None] * z_betas + mu_betas[..., None]
    beta_vec = pm.Deterministic("beta_vec", pm.math.stack([ betas_latent[0] , pm.math.exp(betas_latent[1]) ], axis=0), dims=("betas", "groups", "sessions"),)

    #Delta (logit shift on bimanual trials), one per session
    #   delta ~ N[ mu_b3 + mu_b2 * x_dist , sqrt( sig_b3^2 + sig_b2^2 * x_dist^2 ) ]
    #       (x_dist = dist_amp / stim sd, at most ~2.2 for 32 um)
    #       mu_b3 ~ N[ mu = 0 , sigma = 1.5 ]   (either direction; new in Model D, so wide)
    #       sig_b3 ~ HalfNormal[sig=1]   (new in Model D, so wide)
    #       mu_b2 ~ N[ mu = 0 , sigma = 0.75 ]   (either direction)
    #       sig_b2 ~ HalfNormal[sig=0.5]
    #       (b2 at half the width of b3: at x_dist = 2, ~29 um, neither term is favored)
    z_mu_b3 = pm.Normal('z_mu_b3', mu = 0, sigma = 1, dims = ('groups',))
    z_mu_b2 = pm.Normal('z_mu_b2', mu = 0, sigma = 1, dims = ('groups',))
    mu_b3 = pm.Deterministic('mu_b3', 1.5 * z_mu_b3, dims = ('groups',))
    mu_b2 = pm.Deterministic('mu_b2', 0.75 * z_mu_b2, dims = ('groups',))
    z_sig_b3 = pm.HalfNormal('z_sig_b3', sigma = 1, dims = ('groups',))
    z_sig_b2 = pm.HalfNormal('z_sig_b2', sigma = 1, dims = ('groups',))
    sig_b3 = pm.Deterministic('sig_b3', 1 * z_sig_b3, dims = ('groups',))
    sig_b2 = pm.Deterministic('sig_b2', 0.5 * z_sig_b2, dims = ('groups',))
    z_delta = pm.Normal('z_delta', mu = 0, sigma = 1, dims = ('groups', 'sessions'))
    mean_delta = mu_b3[:, None] + mu_b2[:, None] * sess_x_dist_mut[None, :]
    sd_delta = pm.math.sqrt( sig_b3[:, None]**2 + (sig_b2[:, None] * sess_x_dist_mut[None, :])**2 )
    delta = pm.Deterministic('delta', mean_delta + sd_delta * z_delta, dims = ('groups', 'sessions'))


    #Gamma (gamma_h and gamma_l iid; per hand and condition, pooled over sessions as in Model A)
    #   gamma = 0.25 * w_gam
    #   w_gam ~ LogitNormal[ mu = -2.5 , sigma = 1.5 ]   (gamma: median 0.019, 95% in [0.001, 0.152])
    # REPARAMETRIZATION LogitNormal: X ~ LogitNormal[ mu , sig ]
    #   Z ~ Normal[0,1]
    #   X = 1 / ( 1+ exp (- (sig*Z + mu) ) )
    z_gams = pm.Normal("z_gams", mu=0, sigma=1, dims=("lapses", "groups", "conditions"))
    gams = pm.Deterministic("gams", 0.25 * pm.math.invlogit(1.5 * z_gams - 2.5), dims=("lapses", "groups", "conditions"),)

    gam_h = pm.Deterministic("gam_h", gams[0], dims = ('groups', 'conditions'))
    gam_l = pm.Deterministic("gam_l", gams[1], dims = ('groups', 'conditions'))

    # PSE and JND per condition: delta moves the PSE, and the lapse rates of the
    # condition enter both. Slopes are shared, so the JND only differs through lapses.
    shift = beta_vec[0][:, None, :] + pm.math.stack([pm.math.zeros_like(delta), delta], axis=1)     # (groups, conditions, sessions)
    PSE = pm.Deterministic("PSE", (-shift + pm.math.log( (1-2*gam_h) / (1-2*gam_l) )[..., None]) / beta_vec[1][:, None, :], dims=("groups", "conditions", "sessions"))
    JND = pm.Deterministic("JND", (pm.math.log( ((3-4*gam_h)*(3-4*gam_l)) / ((1-4*gam_h)*(1-4*gam_l)) ))[..., None] / (2*beta_vec[1][:, None, :]) , dims=("groups", "conditions", "sessions"))


    # Per-trial quantities are plain tensors, not pm.Deterministic: storing one value
    # per trial for every draw would add ~2 GB per variable to the trace
    logistic_arg = beta_vec[0][grp_idx_mut, sess_idx_mut] + beta_vec[1][grp_idx_mut, sess_idx_mut] * x_stim_mut + delta[grp_idx_mut, sess_idx_mut] * cond_idx_mut

    gam_h_trial = gam_h[grp_idx_mut, cond_idx_mut]
    gam_l_trial = gam_l[grp_idx_mut, cond_idx_mut]
    p = gam_h_trial + (1 - gam_h_trial - gam_l_trial)*pm.math.invlogit(logistic_arg)

    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-8,1-1e-8), observed=obs_data_mut, dims=('trials',))

print('model is built!')
