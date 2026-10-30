#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

Summary statistics and thresholds for Model D (Steps 3 and 5 of the workflow).

The same statistics serve the prior predictive checks and, later, the posterior
retrodictive checks. They are functions of the responses and the design only,
so they don't depend on any model's assumptions. The session PSE is the
Spearman-Karber estimate: the mean of the distribution function traced by the
response proportions, with the mass below 6 um placed at 6 um and the mass
above 50 um at 50 um.

Thresholds mark where values become extreme for a trained monkey in this design.
They are approximate, in the sense of the workflow: order-of-magnitude limits
from the meaning of the quantities and the stimulus range, not from any data or
fit. A prior is consistent with one if 0.1%-5% of its mass lies beyond it.

Statistics without thresholds cover behavior we expect to be hard to model.
Hands are independent and trials exchangeable in Model D, so these are mainly
for the retrodictive checks.

@author: vmschroe
"""

import numpy as np

STIM_RANGE = (6, 50)       # stimulus amplitudes (um)

# Prior pushforward thresholds (Step 5): (lo, hi); None = no limit on that side
PUSHFORWARD_THRESHOLDS = {
    'session PSE, unimanual (um)': STIM_RANGE,        # boundary outside the stimuli
    'session PSE, bimanual (um)': STIM_RANGE,
    'session JND (um)': (1, 22),                      # far sharper than tactile acuity; half the stimulus range
    'typical gam_h': (None, 0.15),                    # a trained animal lapses rarely
    'typical gam_l': (None, 0.15),
    'session gam_h': (None, 0.15),
    'session gam_l': (None, 0.15),
    'typical shift at 6 um (um)': (-11, 11),          # a quarter of the stimulus range, from the weakest distractor
    'typical shift at 32 um (um)': (-22, 22),         # half the stimulus range: the boundary leaves the stimuli
    'between-session sd of PSE (um)': (None, 11),     # a quarter of the stimulus range, day to day
}

# Prior predictive thresholds (Step 3)
PREDICTIVE_THRESHOLDS = {
    'accuracy': (0.6, 0.98),          # near chance, or better than the hard stimuli allow
    'p_high_6um': (None, 0.3),        # "high" on the weakest stimulus, any hand and condition
    'p_high_50um': (0.7, None),       # "high" on the strongest stimulus
    'mid_diff_bi_uni': (-0.5, 0.5),   # bi - uni in P(high) at the stimuli next to the boundary
    'sess_sd_mid': (None, 0.25),      # between-session sd of P(high) next to the boundary
}


def prepare(design):
    """Cell indices for summary_stats, from the arrays of ReadyData_Sirius_D.pkl."""
    stims = np.unique(design['stim_amp'])
    n_sess = len(design['sess_x_dist'])
    stim_idx = np.searchsorted(stims, design['stim_amp'])
    cell = (design['grp_idx'] * 2 + design['cond_idx']) * n_sess * len(stims) + design['sess_idx'] * len(stims) + stim_idx
    shape = (2, 2, n_sess, len(stims))
    n_cell = np.bincount(cell, minlength=np.prod(shape)).reshape(shape)
    assert (n_cell > 0).all()
    same_sess = design['sess_idx'][1:] == design['sess_idx'][:-1]
    frac = (design['trial_pos'] + 0.5) / np.bincount(design['sess_idx'])[design['sess_idx']]
    return {'stims': stims, 'cell': cell, 'shape': shape, 'n_cell': n_cell, 'same_sess': same_sess,
            'early': frac < 1 / 3, 'late': frac > 2 / 3, 'correct_side': (design['stim_amp'] > 28).astype(int),
            'high_right': design['high_right'] == 1, 'grp_idx': design['grp_idx'],
            'dist_um': design['sess_x_dist'] * design['x_sig'],
            'mid': np.isin(stims, [24, 32])}


def spearman_karber(p, stims):
    """Mean of the distribution function p along the last axis."""
    inner = (np.diff(p, axis=-1) * (stims[:-1] + stims[1:]) / 2).sum(axis=-1)
    return inner + p[..., 0] * stims[0] + (1 - p[..., -1]) * stims[-1]


def residuals(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (intercept + slope * x), intercept, slope


def corr(a, b):
    return np.nan if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]


def summary_stats(y, prep):
    """All summary statistics of one dataset y, as a flat dict of numbers."""
    stims, n_cell, mid, dist = prep['stims'], prep['n_cell'], prep['mid'], prep['dist_um']
    n_high = np.bincount(prep['cell'], weights=y, minlength=n_cell.size).reshape(prep['shape'])   # (hand, cond, sess, stim)
    p_cell = n_high / n_cell
    stats = {'accuracy': np.mean(y == prep['correct_side'])}

    # pooled over sessions: the psychometric curves and the two ends of each
    for i, hand in enumerate(['left', 'right']):
        for c, cond in enumerate(['uni', 'bi']):
            pooled = n_high[i, c].sum(axis=0) / n_cell[i, c].sum(axis=0)
            for s, stim in enumerate(stims):
                stats[f'p_high_{hand}_{cond}_{stim}'] = pooled[s]
            p_mid = n_high[i, c][:, mid].sum(axis=1) / n_cell[i, c][:, mid].sum(axis=1)
            stats[f'mid_{hand}_{cond}'] = p_mid.mean()
            stats[f'sess_sd_mid_{hand}_{cond}'] = p_mid.std(ddof=1)
        stats[f'mid_diff_{hand}'] = stats[f'mid_{hand}_bi'] - stats[f'mid_{hand}_uni']
    cells = [f'{hand}_{cond}' for hand in ['left', 'right'] for cond in ['uni', 'bi']]
    stats['p_high_6um'] = max(stats[f'p_high_{c}_6'] for c in cells)      # worst cell, as for the thresholds
    stats['p_high_50um'] = min(stats[f'p_high_{c}_50'] for c in cells)
    stats['mid_diff_bi_uni'] = max((stats[f'mid_diff_{h}'] for h in ['left', 'right']), key=abs)
    stats['sess_sd_mid'] = max(stats[f'sess_sd_mid_{c}'] for c in cells)

    # model-free PSE shift against distractor amplitude: the presence term is the intercept
    pse = spearman_karber(p_cell, stims.astype(float))    # (hand, cond, sess)
    resid_bi = {}
    for i, hand in enumerate(['left', 'right']):
        shift = pse[i, 1] - pse[i, 0]
        resid, intercept, slope = residuals(shift, dist)
        stats[f'shift_mean_{hand}'] = shift.mean()
        stats[f'shift_intercept_{hand}'] = intercept
        stats[f'shift_slope_{hand}'] = slope
        stats[f'shift_resid_sd_weak_{hand}'] = resid[dist <= 12].std(ddof=1)
        stats[f'shift_resid_sd_strong_{hand}'] = resid[dist >= 24].std(ddof=1)
        resid_bi[hand] = residuals(pse[i, 1], dist)[0]
        stats[f'corr_uni_bi_{hand}'] = corr(pse[i, 0], resid_bi[hand])

    # behavior the model leaves out
    stats['corr_hands_uni'] = corr(pse[0, 0], pse[1, 0])
    stats['corr_hands_bi'] = corr(resid_bi['left'], resid_bi['right'])
    stats['p_repeat'] = np.mean((y[1:] == y[:-1])[prep['same_sess']])
    correct = y == prep['correct_side']
    stats['accuracy_late_minus_early'] = correct[prep['late']].mean() - correct[prep['early']].mean()
    for i, hand in enumerate(['left', 'right']):
        trials = prep['grp_idx'] == i
        stats[f'target_side_bias_{hand}'] = y[trials & prep['high_right']].mean() - y[trials & ~prep['high_right']].mean()
    return stats
