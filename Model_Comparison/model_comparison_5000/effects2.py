"""Distractor effects, key parameters and the lapse posterior predictive check for every model fitted with
4 x 5000 draws. PSE and JND are found numerically on each model's own psychometric curves.
Usage: effects2.py [fits dir]"""
import sys, json, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az
from models import load_data, posterior_arrays, p_from_arrays, ALL_MODELS, X_SIG, X_MU

FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
import os
RES = Path(os.environ.get('RESDIR', BASE / 'results5000')); RES.mkdir(exist_ok=True)
data = load_data()
nS = len(data['sessions'])
sess_amp = np.round(data['sess_d'] * X_SIG).astype(int)
N_DRAWS = 2000
XGRID = np.linspace(-2.5, 2.5, 501)
BINS = [(6, 6, '6'), (12, 12, '12'), (16, 22, '16-22'), (24, 30, '24-30'), (32, 32, '32')]


def hdi(v):
    v = np.asarray(v).ravel(); v = v[np.isfinite(v)]
    lo, hi = az.hdi(v, hdi_prob=0.95)
    return v.mean(), lo, hi


def crossing(P, level):
    above = P >= level
    idx = above.argmax(-1)
    ok = above.any(-1) & (idx > 0)
    i1 = np.clip(idx, 1, len(XGRID) - 1); i0 = i1 - 1
    p0 = np.take_along_axis(P, i0[..., None], -1)[..., 0]; p1 = np.take_along_axis(P, i1[..., None], -1)[..., 0]
    xc = XGRID[i0] + (level - p0) / np.where(p1 > p0, p1 - p0, np.nan) * (XGRID[1] - XGRID[0])
    return np.where(ok, xc, np.nan)


def curves(name, A, h, c):
    G = len(XGRID)
    s = np.repeat(np.arange(nS), G)
    fake = dict(x=np.tile(XGRID, nS), sess=s, hand=np.full(nS * G, h), c=np.full(nS * G, c),
                d=c * data['sess_d'][s], grp4=np.full(nS * G, 2 * h + c), dA={'grp_idx': np.full(nS * G, 2 * h + c)},
                sess_d=data['sess_d'])
    return p_from_arrays(name, A, fake).reshape(-1, nS, G)


eff_rows, par_rows, ppc_rows, sess_rows = [], [], [], []
models = [m for m in ALL_MODELS if (FITS / f'{m}.nc').exists()]
CACHE = RES / 'effects_cache'; CACHE.mkdir(exist_ok=True)
for m in models:
    cf = CACHE / f'{m}.pkl'
    if cf.exists() and cf.stat().st_mtime > (FITS / f'{m}.nc').stat().st_mtime:
        cached = pickle.load(open(cf, 'rb'))
        for lst, key in [(eff_rows, 'eff'), (par_rows, 'par'), (ppc_rows, 'ppc'), (sess_rows, 'sess')]:
            lst.extend(cached[key])
        continue
    n0 = {k: len(v) for k, v in [('eff', eff_rows), ('par', par_rows), ('ppc', ppc_rows), ('sess', sess_rows)]}
    post = az.from_netcdf(FITS / f'{m}.nc').posterior
    n_tot = post.sizes['chain'] * post.sizes['draw']
    draws = np.linspace(0, n_tot - 1, N_DRAWS).astype(int)
    A = posterior_arrays(post, draws)
    for h, hand in enumerate(['left', 'right']):
        pse, jnd = {}, {}
        for c in (0, 1):
            P = curves(m, A, h, c)
            pse[c] = crossing(P, 0.5) * X_SIG + X_MU
            jnd[c] = (crossing(P, 0.75) - crossing(P, 0.25)) / 2 * X_SIG
        shift = pse[1] - pse[0]
        for q, v in [('PSE uni', np.nanmean(pse[0], 1)), ('PSE shift', np.nanmean(shift, 1)),
                     ('JND shift', np.nanmean(jnd[1] - jnd[0], 1)),
                     ('PSE shift @6um', np.nanmean(shift[:, sess_amp == 6], 1)),
                     ('PSE shift @32um', np.nanmean(shift[:, sess_amp == 32], 1))]:
            mean, lo, hi = hdi(v)
            eff_rows.append({'model': m, 'hand': hand, 'quantity': q, 'mean': mean, 'lo': lo, 'hi': hi})
        for si in range(nS):
            sess_rows.append({'model': m, 'hand': hand, 'session': data['sessions'][si], 'dist_um': sess_amp[si],
                              'pse_uni': np.nanmean(pse[0][:, si]), 'pse_uni_lo': np.nanquantile(pse[0][:, si], 0.025),
                              'pse_uni_hi': np.nanquantile(pse[0][:, si], 0.975), 'pse_bi': np.nanmean(pse[1][:, si]),
                              'shift': np.nanmean(shift[:, si]), 'shift_lo': np.nanquantile(shift[:, si], 0.025),
                              'shift_hi': np.nanquantile(shift[:, si], 0.975)})
    # key parameters
    def add(label, var, sel):
        if var in post:
            mean, lo, hi = hdi(post[var].sel(**sel).values)
            par_rows.append({'model': m, 'parameter': label, 'mean': mean, 'lo': lo, 'hi': hi})
    for hand in ['left', 'right']:
        if 'mu_betas' in post and 'b2' in post['mu_betas'].betas.values:
            add(f'mu_b2 {hand}', 'mu_betas', {'betas': 'b2', 'groups': hand})
            add(f'sig_b2 {hand}', 'sig_betas', {'betas': 'b2', 'groups': hand})
        if m != 'A' and not m.startswith('B'):
            add(f'sig_b0 {hand}', 'sig_betas', {'betas': 'b0', 'groups': hand})
            add(f'sig_b1 {hand}', 'sig_betas', {'betas': 'b1', 'groups': hand})
            add(f'mu_b0 {hand}', 'mu_betas', {'betas': 'b0', 'groups': hand})
            add(f'mu_b1 {hand}', 'mu_betas', {'betas': 'b1', 'groups': hand})
        add(f'mu_b3 {hand}', 'mu_b3', {'groups': hand})
        add(f'mu_db1 {hand}', 'mu_db1', {'groups': hand})
        add(f'kappa_h {hand}', 'kappa', {'lapses': 'gam_h', 'groups': hand})
        add(f'kappa_l {hand}', 'kappa', {'lapses': 'gam_l', 'groups': hand})
        add(f'pi {hand}', 'pi', {'groups': hand})
        add(f'lam {hand}', 'lam', {'groups': hand})
        add(f'tau {hand}', 'tau', {'groups': hand})
    if 'sig_shared' in post:
        mean, lo, hi = hdi(post['sig_shared'].values)
        par_rows.append({'model': m, 'parameter': 'sig_shared', 'mean': mean, 'lo': lo, 'hi': hi})
    if m.startswith('B') and 'sig_gams' in post:
        for g in ['left_bi', 'right_bi']:
            add(f'sig_gam_h {g}', 'sig_gams', {'betas': 'b0', 'groups': g})
    # lapse posterior predictive check: bimanual weak-stimulus 'high' and strong-stimulus 'low' answers
    P = p_from_arrays(m, A, data)
    rng = np.random.default_rng(0)
    yrep = rng.random(P.shape) < P
    stim = np.round(data['x'] * X_SIG + X_MU).astype(int)
    amp_t = sess_amp[data['sess']]
    for h, hand in enumerate(['left', 'right']):
        for check, sel_stim, resp in [('high @ stim<=12', stim <= 12, 1), ('low @ stim>=44', stim >= 44, 0)]:
            for lo_a, hi_a, lab in BINS:
                mask = (data['hand'] == h) & (data['c'] == 1) & sel_stim & (amp_t >= lo_a) & (amp_t <= hi_a)
                obs = np.mean(data['y'][mask] == resp)
                rep = np.mean(yrep[:, mask] == resp, 1)
                q = np.quantile(rep, [0.025, 0.975])
                ppc_rows.append({'model': m, 'hand': hand, 'check': check, 'distractor': lab, 'n': int(mask.sum()),
                                 'k': int(np.sum(data['y'][mask] == resp)), 'observed': obs, 'pred_mean': rep.mean(),
                                 'lo': q[0], 'hi': q[1], 'inside': bool(q[0] <= obs <= q[1])})
    pickle.dump({k: v[n0[k]:] for k, v in [('eff', eff_rows), ('par', par_rows), ('ppc', ppc_rows), ('sess', sess_rows)]},
                open(cf, 'wb'))
    del A, P, yrep
    print('done', m, flush=True)

pd.DataFrame(eff_rows).to_csv(RES / 'effects.csv', index=False)
pd.DataFrame(par_rows).to_csv(RES / 'parameters.csv', index=False)
pd.DataFrame(ppc_rows).to_csv(RES / 'ppc_lapses.csv', index=False)
pd.DataFrame(sess_rows).to_csv(RES / 'session_shifts.csv', index=False)
ppc = pd.DataFrame(ppc_rows)
print("PPC cells inside 95% interval:", ppc.groupby('model')['inside'].sum().to_dict(), "of", ppc.groupby('model').size().iloc[0])
