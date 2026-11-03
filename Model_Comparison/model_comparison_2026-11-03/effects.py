"""What each model says about the distractor: PSE and JND shifts (bimanual - unimanual, averaged over
the 42 sessions, found numerically on each model's own psychometric curves so that mixtures and
condition-specific lapses are handled alike), the key new parameters, and a posterior predictive check
of the responses that drive the model differences (weak-stimulus "high" and strong-stimulus "low"
answers on bimanual trials, by distractor amplitude)."""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az
from pathlib import Path
from models import load_data, trial_p, ALL_MODELS, FAMILY, X_SIG, X_MU

FITS = Path(__file__).resolve().parent / 'fits'; RES = Path(__file__).resolve().parent / 'results'; RES.mkdir(exist_ok=True)
data = load_data()
nS = len(data['sessions'])
sess_amp = np.round(data['sess_d'] * X_SIG).astype(int)
N_DRAWS = 1000
XGRID = np.linspace(-2.5, 2.5, 501)


def hdi_str(v, fmt='.2f'):
    v = np.asarray(v).ravel(); v = v[np.isfinite(v)]
    lo, hi = az.hdi(v, hdi_prob=0.95)
    return f"{v.mean():{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def crossing(P, level):
    """x where P (..., grid) first reaches level, by linear interpolation; NaN if never."""
    above = P >= level
    idx = above.argmax(-1)
    ok = above.any(-1) & (idx > 0)
    i1 = np.clip(idx, 1, len(XGRID) - 1); i0 = i1 - 1
    p0 = np.take_along_axis(P, i0[..., None], -1)[..., 0]; p1 = np.take_along_axis(P, i1[..., None], -1)[..., 0]
    xc = XGRID[i0] + (level - p0) / np.where(p1 > p0, p1 - p0, np.nan) * (XGRID[1] - XGRID[0])
    return np.where(ok, xc, np.nan)


def curves(name, post, draws, h, c):
    G = len(XGRID)
    s = np.repeat(np.arange(nS), G)
    fake = dict(x=np.tile(XGRID, nS), sess=s, hand=np.full(nS * G, h), c=np.full(nS * G, c),
                d=c * data['sess_d'][s], grp4=np.full(nS * G, 2 * h + c),
                dA={'grp_idx': np.full(nS * G, 2 * h + c)}, sess_d=data['sess_d'])
    return trial_p(name, post, fake, draws).reshape(len(draws), nS, G)


rows, par_rows, ppc_rows = [], [], []
models = [m for m in ALL_MODELS if (FITS / f'{m}.nc').exists()]
for m in models:
    post = az.from_netcdf(FITS / f'{m}.nc').posterior
    n_tot = post.sizes['chain'] * post.sizes['draw']
    draws = np.linspace(0, n_tot - 1, N_DRAWS).astype(int)
    res = {'model': m}
    for h, hand in enumerate(['left', 'right']):
        pse, jnd = {}, {}
        for c in (0, 1):
            P = curves(m, post, draws, h, c)
            pse[c] = crossing(P, 0.5) * X_SIG + X_MU
            jnd[c] = (crossing(P, 0.75) - crossing(P, 0.25)) / 2 * X_SIG
        res[f'{hand} PSE uni'] = hdi_str(np.nanmean(pse[0], 1))
        res[f'{hand} PSE shift'] = hdi_str(np.nanmean(pse[1] - pse[0], 1))
        res[f'{hand} JND shift'] = hdi_str(np.nanmean(jnd[1] - jnd[0], 1))
        # PSE shift at the weakest and strongest distractor (mean over those sessions)
        for a in (6, 32):
            res[f'{hand} PSE shift @{a}um'] = hdi_str(np.nanmean((pse[1] - pse[0])[:, sess_amp == a], 1))
    rows.append(res)

    # key parameters
    pr = {'model': m}
    def add(label, var, sel=None, fmt='.2f', fn=None):
        if var in post:
            da = post[var] if sel is None else post[var].sel(**sel)
            vals = da.values if fn is None else fn(da.values)
            pr[label] = hdi_str(vals, fmt)
    for hand in ['left', 'right']:
        if m not in ('A', 'B', 'B-wide'):
            add(f'mu_b2 {hand}', 'mu_betas', {'betas': 'b2', 'groups': hand})
            add(f'sig_b2 {hand}', 'sig_betas', {'betas': 'b2', 'groups': hand})
        add(f'mu_b3 {hand}', 'mu_b3', {'groups': hand})
        add(f'mu_db1 {hand}', 'mu_db1', {'groups': hand})
        add(f'kappa_h {hand}', 'kappa', {'lapses': 'gam_h', 'groups': hand})
        add(f'kappa_l {hand}', 'kappa', {'lapses': 'gam_l', 'groups': hand})
        add(f'pi {hand}', 'pi', {'groups': hand}, fmt='.3f')
    if m in ('B', 'B-wide'):
        for g in ['left_bi', 'right_bi']:
            add(f'sig_gam_h {g}', 'sig_gams', {'betas': 'b0', 'groups': g})
    par_rows.append(pr)

    # posterior predictive check of the weak-stimulus 'high' and strong-stimulus 'low' answers
    P = trial_p(m, post, data, draws)                      # (draws, trials)
    rng = np.random.default_rng(0)
    yrep = rng.random(P.shape) < P
    stim = np.round(data['x'] * X_SIG + X_MU).astype(int)
    amp_t = sess_amp[data['sess']]
    for hand_i, hand in enumerate(['left', 'right']):
        for label, sel_stim, resp in [('high @ stim<=12', stim <= 12, 1), ('low @ stim>=44', stim >= 44, 0)]:
            for lo, hi_, alab in [(0, 12, '6-12'), (13, 30, '16-30'), (31, 32, '32')]:
                mask = (data['hand'] == hand_i) & (data['c'] == 1) & sel_stim & (amp_t >= lo) & (amp_t <= hi_)
                obs = np.mean(data['y'][mask] == resp)
                rep = np.mean(yrep[:, mask] == resp, 1)
                q = np.quantile(rep, [0.025, 0.975])
                ppc_rows.append({'model': m, 'hand': hand, 'check': label, 'distractor': alab, 'n': int(mask.sum()),
                                 'observed': obs, 'pred_mean': rep.mean(), 'lo': q[0], 'hi': q[1],
                                 'inside': bool(q[0] <= obs <= q[1])})
    print('done', m, flush=True)

eff = pd.DataFrame(rows); eff.to_csv(RES / 'effects.csv', index=False)
par = pd.DataFrame(par_rows); par.to_csv(RES / 'parameters.csv', index=False)
ppc = pd.DataFrame(ppc_rows); ppc.to_csv(RES / 'ppc_lapses.csv', index=False)
pd.set_option('display.width', 250)
print("\n=== Distractor effect by model (posterior mean [95% HDI], um; averages over the 42 sessions)")
print(eff.set_index('model').T.to_string())
print("\n=== Key parameters")
print(par.set_index('model').T.dropna(how='all').to_string())
print("\n=== PPC: bimanual answers that the models disagree on (observed, predicted mean [95%], inside?)")
ppc['cell'] = ppc.apply(lambda r: f"{r.pred_mean:.3f} [{r.lo:.3f},{r.hi:.3f}]{'' if r.inside else ' *'}", axis=1)
obs = ppc.drop_duplicates(['hand', 'check', 'distractor']).set_index(['hand', 'check', 'distractor'])[['n', 'observed']]
print(obs.round(3).join(ppc.pivot_table(index=['hand', 'check', 'distractor'], columns='model', values='cell', aggfunc='first')).to_string())
print("inside count by model:", ppc.groupby('model')['inside'].sum().to_dict(), "of", ppc.groupby('model').size().iloc[0])
