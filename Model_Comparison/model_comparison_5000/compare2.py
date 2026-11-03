"""Tables for the report from the 4 x 5000-draw fits: trial-level PSIS-LOO (SE over trials and clustered by
session), leave-one-session-out, sampling diagnostics, and the differences to Model B by distractor amplitude.
Usage: compare2.py [fits dir]"""
import sys, json, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import load_data, ALL_MODELS, X_SIG, X_MU

FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
import os
RES = Path(os.environ.get('RESDIR', BASE / 'results5000')); RES.mkdir(exist_ok=True)
data = load_data()
sess = data['sess']; nS = len(data['sessions'])
sess_amp = np.round(data['sess_d'] * X_SIG).astype(int)
trial_amp = sess_amp[sess]
AMP_BINS, AMP_LABELS = [0, 6, 12, 22, 30, 32], ['6', '12', '16-22', '24-30', '32']

def get_loo(m):
    r = pickle.load(open(FITS / f'{m}.loo.pkl', 'rb'))
    if isinstance(r, dict):
        return r
    return {'elpd_loo': r.elpd_loo, 'se': r.se, 'p_loo': r.p_loo, 'loo_i': r.loo_i.values, 'pareto_k': r.pareto_k.values}

models = [m for m in ALL_MODELS if (FITS / f'{m}.loo.pkl').exists()]
L = {m: get_loo(m) for m in models}
diag = {m: json.load(open(FITS / f'{m}.diag.json')) for m in models}
S = {}
for m in models:
    f = FITS / f'{m}.loso_s1.pkl'
    if f.exists():
        S[m] = pickle.load(open(f, 'rb'))
li = {m: L[m]['loo_i'] for m in models}
best = max(models, key=lambda m: L[m]['elpd_loo'])
ls = {m: S[m]['loo'].loo_i.values for m in S}
best_s = max(S, key=lambda m: ls[m].sum()) if S else None

rows = []
for m in models:
    d = li[m] - li[best]
    cl = pd.Series(d).groupby(sess).sum()
    r = {'model': m, 'elpd_loo': L[m]['elpd_loo'], 'se': L[m]['se'], 'p_loo': L[m]['p_loo'],
         'd_trial': d.sum(), 'se_d_trial': np.sqrt(len(d) * np.var(d)), 'se_d_trial_session': np.sqrt(nS * cl.var()) if m != best else 0.0,
         'max_k_trial': float(np.max(L[m]['pareto_k'])),
         'sampler': diag[m].get('sampler', 'pymc'), 'draws': diag[m].get('draws', 1000), 'divergences': diag[m]['divergences'],
         'max_r_hat': diag[m]['max_r_hat'], 'min_ess_bulk': diag[m]['min_ess_bulk'], 'minutes': diag[m]['minutes']}
    if m in S:
        ds_ = ls[m] - ls[best_s]
        r.update({'elpd_loso': ls[m].sum(), 'se_loso': S[m]['loo'].se, 'd_loso': ds_.sum(),
                  'se_d_loso': np.sqrt(nS * np.var(ds_, ddof=1)) if m != best_s else 0.0,
                  'max_k_loso': float(S[m]['loo'].pareto_k.max()), 'n_k_loso_gt_0.7': int((S[m]['loo'].pareto_k > 0.7).sum())})
    rows.append(r)
tab = pd.DataFrame(rows).sort_values('elpd_loo', ascending=False)
tab.to_csv(RES / 'comparison.csv', index=False)
pd.set_option('display.width', 250)
print(f"best trial-level: {best}; best LOSO: {best_s}")
print(tab[['model', 'p_loo', 'elpd_loo', 'd_trial', 'se_d_trial_session', 'elpd_loso', 'd_loso', 'se_d_loso', 'max_k_trial',
           'max_k_loso', 'n_k_loso_gt_0.7', 'divergences', 'max_r_hat', 'min_ess_bulk', 'minutes']].round(2).to_string(index=False))

# differences to Model B by the session's distractor amplitude (trial-level and LOSO)
rows = []
for m in models:
    if m == 'B':
        continue
    d = pd.Series(li[m] - li['B'])
    r = {'model': m, 'trial_total': d.sum(), 'trial_se_session': np.sqrt(nS * d.groupby(sess).sum().var())}
    r.update(d.groupby(pd.cut(trial_amp, AMP_BINS, labels=AMP_LABELS)).sum().add_prefix('trial ').to_dict())
    if m in S and 'B' in S:
        e = pd.Series(ls[m] - ls['B'])
        r.update({'loso_total': e.sum(), 'loso_se': np.sqrt(nS * np.var(e, ddof=1))})
        r.update(e.groupby(pd.cut(sess_amp, AMP_BINS, labels=AMP_LABELS)).sum().add_prefix('loso ').to_dict())
    rows.append(r)
vb = pd.DataFrame(rows).sort_values('trial_total', ascending=False)
vb.to_csv(RES / 'vs_B_by_amplitude.csv', index=False)
print("\nDifference to Model B, by the session's distractor amplitude")
print(vb.round(1).to_string(index=False))
