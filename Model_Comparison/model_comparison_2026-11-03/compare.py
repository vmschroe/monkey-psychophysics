"""Compare all fitted models: trial-level PSIS-LOO (as in Vera's reports) with trial-level and
session-clustered standard errors, leave-one-session-out CV, where the differences come from,
and what each model says about the distractor effect. Writes results/*.csv and results/summary.json."""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az
from pathlib import Path
from models import load_data, trial_p, ALL_MODELS, FAMILY, X_SIG, X_MU

FITS = Path(__file__).resolve().parent / 'fits'; RES = Path(__file__).resolve().parent / 'results'; RES.mkdir(exist_ok=True)
data = load_data()
models = [m for m in ALL_MODELS if (FITS / f'{m}.loo.pkl').exists()]
loo = {m: pickle.load(open(FITS / f'{m}.loo.pkl', 'rb')) for m in models}
diag = {m: json.load(open(FITS / f'{m}.diag.json')) for m in models}
li = {m: loo[m].loo_i.values for m in models}
sess = data['sess']; nS = len(data['sessions'])
sess_amp = np.round(data['sess_d'] * X_SIG).astype(int)
trial_amp = sess_amp[sess]
summary = {}

# ---------------- 1. trial-level PSIS-LOO --------------------------------------------------
best = max(models, key=lambda m: loo[m].elpd_loo)
rows = []
for m in models:
    d = li[m] - li[best]
    cl = pd.Series(d).groupby(sess).sum()
    rows.append({'model': m, 'elpd_loo': loo[m].elpd_loo, 'se': loo[m].se, 'p_loo': loo[m].p_loo,
                 'diff_vs_best': d.sum(), 'se_diff_trial': np.sqrt(len(d) * np.var(d)),
                 'se_diff_session': np.sqrt(nS * cl.var()) if m != best else 0.0,
                 'max_k': float(loo[m].pareto_k.max()),
                 'divergences': diag[m]['divergences'], 'max_r_hat': diag[m]['max_r_hat'],
                 'min_ess_bulk': diag[m]['min_ess_bulk'], 'minutes': diag[m]['minutes']})
trial_tab = pd.DataFrame(rows).sort_values('elpd_loo', ascending=False)
trial_tab['z_trial'] = trial_tab['diff_vs_best'] / trial_tab['se_diff_trial']
trial_tab['z_session'] = trial_tab['diff_vs_best'] / trial_tab['se_diff_session']
trial_tab.to_csv(RES / 'loo_trial.csv', index=False)
print("=== Trial-level PSIS-LOO (best:", best, ")")
print(trial_tab.round(2).to_string(index=False))
try:
    w = az.compare({m: loo[m] for m in models}, method='stacking')['weight']
    print("stacking weights:", w.round(3).to_dict())
    summary['stacking'] = w.round(3).to_dict()
except Exception as e:
    print("stacking failed:", e)

# pairwise differences against B (Vera's preferred model)
pair_rows = []
for m in models:
    if m == 'B':
        continue
    d = li[m] - li['B']
    cl = pd.Series(d).groupby(sess).sum()
    by_amp = pd.Series(d).groupby(pd.cut(trial_amp, [0, 6, 12, 22, 30, 32], labels=['6', '12', '16-22', '24-30', '32'])).sum()
    pair_rows.append({'model': m, 'diff_vs_B': d.sum(), 'se_trial': np.sqrt(len(d) * np.var(d)),
                      'se_session': np.sqrt(nS * cl.var()), **{f'dist {k}': v for k, v in by_amp.items()},
                      'weak stim (<=12) in 32um sessions': d[(trial_amp == 32) & (data['x'] * X_SIG + X_MU <= 12.5)].sum()})
pair_tab = pd.DataFrame(pair_rows).sort_values('diff_vs_B', ascending=False)
pair_tab.to_csv(RES / 'loo_vs_B_by_amplitude.csv', index=False)
print("\n=== elpd_loo(model) - elpd_loo(B), summed by the session's distractor amplitude")
print(pair_tab.round(1).to_string(index=False))

# ---------------- 2. leave-one-session-out ------------------------------------------------
loso = {}
for m in models:
    f = FITS / f'{m}.loso_s1.pkl'
    if f.exists():
        loso[m] = pickle.load(open(f, 'rb'))
if loso:
    ls = {m: loso[m]['loo'].loo_i.values for m in loso}
    best_s = max(loso, key=lambda m: ls[m].sum())
    rows = []
    for m in loso:
        d = ls[m] - ls[best_s]
        rows.append({'model': m, 'elpd_loso': ls[m].sum(), 'se': loso[m]['loo'].se, 'diff_vs_best': d.sum(),
                     'se_diff': np.sqrt(nS * np.var(d)), 'max_k': float(loso[m]['loo'].pareto_k.max()),
                     'n_k_gt_0.7': int((loso[m]['loo'].pareto_k > 0.7).sum()),
                     'min_is_ess': loso[m]['summary']['min_is_ess']})
    loso_tab = pd.DataFrame(rows).sort_values('elpd_loso', ascending=False)
    loso_tab['z'] = loso_tab['diff_vs_best'] / loso_tab['se_diff']
    loso_tab.to_csv(RES / 'loso.csv', index=False)
    print("\n=== Leave-one-session-out (best:", best_s, ")")
    print(loso_tab.round(2).to_string(index=False))
    rows = []
    for m in loso:
        if m == 'B':
            continue
        d = pd.Series(ls[m] - ls['B'])
        rows.append({'model': m, 'diff_vs_B': d.sum(), 'se': np.sqrt(nS * np.var(d)),
                     **d.groupby(pd.cut(sess_amp, [0, 6, 12, 22, 30, 32], labels=['6', '12', '16-22', '24-30', '32'])).sum().add_prefix('dist ').to_dict()})
    print("\n=== LOSO: elpd(model) - elpd(B) by distractor amplitude")
    print(pd.DataFrame(rows).sort_values('diff_vs_B', ascending=False).round(1).to_string(index=False))

json.dump(summary, open(RES / 'summary.json', 'w'), indent=1)
