"""Targeted rerun of Model A's synthetic study at the real-data values (posterior means of Model A's
real fit, as used for 'data A' in Model_Recov_BA.py), on the real trial design: are the lapse rates
recoverable when they are as small as the real unimanual ones (gamma_h ~0.002)?"""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pymc as pm, arviz as az
from models import load_data, W

R = 6
data = load_data()
dA = data['dA']
means = pickle.load(open(W / 'Model_A/Results_A.pkl', 'rb'))['az_summary_trace']['mean']
groups = ['left_uni', 'left_bi', 'right_uni', 'right_bi']
true = {p: np.array([means[f'{v}[{b + ", " if b else ""}{g}]'] for g in groups])
        for p, v, b in [('b0', 'beta_vec', 'b0'), ('b1', 'beta_vec', 'b1'), ('gam_h', 'gam_h', None), ('gam_l', 'gam_l', None)]}
g, x = dA['grp_idx'], dA['cov_mat'][:, 1]
psi = true['gam_h'][g] + (1 - true['gam_h'][g] - true['gam_l'][g]) / (1 + np.exp(-(true['b0'][g] + true['b1'][g] * x)))
rng = np.random.default_rng(2026)
rows = []
src = (W / 'Model_A/Build_Model_A.py').read_text()
for r in range(R):
    y = rng.binomial(1, psi)
    ns = {'cov_mat': dA['cov_mat'], 'grp_idx': g, 'obs_data': y}
    exec(src, ns)
    with ns['model_A']:
        tr = pm.sample(draws=1000, tune=1000, chains=4, cores=2, random_seed=r, progressbar=False)
    for p in ['gam_h', 'gam_l']:
        for gi, gname in enumerate(groups):
            v = tr.posterior[p].sel(groups=gname).values.ravel()
            lo, hi = az.hdi(v, hdi_prob=0.95)
            rows.append({'rep': r, 'param': p, 'group': gname, 'true': true[p][gi], 'mean': v.mean(), 'sd': v.std(),
                         'covered': lo <= true[p][gi] <= hi, 'z_err': (v.mean() - true[p][gi]) / v.std()})
    print('rep', r, 'done', flush=True)
res = pd.DataFrame(rows)
res.to_csv(Path(__file__).resolve().parent / 'results' / 'recover_A_lapses.csv', index=False)
print(res.groupby(['param', 'group']).agg(true=('true', 'first'), mean_est=('mean', 'mean'), mean_sd=('sd', 'mean'),
                                          bias_in_sd=('z_err', 'mean'), coverage=('covered', 'mean')).round(4).to_string())
