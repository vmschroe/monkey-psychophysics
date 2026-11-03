"""Check the PSIS leave-one-session-out estimate against brute force: refit a model without one
session's trials and compute log p(y_s | y_-s) = log mean_j p(y_s | phi_j^(-s)) with the same
session-level integration. Usage: exact_loso.py <model> <session index>"""
import sys, json, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pymc as pm, arviz as az
from scipy.special import logsumexp
from models import load_data, build
import loso

name, s_out = sys.argv[1], int(sys.argv[2])
data = load_data()
keep = data['sess'] != s_out
sub = dict(data)
for key in ('x', 'd', 'c', 'hand', 'sess', 'y', 'grp4'):
    sub[key] = data[key][keep]
for key in ('dA', 'dB', 'dC'):
    dd = dict(data[key])
    for k2 in ('cov_mat', 'grp_idx', 'resp', 'sess_idx'):
        if k2 in dd:
            dd[k2] = dd[k2][keep]
    sub[key] = dd
model = build(name, sub)
with model:
    tr = pm.sample(draws=1000, tune=1000, chains=4, cores=2, random_seed=321, target_accept=0.9, progressbar=False)
full_post = az.from_netcdf(f'{Path(__file__).resolve().parent}/fits/{name}.nc').posterior
loso.N_PHI = 1000
logp_minus, _ = loso.loso_model(name, tr.posterior, data, seed=5, post_u=full_post, only=[s_out])
exact = logsumexp(logp_minus[:, s_out]) - np.log(logp_minus.shape[0])
psis = pickle.load(open(f'{Path(__file__).resolve().parent}/fits/{name}.loso_s1.pkl', 'rb'))
est = float(psis['loo'].loo_i.values[s_out]); k = float(psis['loo'].pareto_k.values[s_out])
out = {'model': name, 'session': s_out, 'exact_refit': float(exact), 'psis': est, 'pareto_k': k,
       'divergences_refit': int(tr.sample_stats['diverging'].sum())}
print(json.dumps(out))
json.dump(out, open(f'{Path(__file__).resolve().parent}/fits/exact_{name}_{s_out}.json', 'w'))
