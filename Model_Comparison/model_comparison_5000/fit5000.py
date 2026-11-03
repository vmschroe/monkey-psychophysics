"""Fit one model to the real data with Vera's chain length (4 chains x 5000 draws after 1000 tuning steps):
nutpie's NUTS for Model C and its variants, PyMC's NUTS for A and the B family. PSIS-LOO is computed from the
numpy likelihood (identical to az.loo on PyMC's log-likelihood, checked to 1e-15), so the 20,000 x 12,937
log-likelihood array is never stored.
Usage: fit5000.py <model> [draws] [outdir]"""
import sys, time, pickle, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, arviz as az, nutpie
from models import load_data, build, loo_numpy

name = sys.argv[1]
DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parent / 'fits5000'
OUT.mkdir(exist_ok=True)
TUNE, CHAINS, SEED, TARGET = 1000, 4, 123, 0.9

data = load_data()
model = build(name, data)
t0 = time.time()
# Vera's A and B (|z| parameterization of the sds) sample faster with PyMC's NUTS; the C family with nutpie's
SAMPLER = 'pymc' if name == 'A' or name.startswith('B') else 'nutpie'
if SAMPLER == 'nutpie':
    compiled = nutpie.compile_pymc_model(model, backend='numba')
    trace = nutpie.sample(compiled, draws=DRAWS, tune=TUNE, chains=CHAINS, cores=2, seed=SEED, target_accept=TARGET,
                          progress_bar=False)
else:
    import pymc as pm
    with model:
        trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=2, random_seed=SEED, target_accept=TARGET,
                          progressbar=False)
minutes = (time.time() - t0) / 60
post = trace.posterior

# diagnostics on everything except Model B's z_sig_* (sign unidentified through abs(), see Vera's reports)
names = [v for v in post.data_vars if not (name.startswith('B') and v.startswith('z_sig_'))]
st = trace.sample_stats
depth = st['depth'] if 'depth' in st else st.get('tree_depth')
diag = {'model': name, 'sampler': SAMPLER, 'draws': DRAWS, 'minutes': round(minutes, 1),
        'divergences': int(st['diverging'].sum()),
        'max_tree_depth': int(depth.max()) if depth is not None else None,
        'max_r_hat': float(az.rhat(post, var_names=names).to_array().max()),
        'min_ess_bulk': float(az.ess(post, var_names=names).to_array().min()),
        'min_ess_tail': float(az.ess(post, var_names=names, method='tail').to_array().min())}
t1 = time.time()
loo = loo_numpy(name, post, data)
diag.update({'elpd_loo': float(loo['elpd_loo']), 'se': float(loo['se']), 'p_loo': float(loo['p_loo']),
             'max_pareto_k': float(loo['pareto_k'].max()), 'n_k_gt_0.7': int((loo['pareto_k'] > 0.7).sum()),
             'loo_minutes': round((time.time() - t1) / 60, 1)})
print(json.dumps(diag), flush=True)
pickle.dump(loo, open(OUT / f'{name}.loo.pkl', 'wb'))
az.InferenceData(posterior=post, sample_stats=st).to_netcdf(OUT / f'{name}.nc')
json.dump(diag, open(OUT / f'{name}.diag.json', 'w'))       # written last: other scripts wait for it
print('saved', name, flush=True)
