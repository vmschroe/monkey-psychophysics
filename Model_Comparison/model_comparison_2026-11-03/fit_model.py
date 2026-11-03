"""Fit one model to the real data (same sampler settings for every model) and save its
posterior, pointwise PSIS-LOO and sampling diagnostics to fits/<name>.*"""
import sys, time, pickle, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pymc as pm, arviz as az
from pathlib import Path
from models import load_data, build

name = sys.argv[1]
OUT = Path(__file__).resolve().parent / 'fits'; OUT.mkdir(exist_ok=True)
DRAWS, TUNE, CHAINS, SEED, TARGET = 1000, 1000, 4, 123, 0.9

data = load_data()
model = build(name, data)
t0 = time.time()
with model:
    trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=2, random_seed=SEED, target_accept=TARGET,
                      progressbar=False, idata_kwargs={"log_likelihood": True})
minutes = (time.time() - t0) / 60
loo = az.loo(trace, pointwise=True)

# diagnostics on everything except Model B's z_sig_* (sign unidentified through abs(), see Vera's reports)
skip = (lambda n: n.startswith('z_sig_')) if name in ('B', 'B-wide') else (lambda n: False)
names = [rv.name for rv in model.free_RVs if not skip(rv.name)] + [d.name for d in model.deterministics]
diag = {'model': name, 'minutes': round(minutes, 1),
        'divergences': int(trace.sample_stats['diverging'].sum()),
        'max_tree_depth': int(trace.sample_stats['tree_depth'].max()),
        'max_r_hat': float(az.rhat(trace, var_names=names).to_array().max()),
        'min_ess_bulk': float(az.ess(trace, var_names=names).to_array().min()),
        'min_ess_tail': float(az.ess(trace, var_names=names, method='tail').to_array().min()),
        'elpd_loo': float(loo.elpd_loo), 'se': float(loo.se), 'p_loo': float(loo.p_loo),
        'max_pareto_k': float(loo.pareto_k.max()), 'n_k_gt_0.7': int((loo.pareto_k > 0.7).sum())}
print(json.dumps(diag))
pickle.dump(loo, open(OUT / f'{name}.loo.pkl', 'wb'))
json.dump(diag, open(OUT / f'{name}.diag.json', 'w'))
az.InferenceData(posterior=trace.posterior, sample_stats=trace.sample_stats).to_netcdf(OUT / f'{name}.nc')
print('saved', name)
