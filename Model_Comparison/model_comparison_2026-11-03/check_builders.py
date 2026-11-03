"""Before fitting: (1) my builder with every switch off reproduces Vera's Model C log-likelihood
on her refit posterior; (2) for every model, PyMC's log-likelihood equals the numpy mirror
(trial_p) at prior draws."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pymc as pm, arviz as az
from models import load_data, build, build_family, trial_p, ALL_MODELS

data = load_data()
y = data['y']

# (1) builder 'C' == Vera's C on Vera-C posterior draws
trC = az.from_netcdf(Path(__file__).resolve().parent / 'fits' / 'C.nc')
sub = trC.posterior.isel(draw=slice(0, 25))
mine = build_family(data)
with mine:
    ll = pm.compute_log_likelihood(az.InferenceData(posterior=sub), var_names=['resp'], progressbar=False,
                                   extend_inferencedata=False)
ref = trC.log_likelihood['resp'].isel(draw=slice(0, 25)).values
print(f"builder C vs Vera's C: max |diff| = {np.max(np.abs(ll['resp'].values - ref)):.2e}")

# (2) PyMC log-likelihood vs numpy mirror at prior draws, every model
for name in ALL_MODELS:
    model = build(name, data)
    with model:
        pri = pm.sample_prior_predictive(draws=10, random_seed=7).prior
        ll = pm.compute_log_likelihood(az.InferenceData(posterior=pri), var_names=['resp'], progressbar=False,
                                       extend_inferencedata=False)['resp'].values.reshape(10, -1)
    p = np.clip(trial_p(name, pri, data), 1e-8, 1 - 1e-8)
    ll_np = y * np.log(p) + (1 - y) * np.log1p(-p)
    print(f"{name:8s}: {len(model.free_RVs)} free RVs, max |pymc - numpy| = {np.max(np.abs(ll - ll_np)):.2e}")
