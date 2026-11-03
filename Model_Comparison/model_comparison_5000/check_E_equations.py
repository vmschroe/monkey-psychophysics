"""Independent check that the equations written in the conclusions note (Section 2) reproduce the fitted Model E:
rebuild the trial log-likelihood from the hand-level parameters and the standard-normal z's exactly as written,
and compare PSIS-LOO on a random subset of trials with the stored pointwise values.
Usage: check_E_equations.py [fits dir]  (default fits5000/, written by fit5000.py)"""
import sys, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE)); warnings.filterwarnings('ignore')
FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
import numpy as np, arviz as az, xarray as xr
from models import load_data, X_SIG
from scipy.special import expit

data = load_data()
post = az.from_netcdf(FITS / 'D4+crit.nc').posterior
stored = pickle.load(open(FITS / 'D4+crit.loo.pkl', 'rb'))
C, D = post.sizes['chain'], post.sizes['draw']
def v(name, **sel):
    return post[name].sel(**sel).values.reshape(C * D, *post[name].sel(**sel).shape[2:])

d_s = data['sess_d']; d_bar = d_s.mean()
# hand-level parameters (draws, hands)
mu0, s0 = v('mu_betas', betas='b0'), v('sig_betas', betas='b0')
mu1, s1 = v('mu_betas', betas='b1'), v('sig_betas', betas='b1')
b2, b3, lam = v('mu_betas', betas='b2'), v('mu_b3'), v('lam')
z0, z1 = v('z_b0'), v('z_b1')                                    # (draws, hands, sessions)
# session level, as written: beta0 = mu0 + lam (d_s - dbar) + s0 z0;  beta1 = mu1 + s1 z1
beta0 = mu0[..., None] + lam[..., None] * (d_s - d_bar)[None, None, :] + s0[..., None] * z0
beta1 = mu1[..., None] + s1[..., None] * z1
# lapses: omega^k = mu^k + s^k z;  omega^{k,bi} = mu^{k,bi} + kappa^k (d_s - dbar) + s^{k,bi} z
om_u = v('mu_gams')[..., None] + v('sig_gams')[..., None] * v('z_gams')                       # (draws, k, hands, sessions)
om_b = v('mu_gams_bi')[..., None] + v('kappa')[..., None] * (d_s - d_bar) + v('sig_gams_bi')[..., None] * v('z_gams_bi')

rng = np.random.default_rng(7)
idx = np.sort(rng.choice(len(data['y']), 1500, replace=False))
h, s, x, c, y = data['hand'][idx], data['sess'][idx], data['x'][idx], data['c'][idx], data['y'][idx]
d_i = c * d_s[s]
eta = beta0[:, h, s] + beta1[:, h, s] * x + b2[:, h] * d_i + b3[:, h] * c
g = np.where(c[None, None, :] == 1, 0.25 * expit(om_b[:, :, h, s]), 0.25 * expit(om_u[:, :, h, s]))
p = g[:, 0] + (1 - g[:, 0] - g[:, 1]) * expit(eta)
p = np.clip(p, 1e-8, 1 - 1e-8)
ll = y * np.log(p) + (1 - y) * np.log1p(-p)
res = az.loo(az.InferenceData(log_likelihood=xr.Dataset({'y': xr.DataArray(ll.reshape(C, D, -1), dims=('chain', 'draw', 'obs'))})),
             pointwise=True, reff=stored['reff'])
diff = np.abs(res.loo_i.values - stored['loo_i'][idx])
print(f'max |loo_i(equations) - loo_i(stored)| over {len(idx)} trials: {diff.max():.2e}')
# the deterministic session parameters stored by the sampler agree with the written session-level equations
print('beta0 max diff', np.abs(beta0 - v('beta_vec', betas='b0')).max(), ' beta1 max diff', np.abs(beta1 - v('beta_vec', betas='b1')).max())
print('gams_bi max diff', np.abs(0.25 * expit(om_b) - v('gams_bi')).max())

# the closed-form PSE of Section 2.5 against the numerically found values in effects.csv (2000 thinned draws)
import pandas as pd
eff = pd.read_csv(BASE / 'results5000' / 'effects.csv')
th = np.linspace(0, C * D - 1, 2000).astype(int)
gu, gb = 0.25 * expit(om_u[th]), 0.25 * expit(om_b[th])           # (draws, k, hands, sessions)
for hh, hand in enumerate(['left', 'right']):
    def pse(cond):
        g = gb if cond else gu
        lnr = np.log((1 - 2 * g[:, 0, hh]) / (1 - 2 * g[:, 1, hh]))
        dist = cond * (b2[th, hh, None] * d_s[None, :] + b3[th, hh, None])
        return 28 + X_SIG / beta1[th, hh] * (lnr - beta0[th, hh] - dist)
    shift = (pse(1) - pse(0)).mean(1)
    num = eff[(eff.model == 'D4+crit') & (eff.hand == hand) & (eff.quantity == 'PSE shift')]['mean'].item()
    print(f'{hand}: PSE shift closed form {shift.mean():.3f}  numerical {num:.3f}')
