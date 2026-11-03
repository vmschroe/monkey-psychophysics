"""Export per-trial LOO values and per-session leave-one-session-out values of every model to CSV,
so the comparison can be redone without the (large) posterior files."""
import sys, json, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import load_data, ALL_MODELS, X_SIG, X_MU

FITS, RES = BASE / 'fits', BASE / 'results'
data = load_data()
trial = pd.DataFrame({'trial': np.arange(len(data['y'])), 'session': np.array(data['sessions'])[data['sess']],
                      'hand': np.where(data['hand'] == 0, 'left', 'right'), 'condition': np.where(data['c'] == 1, 'bi', 'uni'),
                      'stim_um': np.round(data['x'] * X_SIG + X_MU).astype(int),
                      'dist_um': np.round(data['d'] * X_SIG).astype(int), 'response_high': data['y']})
sess = pd.DataFrame({'session': data['sessions'], 'dist_um': np.round(data['sess_d'] * X_SIG).astype(int)})
diags = []
for m in ALL_MODELS:
    f = FITS / f'{m}.loo.pkl'
    if not f.exists():
        continue
    loo = pickle.load(open(f, 'rb'))
    trial[f'elpd_{m}'] = loo.loo_i.values
    trial[f'k_{m}'] = loo.pareto_k.values
    g = FITS / f'{m}.loso_s1.pkl'
    if g.exists():
        ls = pickle.load(open(g, 'rb'))
        sess[f'elpd_{m}'] = ls['loo'].loo_i.values
        sess[f'k_{m}'] = ls['loo'].pareto_k.values
    diags.append(json.load(open(FITS / f'{m}.diag.json')))
trial.to_csv(RES / 'pointwise_loo_by_trial.csv', index=False)
sess.to_csv(RES / 'pointwise_loso_by_session.csv', index=False)
pd.DataFrame(diags).to_csv(RES / 'sampling_diagnostics.csv', index=False)
print('exported', trial.shape, sess.shape, len(diags))
