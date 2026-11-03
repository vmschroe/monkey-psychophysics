"""Nominal parameter counts of every model: population-level (hand/group level) and per-session."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import load_data, build, ALL_MODELS

data = load_data()
nS = len(data['sessions'])
rows = []
for m in ALL_MODELS:
    model = build(m, data)
    pop = sess_ = 0
    for rv in model.free_RVs:
        dims = model.named_vars_to_dims.get(rv.name, ())
        size = int(np.prod(model.initial_point()[model.rvs_to_values[rv].name].shape)) if model.rvs_to_values[rv].name in model.initial_point() else None
        if size is None:
            continue
        if 'sessions' in dims:
            sess_ += size
        else:
            pop += size
    rows.append({'model': m, 'pop_params': pop, 'session_params_total': sess_, 'per_session': sess_ / nS})
    print(m, pop, sess_, sess_ / nS, flush=True)
out = Path(__file__).resolve().parent / 'results5000'; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'param_counts.csv', index=False)
