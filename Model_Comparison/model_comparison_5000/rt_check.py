"""Response times (choice-target onset to start of the response movement) of lapse-like errors and correct
answers, by condition, hand and the session's distractor amplitude. Descriptive only."""
import pickle, sys, pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
d = pickle.load(open(REPO / 'Sirius_Data.pkl', 'rb'))['data'].reset_index(drop=True)
d['rt'] = (d['responseMovStart'] - d['choiceTarget_ON']) * 1000
d['cond'] = np.where(d['dist_amp'] > 0, 'bi', 'uni')
samp = d.groupby('sess_idx')['dist_amp'].transform('max')
d['amp_bin'] = pd.cut(samp, [0, 12, 22, 32], labels=['6-12', '16-22', '24-32'])
d['answer'] = np.where(d['response'] == 1, 'high', 'low')
weak, strong = d[d['stim_amp'] <= 12], d[d['stim_amp'] >= 44]
pd.set_option('display.width', 200)
print(f"median response time {d['rt'].median():.0f} ms")
print("\nWeak targets (6-12 um): median RT (ms) and n by condition and answer")
print(weak.groupby(['cond', 'answer'])['rt'].agg(['median', 'size']).round(0).to_string())
print("\nWeak targets, bimanual: by hand, session distractor amplitude and answer")
print(weak[weak['cond'] == 'bi'].groupby(['side', 'amp_bin', 'answer'], observed=True)['rt'].agg(['median', 'size']).round(0).to_string())
print("\nStrong targets (44-50 um): by condition and answer")
print(strong.groupby(['cond', 'answer'])['rt'].agg(['median', 'size']).round(0).to_string())
for side in ['left', 'right']:
    b = weak[(weak['cond'] == 'bi') & (weak['side'] == side) & (weak['amp_bin'] == '24-32')]
    if (b['answer'] == 'high').sum() > 2:
        p = mannwhitneyu(b[b['answer'] == 'high']['rt'], b[b['answer'] == 'low']['rt']).pvalue
        print(f"{side} hand, 24-32 um sessions, weak targets: 'high' vs 'low' RT Mann-Whitney p = {p:.3g}")
b = weak[weak['cond'] == 'bi']
print(f"all bimanual weak targets: 'high' vs 'low' RT Mann-Whitney p = {mannwhitneyu(b[b['answer'] == 'high']['rt'], b[b['answer'] == 'low']['rt']).pvalue:.3g}")
