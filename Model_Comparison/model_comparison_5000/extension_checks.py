"""Descriptive checks on the raw data for possible model extensions:
  1. trial history: does the previous completed trial (its choice, its outcome) bias the current choice?
  2. response side: is 'high' chosen more when the 'high' target is on one side (a saccade-direction bias)?
  3. within-session drift: do lapse-like answers become more frequent late in a session?
  4. criterion adaptation: does the unimanual bias depend on the session's distractor amplitude?
Each check is a logistic regression with session x hand x condition intercepts and per hand x condition
stimulus slopes, so the effect of interest is measured against the psychometric curve."""
import os, numpy as np, pandas as pd, pickle, warnings
# Needs two trial tables built during the first review: RAW_ALL (every row of every per-session trial
# file, with a "file" column) and VERA_REBUILT (Vera's 12,937 selected trials with their raw row index).
RAW_ALL = os.environ.get('RAW_ALL', 'raw_all.pkl'); VERA_REBUILT = os.environ.get('VERA_REBUILT', 'vera_rebuilt.pkl')
REPO = os.environ.get('MP_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import statsmodels.api as sm
warnings.filterwarnings('ignore')

raw = pd.read_pickle(RAW_ALL)          # every trial of every file, in file order
vr = pd.read_pickle(VERA_REBUILT)       # Vera's 12,937 analysed trials (raw index kept)
D = pickle.load(open(os.path.join(REPO, 'Sirius_Data.pkl'), 'rb'))['data'].reset_index(drop=True)
assert np.array_equal(vr['response'].to_numpy(), D['response'].to_numpy())
X_SIG = np.std([6, 12, 18, 24, 32, 38, 44, 50])

# ---- trial history from the raw stream: previous trial with a choice (correct or incorrect), same file
raw = raw.sort_values(['file', 'start_time']).copy()
raw['choice_made'] = raw['outcomeID'].isin([10, 31, 32, 33, 34, 11, 61, 62, 63, 64])
raw['chose_high'] = raw['selectedChoiceTarget_ID'].isin([3, 4]).astype(float)
raw['rewarded'] = (raw['outcomeID'] == 10).astype(float)
raw['seq'] = raw.groupby('file').cumcount()
ch = raw[raw['choice_made']].copy()
ch['prev_high'] = ch.groupby('file')['chose_high'].shift(1)
ch['prev_rew'] = ch.groupby('file')['rewarded'].shift(1)
ch['prev_gap'] = ch['seq'] - ch.groupby('file')['seq'].shift(1)            # 1 = immediately preceding trial
ch['prev_stim_high'] = ch.groupby('file')['tact2_Left_AMP'].shift(1)      # placeholder, replaced below
vr = vr.join(ch[['prev_high', 'prev_rew', 'prev_gap']], how='left')
vr['trial_in_session'] = vr.groupby('sess_date').cumcount() / vr.groupby('sess_date')['sess_date'].transform('size')

# ---- target side of the 'high' choice: IDs 3/4 are 'high'; position x of that target
def high_on_right(r):
    for idc, pos in [(r['choice1_ID'], r['choice1_Position']), (r['choice2_ID'], r['choice2_Position'])]:
        if idc in (3, 4):
            return float(str(pos).strip('[]').split(',')[0]) > 0
    return np.nan
vr['high_right'] = vr.apply(high_on_right, axis=1).astype(float)

df = pd.DataFrame({'y': D['response'].to_numpy(), 'x': (D['stim_amp'].to_numpy() - 28) / X_SIG,
                   'hand': D['side'].to_numpy(), 'cond': np.where(D['dist_amp'] > 0, 'bi', 'uni'),
                   'sess': D['sess_idx'].to_numpy(), 'samp': D.groupby('sess_idx')['dist_amp'].transform('max').to_numpy(),
                   'prev_high': vr['prev_high'].to_numpy(), 'prev_rew': vr['prev_rew'].to_numpy(),
                   'prev_gap': vr['prev_gap'].to_numpy(), 'pos': vr['trial_in_session'].to_numpy(),
                   'high_right': vr['high_right'].to_numpy()})
df['cell'] = df['hand'] + '_' + df['cond']
out = {}

def fit(formula_cols, data, label):
    X = pd.get_dummies(data[['sess_cell']], drop_first=False).astype(float)
    for c in ['left_uni', 'left_bi', 'right_uni', 'right_bi']:
        X[f'x_{c}'] = data['x'] * (data['cell'] == c)
    for name, col in formula_cols.items():
        X[name] = col
    m = sm.GLM(data['y'], X, family=sm.families.Binomial()).fit()
    res = {k: (m.params[k], m.bse[k]) for k in formula_cols}
    print(f"{label}: " + '; '.join(f"{k} = {v:.3f} (SE {s:.3f}, z {v / s:.1f})" for k, (v, s) in res.items()))
    return res

df['sess_cell'] = df['sess'].astype(str) + '_' + df['cell']

# 1. trial history (previous choice coded +-1, and its interaction with reward: win-stay / lose-shift)
h = df.dropna(subset=['prev_high', 'prev_rew']).copy()
pc = 2 * h['prev_high'] - 1
out['history'] = fit({'prev_choice': pc, 'prev_choice_x_rewarded': pc * h['prev_rew']}, h,
                     f"1. history ({len(h)} trials)")
hb = h[h['prev_gap'] == 1]
fit({'prev_choice': 2 * hb['prev_high'] - 1, 'prev_choice_x_rewarded': (2 * hb['prev_high'] - 1) * hb['prev_rew']}, hb,
    f"   history, previous trial immediately before ({len(hb)} trials)")

# 2. response side
s2 = df.dropna(subset=['high_right']).copy()
out['side'] = fit({'high_target_right': 2 * s2['high_right'] - 1}, s2, f"2. side of the 'high' target ({len(s2)} trials)")

# 3. within-session drift of lapse-like answers: weak stimuli 'high', strong stimuli 'low', by session third
stim = np.round(df['x'] * X_SIG + 28)
df['third'] = pd.cut(df['pos'], [0, 1 / 3, 2 / 3, 1.01], labels=['early', 'middle', 'late'], include_lowest=True)
lap = pd.DataFrame({'weak stim: P(high)': df[stim <= 12].groupby(['cond', 'third'])['y'].mean(),
                    'strong stim: P(low)': 1 - df[stim >= 44].groupby(['cond', 'third'])['y'].mean()}).round(3)
print("3. lapse-like answers by third of the session:\n" + lap.to_string())
out['drift'] = fit({'pos_centered': df['pos'] - 0.5}, df, "   overall P(high) trend within session")

# 4. criterion adaptation: unimanual PSE vs the session's distractor amplitude (session intercepts replaced by
#    hand intercepts + amplitude slope + session order), unimanual trials only
u = df[df['cond'] == 'uni'].copy()
Xu = pd.get_dummies(u['hand']).astype(float)
for hnd in ['left', 'right']:
    Xu[f'x_{hnd}'] = u['x'] * (u['hand'] == hnd)
    Xu[f'amp10_{hnd}'] = (u['samp'] / 10) * (u['hand'] == hnd)
    Xu[f'order10_{hnd}'] = (u['sess'] / 10) * (u['hand'] == hnd)
mu_ = sm.GLM(u['y'], Xu, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': u['sess']})
for hnd in ['left', 'right']:
    b, se = mu_.params[f'amp10_{hnd}'], mu_.bse[f'amp10_{hnd}']
    slope = mu_.params[f'x_{hnd}']
    print(f"4. {hnd} unimanual: logit shift per 10 um of session distractor {b:.3f} (session-clustered SE {se:.3f}); "
          f"PSE shift = {-b / slope * X_SIG:.2f} um per 10 um; order per 10 sessions {mu_.params[f'order10_{hnd}']:.3f} (SE {mu_.bse[f'order10_{hnd}']:.3f})")

# frequency-manipulation trials that were dropped
fm = raw[raw['outcomeID'].isin([10, 31, 32, 33, 34]) & ~raw['repeatedTrial'].astype(bool)
         & ~(raw['tact2_Left_FR'].isin([0, 200]) & raw['tact2_Right_FR'].isin([0, 200]))]
print(f"dropped frequency-manipulation trials with a valid choice: {len(fm)}; FR values: "
      f"{sorted(set(fm['tact2_Left_FR']) | set(fm['tact2_Right_FR']))}")
pickle.dump(out, open('results5000/extension_checks.pkl', 'wb'))
