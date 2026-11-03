"""LaTeX tables and number macros for the report, read straight from the result files, so that every number
in the PDF comes from the fits. Usage: report_tables.py [fits dir]; results dir from RESDIR (default
results5000); writes RESDIR/tex/*.tex and prints the key pairwise comparisons."""
import os, sys, json, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from models import load_data, ALL_MODELS

FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
RES = Path(os.environ.get('RESDIR', BASE / 'results5000'))
TEX = RES / 'tex'; TEX.mkdir(exist_ok=True, parents=True)
data = load_data(); sess = data['sess']; nS = len(data['sessions'])
REF = 'C+LKP'                                    # Model D, the reference of the tables
REC = 'D4+crit'                                  # Model E, the recommended model
MINUS = r'\ensuremath{-}'

cmp_ = pd.read_csv(RES / 'comparison.csv').set_index('model')
eff = pd.read_csv(RES / 'effects.csv')
par = pd.read_csv(RES / 'parameters.csv')
ppc = pd.read_csv(RES / 'ppc_lapses.csv')
cnt = pd.read_csv(RES / 'param_counts.csv').set_index('model')
models = [m for m in ALL_MODELS if m in cmp_.index]

_loo, _loso = {}, {}
def loo_i(m):
    if m not in _loo:
        _loo[m] = np.asarray(pickle.load(open(FITS / f'{m}.loo.pkl', 'rb'))['loo_i'], dtype=float).ravel()
    return _loo[m]
def loso_i(m):
    if m not in _loso:
        f = FITS / f'{m}.loso_s1.pkl'
        _loso[m] = pickle.load(open(f, 'rb'))['loo'].loo_i.values if f.exists() else None
    return _loso[m]


def pair(m1, m2):
    """elpd(m1) - elpd(m2): trial-level LOO with the SE clustered by session, and LOSO with the SE over
    sessions (both: sqrt(42 x sample variance of the per-session differences))."""
    d = loo_i(m1) - loo_i(m2)
    cl = pd.Series(d).groupby(sess).sum()
    out = {'trial': d.sum(), 'trial_se': np.sqrt(nS * cl.var(ddof=1)), 'trial_se_iid': np.sqrt(len(d) * np.var(d))}
    a, b = loso_i(m1), loso_i(m2)
    if a is not None and b is not None:
        e = a - b
        out.update({'loso': e.sum(), 'loso_se': np.sqrt(nS * np.var(e, ddof=1)), 'loso_n_better': int((e > 0).sum())})
    return out


GROUPS = [("Vera's models", ['A', 'B', 'B-wide', 'B-fixlapse', 'C']),
          ('Variants of C', ['C+L', 'C+LK', 'C+LP', 'C+LPS', 'C+LKPS', 'C+Mmax', 'C+Mdist', 'C+LMmax', 'C+LPMmax', 'C+LPMdist']),
          ('Model D, recommended in the previous report', ['C+LKP']),
          ('Simpler versions of D', ['D4', 'D1', 'D2', 'D3', 'D0']),
          ('Extensions of D', ['D+shared', 'D+crit', 'D+order', 'D+crit+order']),
          ('Simpler versions with the trend', ['D4+crit', 'D3+crit', 'D4+crit+shared'])]
NAME = {'C+LKP': 'D (= C+LKP)', 'D4+crit': 'E (= D4+crit)', 'D4+crit+shared': 'E+shared'}
DESC = {'A': 'pooled: no session differences',
        'B': 'separate unimanual and bimanual hierarchies',
        'B-wide': 'B with a 3$\\times$ wider prior on the lapse sd',
        'B-fixlapse': 'B with lapses fixed per group',
        'C': 'shared curve, bimanual offset $\\propto$ amplitude',
        'C+L': 'C + own lapse rates on bimanual trials',
        'C+LK': 'C+L + bimanual lapses depend on amplitude',
        'C+LP': 'C+L + presence offset',
        'C+LPS': 'C+LP + bimanual slope change',
        'C+LKPS': 'D + bimanual slope change',
        'C+Mmax': 'C + judges the stronger vibration',
        'C+Mdist': 'C + judges the distractor',
        'C+LMmax': 'C+L + judges the stronger vibration',
        'C+LPMmax': 'C+LP + judges the stronger vibration',
        'C+LPMdist': 'C+LP + judges the distractor',
        'C+LKP': 'C+LK + presence offset',
        'D4': 'D, distractor terms fixed per hand',
        'D1': 'D, lapses fixed per hand and condition',
        'D2': 'D1 and D4 combined',
        'D3': 'D2 + slope fixed: only the bias varies',
        'D0': 'D with no session differences',
        'D+shared': 'D + daily bias shared by the two hands',
        'D+crit': 'D + bias follows the distractor amplitude',
        'D+order': 'D + bias drifts over the study',
        'D+crit+order': 'D + both',
        'D4+crit': 'D4 + bias follows the distractor amplitude',
        'D3+crit': 'D3 + bias follows the distractor amplitude',
        'D4+crit+shared': 'E + daily bias shared by the two hands'}


def f1(v):
    t = f'{v:.1f}'
    return ('0.0' if t == '-0.0' else t).replace('-', MINUS)


def fmt_d(v, se):
    if v == 0 and se == 0:
        return '0', ''
    return f1(v), f'({se:.1f})'


# ---- every model against Model D (for the figures) ----------------------------------------------------
rows = []
for m in models:
    pr = pair(m, REF) if m != REF else {'trial': 0.0, 'trial_se': 0.0, 'loso': 0.0, 'loso_se': 0.0}
    rows.append({'model': m, **{k: pr.get(k, np.nan) for k in ['trial', 'trial_se', 'loso', 'loso_se', 'loso_n_better']}})
pd.DataFrame(rows).to_csv(RES / 'pairwise_vs_D.csv', index=False)

# ---- Table: comparison against Model D -------------------------------------------------------------
lines = [r'\setlength{\tabcolsep}{4.5pt}', r'\begin{tabularx}{\textwidth}{@{}l>{\raggedright\arraybackslash}Xrrr@{\ }lr@{\ }l@{}}', r'\toprule',
         r'Model & What it is & Parameters & $p_\mathrm{loo}$ & \multicolumn{2}{c}{Trial LOO} & \multicolumn{2}{c}{LOSO} \\',
         r' & & pop.\,/\,session & & \multicolumn{2}{c}{vs D (SE)} & \multicolumn{2}{c}{vs D (SE)} \\', r'\midrule']
for gi, (g, ms) in enumerate(GROUPS):
    ms = [m for m in ms if m in models]
    if not ms:
        continue
    if gi:
        lines.append(r'\addlinespace[3pt]')
    lines.append(rf'\multicolumn{{8}}{{@{{}}l}}{{\textit{{{g}}}}} \\')
    for m in ms:
        pr = pair(m, REF) if m != REF else {'trial': 0, 'trial_se': 0, 'loso': 0, 'loso_se': 0}
        t, tse = fmt_d(pr['trial'], pr['trial_se'])
        l_, lse = fmt_d(pr.get('loso', np.nan), pr.get('loso_se', np.nan)) if 'loso' in pr else ('--', '')
        nm = NAME.get(m, m)
        if m in (REF, REC):
            nm = r'\textbf{' + nm + '}'
        lines.append(f"{nm} & {DESC[m]} & {cnt.loc[m, 'pop_params']}\\,/\\,{cnt.loc[m, 'per_session']:.0f} & "
                     f"{cmp_.loc[m, 'p_loo']:.0f} & {t} & {tse} & {l_} & {lse} \\\\")
lines += [r'\bottomrule', r'\end{tabularx}']
(TEX / 'tab_comparison.tex').write_text('\n'.join(lines) + '\n')

# ---- Table: sampling diagnostics -------------------------------------------------------------------
diag = {m: json.load(open(FITS / f'{m}.diag.json')) for m in models}
lines = [r'\begin{tabular}{@{}lllrrrrrrr@{}}', r'\toprule',
         r'Model & Sampler & Minutes & Div. & max $\hat R$ & min ESS & min ESS & max $k$ & max $k$ & sessions \\',
         r' & & & & & bulk & tail & trial & LOSO & $k>0.7$ \\', r'\midrule']
for g, ms in GROUPS:
    for m in [m for m in ms if m in models]:
        dg = diag[m]
        lines.append(f"{NAME.get(m, m)} & {'PyMC' if dg.get('sampler', 'pymc') == 'pymc' else 'nutpie'} & {dg['minutes']:.0f} & "
                     f"{dg['divergences']} & {dg['max_r_hat']:.4f} & {dg['min_ess_bulk']:.0f} & {dg['min_ess_tail']:.0f} & "
                     f"{dg['max_pareto_k']:.2f} & "
                     + (f"{cmp_.loc[m, 'max_k_loso']:.2f} & {int(cmp_.loc[m, 'n_k_loso_gt_0.7'])}" if 'max_k_loso' in cmp_.columns and pd.notna(cmp_.loc[m, 'max_k_loso']) else '-- & --')
                     + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TEX / 'tab_diagnostics.tex').write_text('\n'.join(lines) + '\n')

# ---- Table: distractor effects in the main models ---------------------------------------------------
SHOW = [m for m in ['B', 'C', 'C+LKP', 'D4+crit'] if m in set(eff['model'])]
Q = [('PSE shift', 'PSE shift, mean'), ('PSE shift @6um', 'PSE shift, 6\\,\\textmu m'),
     ('PSE shift @32um', 'PSE shift, 32\\,\\textmu m'), ('JND shift', 'JND change, mean')]


def cell(m, hand, q):
    r = eff[(eff['model'] == m) & (eff['hand'] == hand) & (eff['quantity'] == q)]
    if not len(r):
        return '--'
    r = r.iloc[0]
    if max(abs(r['mean']), abs(r['lo']), abs(r['hi'])) < 0.05:
        return '0 (by design)'
    return f"{f1(r['mean'])} [{f1(r['lo'])}, {f1(r['hi'])}]"


lines = [r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}ll' + 'c' * len(SHOW) + '@{}}', r'\toprule',
         'Quantity (\\textmu m) & Hand & ' + ' & '.join(NAME.get(m, m).replace(' (= C+LKP)', '').replace(' (= D4+crit)', '') for m in SHOW) + r' \\', r'\midrule']
for q, lab in Q:
    for hand in ['left', 'right']:
        lines.append(f"{lab if hand == 'left' else ''} & {hand} & " + ' & '.join(cell(m, hand, q) for m in SHOW) + r' \\')
    lines.append(r'\addlinespace[2pt]')
lines = lines[:-1] + [r'\bottomrule', r'\end{tabular}']
(TEX / 'tab_effects.tex').write_text('\n'.join(lines) + '\n')

# ---- number macros (filled below) --------------------------------------------------------------------
mac = {}
def put(name, v, nd=1):
    if isinstance(v, (float, np.floating)):
        t = f'{float(v):.{nd}f}'
        t = t[1:] if t.startswith('-') and float(t) == 0 else t
        mac[name] = t.replace('-', MINUS)
    else:
        mac[name] = str(v)

# ---- 1000 versus 5000 draws --------------------------------------------------------------------------
OLD = BASE / 'fits'
rows = []
for m in models:
    if (OLD / f'{m}.diag.json').exists():
        o = json.load(open(OLD / f'{m}.diag.json'))
        r = {'model': m, 'elpd_1000': o['elpd_loo'], 'elpd_5000': diag[m]['elpd_loo']}
        if (OLD / f'{m}.loso_s1.pkl').exists() and loso_i(m) is not None:
            r['loso_1000'] = pickle.load(open(OLD / f'{m}.loso_s1.pkl', 'rb'))['summary']['elpd_loso']
            r['loso_5000'] = float(loso_i(m).sum())
        rows.append(r)
dd = pd.DataFrame(rows)
if len(dd):
    dd['d_trial'] = dd['elpd_5000'] - dd['elpd_1000']
    swaps = sum(1 for i in range(len(dd)) for j in range(i + 1, len(dd))
                if np.sign(dd['elpd_5000'][i] - dd['elpd_5000'][j]) != np.sign(dd['elpd_1000'][i] - dd['elpd_1000'][j]))
    txt = (f"For the {len(dd)} models fit both ways, the trial-level elpd changed by at most {dd['d_trial'].abs().max():.2f} "
           f"(median {dd['d_trial'].abs().median():.2f})")
    if 'loso_5000' in dd and dd['loso_5000'].notna().any():
        dd['d_loso'] = dd['loso_5000'] - dd['loso_1000']
        txt += f" and the LOSO elpd by at most {dd['d_loso'].abs().max():.2f} (median {dd['d_loso'].abs().median():.2f})"
        swaps_l = sum(1 for i in range(len(dd)) for j in range(i + 1, len(dd))
                      if pd.notna(dd['loso_5000'][i]) and pd.notna(dd['loso_5000'][j]) and
                      np.sign(dd['loso_5000'][i] - dd['loso_5000'][j]) != np.sign(dd['loso_1000'][i] - dd['loso_1000'][j]))
    else:
        swaps_l = None
    def swapped(col_new, col_old):
        return [(dd['model'][i], dd['model'][j], dd[col_old][i] - dd[col_old][j], dd[col_new][i] - dd[col_new][j])
                for i in range(len(dd)) for j in range(i + 1, len(dd))
                if pd.notna(dd[col_new][i]) and pd.notna(dd[col_new][j]) and
                np.sign(dd[col_new][i] - dd[col_new][j]) != np.sign(dd[col_old][i] - dd[col_old][j])]
    def describe(pairs, label):
        if not pairs:
            return f'no pair of models changed order under {label}'
        names = '; '.join(f'{a} and {b}, {abs(o):.1f} apart before and {abs(n):.1f} after' for a, b, o, n in pairs)
        n = {1: 'one pair', 2: 'two pairs', 3: 'three pairs'}.get(len(pairs), f'{len(pairs)} pairs')
        return f"{n} changed order under {label} ({names})"
    sw_t = swapped('elpd_5000', 'elpd_1000')
    sw_l = swapped('loso_5000', 'loso_1000') if 'loso_5000' in dd else []
    first = describe(sw_t, 'trial-level LOO')
    txt += '. ' + first[0].upper() + first[1:] + ', and ' + describe(sw_l, 'LOSO') + '.'
    if swaps or swaps_l:
        sw = [(dd['model'][i], dd['model'][j]) for i in range(len(dd)) for j in range(i + 1, len(dd))
              if np.sign(dd['elpd_5000'][i] - dd['elpd_5000'][j]) != np.sign(dd['elpd_1000'][i] - dd['elpd_1000'][j])]
        print('order swaps (trial):', sw)
    dd.to_csv(RES / 'draws_1000_vs_5000.csv', index=False)
    put('drawsN', len(dd)); put('drawsMaxTrial', float(dd['d_trial'].abs().max()), 1)
    if 'd_loso' in dd:
        put('drawsMaxLoso', float(dd['d_loso'].abs().max()), 1)
    put('drawsSwapsTrial', swaps); put('drawsSwapsLoso', swaps_l if swaps_l is not None else 0)
    (TEX / 'draws_note.tex').write_text(txt + '\n')
    print(txt)

# ---- number macros -----------------------------------------------------------------------------------

def put_pair(tag, m1, m2):
    if m1 in models and m2 in models:
        pr = pair(m1, m2)
        put(tag + 'Trial', pr['trial']); put(tag + 'TrialSE', pr['trial_se'])
        if 'loso' in pr:
            put(tag + 'Loso', pr['loso']); put(tag + 'LosoSE', pr['loso_se']); put(tag + 'LosoNbetter', pr['loso_n_better'])
        print(f"{m1:>12} vs {m2:<12} trial {pr['trial']:7.1f} ({pr['trial_se']:4.1f}; iid {pr['trial_se_iid']:4.1f})   "
              f"LOSO {pr.get('loso', np.nan):7.1f} ({pr.get('loso_se', np.nan):4.1f}) sessions better {pr.get('loso_n_better', '-')}")

PAIRS = [('DvsB', 'C+LKP', 'B'), ('DvsC', 'C+LKP', 'C'), ('BvsC', 'B', 'C'), ('LvsC', 'C+L', 'C'), ('LvsB', 'C+L', 'B'),
         ('DvsL', 'C+LKP', 'C+L'), ('DvsLK', 'C+LKP', 'C+LK'), ('DvsLP', 'C+LKP', 'C+LP'), ('SvsD', 'C+LKPS', 'C+LKP'),
         ('BwidevsB', 'B-wide', 'B'), ('BfixvsB', 'B-fixlapse', 'B'), ('MmaxvsC', 'C+Mmax', 'C'), ('MdistvsC', 'C+Mdist', 'C'),
         ('DfourvsD', 'D4', 'C+LKP'), ('DonevsD', 'D1', 'C+LKP'), ('DtwovsD', 'D2', 'C+LKP'), ('DthreevsD', 'D3', 'C+LKP'),
         ('DzerovsD', 'D0', 'C+LKP'), ('DtwovsDone', 'D2', 'D1'), ('DthreevsDtwo', 'D3', 'D2'), ('DzerovsDthree', 'D0', 'D3'),
         ('DonevsB', 'D1', 'B'), ('DtwovsB', 'D2', 'B'), ('DthreevsB', 'D3', 'B'), ('DfourvsB', 'D4', 'B'),
         ('AvsD', 'A', 'C+LKP'), ('DzerovsA', 'D0', 'A'),
         ('CritvsD', 'D+crit', 'C+LKP'), ('OrdervsD', 'D+order', 'C+LKP'), ('CritOrdervsD', 'D+crit+order', 'C+LKP'),
         ('CritOrdervsCrit', 'D+crit+order', 'D+crit'), ('CritOrdervsOrder', 'D+crit+order', 'D+order'),
         ('CritvsOrder', 'D+crit', 'D+order'), ('SharedvsD', 'D+shared', 'C+LKP'), ('CritvsB', 'D+crit', 'B'),
         ('CapturevsD', 'C+LPMdist', 'C+LKP'), ('CapturevsLP', 'C+LPMdist', 'C+LP'), ('MmaxvsB', 'C+Mmax', 'B'),
         ('LKvsL', 'C+LK', 'C+L'), ('LPvsL', 'C+LP', 'C+L'), ('LMmaxvsL', 'C+LMmax', 'C+L'), ('LPMmaxvsLP', 'C+LPMmax', 'C+LP'),
         ('OrdervsB', 'D+order', 'B'), ('CritOrdervsB', 'D+crit+order', 'B'),
         ('DfourCritvsCrit', 'D4+crit', 'D+crit'), ('DthreeCritvsCrit', 'D3+crit', 'D+crit'), ('DfourCritvsD', 'D4+crit', 'C+LKP'),
         ('DthreeCritvsD', 'D3+crit', 'C+LKP'), ('DthreeCritvsDthree', 'D3+crit', 'D3'), ('DfourCritvsDfour', 'D4+crit', 'D4'),
         ('DthreeCritvsB', 'D3+crit', 'B'), ('DfourCritvsB', 'D4+crit', 'B'),
         ('EsharedvsE', 'D4+crit+shared', 'D4+crit'), ('EsharedvsD', 'D4+crit+shared', 'C+LKP'), ('EsharedvsCrit', 'D4+crit+shared', 'D+crit'),
         ('SharedvsCrit', 'D+shared', 'D+crit')]
for tag, m1, m2 in PAIRS:
    put_pair(tag, m1, m2)
for m, tag in [('C+LKP', 'D'), ('B', 'B'), ('C', 'C'), ('D3', 'Dthree'), ('D2', 'Dtwo'), ('D1', 'Done'), ('D0', 'Dzero'),
               ('D4', 'Dfour'), ('D+crit', 'Crit'), ('A', 'A'), ('C+L', 'L'), ('D+order', 'Order'), ('D3+crit', 'DthreeCrit'),
               ('D4+crit', 'DfourCrit'), ('B-fixlapse', 'Bfix')]:
    if m in models:
        put(f'ploo{tag}', cmp_.loc[m, 'p_loo'], 0); put(f'elpd{tag}', cmp_.loc[m, 'elpd_loo'])
        if 'elpd_loso' in cmp_.columns and pd.notna(cmp_.loc[m, 'elpd_loso']):
            put(f'elpdLoso{tag}', cmp_.loc[m, 'elpd_loso'])

# effects and parameters
def eff_put(tag, m, hand, q):
    r = eff[(eff['model'] == m) & (eff['hand'] == hand) & (eff['quantity'] == q)]
    if len(r):
        r = r.iloc[0]; put(tag, r['mean'], 2 if 'JND' in q else 1); put(tag + 'Lo', r['lo'], 2 if 'JND' in q else 1); put(tag + 'Hi', r['hi'], 2 if 'JND' in q else 1)
for m, mt in [('C+LKP', 'D'), ('B', 'B'), ('C', 'C'), ('C+LKPS', 'S'), ('D+crit', 'Crit'), ('D3', 'Dthree'), ('C+L', 'L'),
              ('D+order', 'Order'), ('D4', 'Dfour'), ('D4+crit', 'DfourCrit'), ('D3+crit', 'DthreeCrit'), ('C+LPMdist', 'Capture')]:
    for hand, ht in [('left', 'Left'), ('right', 'Right')]:
        for q, qt in [('PSE shift', 'Shift'), ('PSE shift @6um', 'ShiftSix'), ('PSE shift @32um', 'ShiftThirtytwo'),
                      ('JND shift', 'Jnd'), ('PSE uni', 'PseUni')]:
            eff_put(f'{qt}{mt}{ht}', m, hand, q)

def par_put(tag, m, p, nd=2):
    r = par[(par['model'] == m) & (par['parameter'] == p)]
    if len(r):
        r = r.iloc[0]; put(tag, r['mean'], nd); put(tag + 'Lo', r['lo'], nd); put(tag + 'Hi', r['hi'], nd)
for m, mt in [('C+LKP', 'D'), ('C+LKPS', 'S'), ('D+crit', 'Crit'), ('D+order', 'Order'), ('D+crit+order', 'CritOrder'),
              ('D+shared', 'Shared'), ('C+Mmax', 'Mmax'), ('C', 'C'), ('D3', 'Dthree'), ('C+LPMdist', 'Capture'),
              ('C+LMmax', 'LMmax'), ('D4', 'Dfour'), ('D1', 'Done'), ('D3+crit', 'DthreeCrit'), ('D4+crit', 'DfourCrit'),
              ('D4+crit+shared', 'Eshared')]:
    for hand, ht in [('left', 'Left'), ('right', 'Right')]:
        for p, pt_ in [('mu_b2', 'muBtwo'), ('sig_b2', 'sigBtwo'), ('mu_b3', 'muBthree'), ('kappa_h', 'kappaH'), ('kappa_l', 'kappaL'),
                       ('mu_db1', 'muDbone'), ('lam', 'lam'), ('tau', 'tau'), ('pi', 'pi'), ('sig_b0', 'sigBzero'),
                       ('sig_b1', 'sigBone'), ('mu_b1', 'muBone')]:
            par_put(f'{pt_}{mt}{ht}', m, f'{p} {hand}')
    par_put(f'sigShared{mt}', m, 'sig_shared')
for g, gt in [('left_bi', 'Left'), ('right_bi', 'Right')]:
    par_put(f'sigGamB{gt}', 'B', f'sig_gam_h {g}'); par_put(f'sigGamBwide{gt}', 'B-wide', f'sig_gam_h {g}')

# size of the criterion trend: change of a typical session's unimanual PSE from the 6 to the 32 um sessions
# (D+crit) or from the first to the last session (D+order), in um, from the posterior draws
import arviz as az
from models import X_SIG
def trend_um(m, var, span):
    f = FITS / f'{m}.nc'
    if not f.exists():
        return
    post = az.from_netcdf(f).posterior
    for hand, ht in [('left', 'Left'), ('right', 'Right')]:
        v = post[var].sel(groups=hand).values.ravel(); b1 = post['mu_betas'].sel(betas='b1', groups=hand).values.ravel()
        x = -v * span / b1 * X_SIG
        lo, hi = np.quantile(x, [0.025, 0.975])
        tag = {'D+crit': 'critPse', 'D+order': 'orderPse', 'D3+crit': 'critPseDthree', 'D4+crit': 'critPseDfour'}[m] + ht
        put(tag, float(x.mean())); put(tag + 'Lo', float(lo)); put(tag + 'Hi', float(hi))
        print(f'{m} {hand}: PSE change {x.mean():.2f} [{lo:.2f}, {hi:.2f}] um')
trend_um('D+crit', 'lam', 26 / X_SIG)
trend_um('D+order', 'tau', 4.1)
trend_um('D3+crit', 'lam', 26 / X_SIG)
trend_um('D4+crit', 'lam', 26 / X_SIG)

# Monte Carlo check of LOSO: a second random seed for a few models
seed_d = []
for m in ['B', 'C+LKP', 'D+crit', 'D4+crit']:
    f2 = FITS / f'{m}.loso_s2.pkl'
    if f2.exists() and loso_i(m) is not None:
        seed_d.append(abs(pickle.load(open(f2, 'rb'))['loo'].loo_i.values.sum() - loso_i(m).sum()))
if seed_d:
    put('losoSeedN', len(seed_d)); put('losoSeedMax', float(max(seed_d)), 2)
    print('second LOSO seed: max change of the total', max(seed_d))

# capture fractions in percent
for hand, ht in [('left', 'Left'), ('right', 'Right')]:
    for m, mt in [('C+LPMdist', 'Capture'), ('C+Mmax', 'Mmax'), ('C+LMmax', 'LMmax'), ('C+LPMmax', 'LPMmax'), ('C+Mdist', 'Mdist')]:
        r = par[(par['model'] == m) & (par['parameter'] == f'pi {hand}')]
        if len(r):
            r = r.iloc[0]; put(f'pi{mt}Pct{ht}', 100 * r['mean']); put(f'pi{mt}Pct{ht}Lo', 100 * r['lo']); put(f'pi{mt}Pct{ht}Hi', 100 * r['hi'])

# posterior predictive check counts
for m, mt in [('B', 'B'), ('C', 'C'), ('C+LKP', 'D'), ('C+L', 'L'), ('D+crit', 'Crit'), ('D3', 'Dthree'), ('D1', 'Done'),
              ('D4+crit', 'E'), ('C+LPMdist', 'Capture')]:
    sub = ppc[ppc['model'] == m]
    if len(sub):
        put(f'ppcIn{mt}', int(sub['inside'].sum())); put('ppcN', len(sub))
lc = ppc[(ppc['hand'] == 'left') & (ppc['check'] == 'high @ stim<=12') & (ppc['distractor'].isin(['6', '12']))]
if len(lc):
    put('ppcLowAmpN', int(lc[lc['model'] == 'B']['n'].sum())); put('ppcLowAmpK', int(lc[lc['model'] == 'B']['k'].sum()))
    for m, mt in [('B', 'B'), ('C+L', 'L'), ('C+LKP', 'D')]:
        s_ = lc[lc['model'] == m]
        if len(s_):
            put(f'ppcLowAmpPred{mt}', 100 * float((s_['pred_mean'] * s_['n']).sum() / s_['n'].sum()))

# diagnostics summary
dmax = max(d['max_r_hat'] for d in diag.values()); put('maxRhatAll', dmax, 3)
put('minEssAll', min(d['min_ess_bulk'] for d in diag.values()), 0)
put('totalDiv', sum(d['divergences'] for d in diag.values()))
put('nModels', len(models))
put('totalHours', sum(d['minutes'] for d in diag.values()) / 60)
with open(TEX / 'numbers.tex', 'w') as f:
    for k, v in mac.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(f'{len(mac)} macros; tables in {TEX}')
