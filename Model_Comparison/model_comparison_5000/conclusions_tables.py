"""Numbers and tables for the short conclusions PDF: every listed model against Model E (both scores), Model E's
hand-level posterior summaries, and derived quantities. Writes RESDIR/tex/concl_*.tex (default results5000).
Usage: conclusions_tables.py [fits dir]"""
import os, sys, pickle, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az
from models import load_data, X_SIG

FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
RES = Path(os.environ.get('RESDIR', BASE / 'results5000'))
TEX = RES / 'tex'; TEX.mkdir(parents=True, exist_ok=True)
REC = 'D4+crit'
MINUS = r'\ensuremath{-}'
data = load_data(); sess = data['sess']; nS = len(data['sessions'])
cmp_ = pd.read_csv(RES / 'comparison.csv').set_index('model')
cnt = pd.read_csv(RES / 'param_counts.csv').set_index('model')
eff = pd.read_csv(RES / 'effects.csv')


def loo_i(m):
    return np.asarray(pickle.load(open(FITS / f'{m}.loo.pkl', 'rb'))['loo_i'], float)


def loso_i(m):
    return pickle.load(open(FITS / f'{m}.loso_s1.pkl', 'rb'))['loo'].loo_i.values


def pair(m1, m2):
    d = loo_i(m1) - loo_i(m2); cl = pd.Series(d).groupby(sess).sum()
    e = loso_i(m1) - loso_i(m2)
    return d.sum(), np.sqrt(nS * cl.var(ddof=1)), e.sum(), np.sqrt(nS * np.var(e, ddof=1))


def num(v, nd=1):
    t = f'{float(v):.{nd}f}'
    if t.startswith('-') and float(t) == 0:
        t = t[1:]
    return t.replace('-', MINUS)


mac = {}
def put(k, v, nd=1):
    mac[k] = num(v, nd) if isinstance(v, (float, np.floating)) else str(v)


# ---- Table: the five tied models, the simplest trend model, and the references, all minus E ---------------
ROWS = [('D4+crit', 'E', 'D with fixed distractor terms + trend'),
        ('D4+crit+shared', 'E+shared', 'E + daily bias shared by both hands'),
        ('D+crit', 'D+crit', 'D + trend with distractor amplitude'),
        ('D+order', 'D+order', 'D + trend linear in time'),
        ('D+crit+order', 'D+crit+order', 'D + both trends'),
        ('D+shared', 'D+shared', 'D + daily bias shared by both hands'),
        ('D3+crit', 'D3+crit', 'E, only the bias varies by session'),
        ('C+LKP', 'D', 'previous recommendation (no trend)'),
        ('D4', 'D4', 'E without the trend'),
        ('B', 'B', "Vera's Model B"),
        ('C', 'C', "Vera's Model C")]
N_TOP = 5
lines = [r'\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xrrr@{\ }lr@{\ }l@{}}', r'\toprule',
         r'Model & What it is & \makecell[r]{Parameters\\per session} & $p_\mathrm{loo}$ & \multicolumn{2}{c}{\makecell{Trial-level\\minus E (SE)}} & \multicolumn{2}{c}{\makecell{New sessions\\minus E (SE)}} \\',
         r'\midrule']
for i, (m, nm, desc) in enumerate(ROWS):
    if m == REC:
        t = ['0', '', '0', '']
    else:
        a, ase, b, bse = pair(m, REC)
        t = [num(a), f'({ase:.1f})', num(b), f'({bse:.1f})']
        tag = {'D4+crit+shared': 'Eshared', 'D+crit': 'Crit', 'D+order': 'Order', 'D+crit+order': 'CritOrder',
               'D+shared': 'Dshared', 'D3+crit': 'DthreeCrit', 'C+LKP': 'D', 'D4': 'Dfour', 'B': 'B', 'C': 'C'}[m]
        put(f'cn{tag}vsETrial', a); put(f'cn{tag}vsETrialSE', ase); put(f'cn{tag}vsELoso', b); put(f'cn{tag}vsELosoSE', bse)
        put(f'cnEminus{tag}Trial', -a); put(f'cnEminus{tag}Loso', -b)
    if i == N_TOP:
        lines.append(r'\midrule')
    name = r'\textbf{E}' if m == REC else nm
    lines.append(f"{name} & {desc} & {cnt.loc[m, 'per_session']:.0f} & {cmp_.loc[m, 'p_loo']:.0f} & {t[0]} & {t[1]} & {t[2]} & {t[3]} " + r'\\')
lines += [r'\bottomrule', r'\end{tabularx}']
(TEX / 'concl_models.tex').write_text('\n'.join(lines) + '\n')
# spread of the tied group
top = ['D4+crit', 'D4+crit+shared', 'D+crit', 'D+order', 'D+crit+order']
put('cnTopSpreadTrial', cmp_.loc[top, 'elpd_loo'].max() - cmp_.loc[top, 'elpd_loo'].min())
put('cnTopSpreadLoso', cmp_.loc[top, 'elpd_loso'].max() - cmp_.loc[top, 'elpd_loso'].min())
for m, tag in [('D4+crit', 'E'), ('D+crit', 'Crit'), ('B', 'B'), ('C+LKP', 'D')]:
    put(f'cnPloo{tag}', cmp_.loc[m, 'p_loo'], 0)
# the trend: E against D4 (same model without it)
a, ase, b, bse = pair('D4+crit', 'D4')
put('cnTrendTrial', a); put('cnTrendTrialSE', ase); put('cnTrendLoso', b); put('cnTrendLosoSE', bse)
put('cnTrendNbetter', int((loso_i('D4+crit') - loso_i('D4') > 0).sum()))
for m, tag in [('D+crit', 'Crit'), ('D+order', 'Order'), ('C+LKP', 'D')]:
    put(f'cnPloo{tag}', cmp_.loc[m, 'p_loo'], 0)
put('cnPlooEshared', cmp_.loc['D4+crit+shared', 'p_loo'], 0)
# bimanual lapses: own lapse rates (C+L vs C) and their amplitude dependence (D vs D without kappa = C+LP)
for m1, m2, tag in [('C+L', 'C', 'Lapse'), ('C+LKP', 'C+LP', 'Kappa'), ('C+LPMdist', 'C+LKP', 'CapDist'), ('C+LPMmax', 'C+LKP', 'CapMax')]:
    a, ase, b, bse = pair(m1, m2)
    put(f'cn{tag}Trial', a); put(f'cn{tag}TrialSE', ase); put(f'cn{tag}Loso', b); put(f'cn{tag}LosoSE', bse)
    put(f'cn{tag}Nbetter', int((loso_i(m1) - loso_i(m2) > 0).sum()))

# ---- Model E: hand-level posterior summaries (mean and 95% HDI) ------------------------------------------
post = az.from_netcdf(FITS / f'{REC}.nc').posterior
def draws(var, **sel):
    return post[var].sel(**sel).values.ravel()
def summ(v, nd=2):
    lo, hi = az.hdi(v, hdi_prob=0.95)
    return f'{num(v.mean(), nd)} [{num(lo, nd)}, {num(hi, nd)}]'
P = [(r'$\mu_{0,h}$', 'mean session bias at $d_s=\\bar d$', lambda h: draws('mu_betas', betas='b0', groups=h)),
     (r'$\sigma_{0,h}$', 'session-to-session sd of the bias', lambda h: draws('sig_betas', betas='b0', groups=h)),
     (r'$\lambda_h$', 'criterion trend per unit of $d_s$', lambda h: draws('lam', groups=h)),
     (r'$\mu_{1,h}$', 'mean slope', lambda h: draws('mu_betas', betas='b1', groups=h)),
     (r'$\sigma_{1,h}$', 'session-to-session sd of the slope', lambda h: draws('sig_betas', betas='b1', groups=h)),
     (r'$\beta_{2,h}$', 'distractor amplitude term', lambda h: draws('mu_betas', betas='b2', groups=h)),
     (r'$\beta_{3,h}$', 'distractor presence term', lambda h: draws('mu_b3', groups=h)),
     ]
for k, word in [('gam_h', 'hi'), ('gam_l', 'lo')]:
    W = "``high''" if word == 'hi' else "``low''"
    P += [(rf'$\mu^{{\mathrm{{{word}}}}}_h$', f'mean unimanual {W} lapse logit', lambda h, k=k: draws('mu_gams', lapses=k, groups=h)),
          (rf'$\sigma^{{\mathrm{{{word}}}}}_h$', f'its session-to-session sd', lambda h, k=k: draws('sig_gams', lapses=k, groups=h))]
for k, word in [('gam_h', 'hi'), ('gam_l', 'lo')]:
    W = "``high''" if word == 'hi' else "``low''"
    P += [(rf'$\mu^{{\mathrm{{{word},bi}}}}_h$', f'mean bimanual {W} lapse logit at $d_s=\\bar d$', lambda h, k=k: draws('mu_gams_bi', lapses=k, groups=h)),
          (rf'$\sigma^{{\mathrm{{{word},bi}}}}_h$', f'its session-to-session sd', lambda h, k=k: draws('sig_gams_bi', lapses=k, groups=h)),
          (rf'$\kappa^{{\mathrm{{{word}}}}}_h$', f'change per unit of $d_s$', lambda h, k=k: draws('kappa', lapses=k, groups=h))]
lines = [r'\begin{tabular}{@{}llcc@{}}', r'\toprule', r'Parameter & Meaning & Left hand & Right hand \\', r'\midrule']
for sym, meaning, fn in P:
    lines.append(f"{sym} & {meaning} & {summ(fn('left'))} & {summ(fn('right'))} " + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TEX / 'concl_params.tex').write_text('\n'.join(lines) + '\n')
for h, H in [('left', 'Left'), ('right', 'Right')]:
    for var, sel, tag in [('mu_b3', {}, 'bthree'), ('lam', {}, 'lam'), ('mu_betas', {'betas': 'b2'}, 'btwo')]:
        v = draws(var, groups=h, **sel); lo, hi = az.hdi(v, hdi_prob=0.95)
        put(f'cnE{tag}{H}', v.mean(), 2); put(f'cnE{tag}{H}Lo', lo, 2); put(f'cnE{tag}{H}Hi', hi, 2)
    for k, kt in [('gam_h', 'Hi'), ('gam_l', 'Lo')]:
        v = draws('kappa', lapses=k, groups=h); lo, hi = az.hdi(v, hdi_prob=0.95)
        put(f'cnEkappa{kt}{H}', v.mean(), 2); put(f'cnEkappa{kt}{H}Lo', lo, 2); put(f'cnEkappa{kt}{H}Hi', hi, 2)

# ---- derived quantities ---------------------------------------------------------------------------------
d_bar = float(np.mean(data['sess_d']))
put('cnSigA', X_SIG, 2); put('cnDbar', d_bar, 3); put('cnDbarUm', d_bar * X_SIG, 1)
def rate(logit):
    return 100 * 0.25 / (1 + np.exp(-logit))
for h, H in [('left', 'Left'), ('right', 'Right')]:
    for k, kt in [('gam_h', 'Hi'), ('gam_l', 'Lo')]:
        r = rate(draws('mu_gams', lapses=k, groups=h)); put(f'cnUni{kt}{H}', r.mean(), 1)
        for amp in (6, 32):
            r = rate(draws('mu_gams_bi', lapses=k, groups=h) + draws('kappa', lapses=k, groups=h) * (amp / X_SIG - d_bar))
            lo, hi = az.hdi(r, hdi_prob=0.95)
            put(f'cnBi{kt}{H}{"Six" if amp == 6 else "Thirtytwo"}', r.mean(), 1)
            put(f'cnBi{kt}{H}{"Six" if amp == 6 else "Thirtytwo"}Lo', lo, 1); put(f'cnBi{kt}{H}{"Six" if amp == 6 else "Thirtytwo"}Hi', hi, 1)
    # criterion change of a typical session between the 6 and the 32 um sessions: -26 lam / mu_1 um
    x = -26 * draws('lam', groups=h) / draws('mu_betas', betas='b1', groups=h)
    lo, hi = az.hdi(x, hdi_prob=0.95); put(f'cnCrit{H}', x.mean(), 1); put(f'cnCrit{H}Lo', lo, 1); put(f'cnCrit{H}Hi', hi, 1)
    for q, qt in [('PSE shift', 'Shift'), ('PSE shift @6um', 'ShiftSix'), ('PSE shift @32um', 'ShiftThirtytwo'), ('JND shift', 'Jnd')]:
        r = eff[(eff['model'] == REC) & (eff['hand'] == h) & (eff['quantity'] == q)].iloc[0]
        nd = 2 if q == 'JND shift' else 1
        put(f'cnE{qt}{H}', r['mean'], nd); put(f'cnE{qt}{H}Lo', r['lo'], nd); put(f'cnE{qt}{H}Hi', r['hi'], nd)
r = eff[(eff['model'] == 'B') & (eff['hand'] == 'left') & (eff['quantity'] == 'JND shift')].iloc[0]
put('cnBJndLeft', r['mean'], 2)
with open(TEX / 'concl_numbers.tex', 'w') as f:
    for k, v in mac.items():
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(len(mac), 'macros')
for k in ['cnTopSpreadTrial', 'cnTopSpreadLoso', 'cnTrendLoso', 'cnTrendLosoSE', 'cnTrendNbetter', 'cnBiHiLeftSix', 'cnBiHiLeftThirtytwo',
          'cnBiHiRightSix', 'cnBiHiRightThirtytwo', 'cnUniHiLeft', 'cnUniLoLeft', 'cnCritLeft', 'cnCritRight', 'cnDbar']:
    print(k, mac[k])
