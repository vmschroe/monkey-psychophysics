"""Figures for the report, from RESDIR/*.csv (default results5000) and the posteriors in the fits directory.
Usage: figures.py [fits dir]. Colours follow the model everywhere: B blue, C orange, D aqua, others grey."""
import os, sys, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, arviz as az, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from models import load_data, X_SIG, X_MU, FAMILY

FITS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / 'fits5000'
RES = Path(os.environ.get('RESDIR', BASE / 'results5000'))
FIG = RES / 'figs'; FIG.mkdir(exist_ok=True, parents=True)
data = load_data()
nS = len(data['sessions'])
sess_amp = np.round(data['sess_d'] * X_SIG).astype(int)
D_BAR = float(np.mean(data['sess_d']))
REF = 'C+LKP'          # Model D: the reference of the comparisons
REC = 'D4+crit'        # Model E: the recommended model

SURF, INK, INK2, GRID, GREY = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#9a9893'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
COL = {'B': BLUE, 'C': ORANGE, 'D4+crit': AQUA, 'C+LKP': INK2}
plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.titlesize': 8,
                     'axes.titlelocation': 'left', 'axes.titleweight': 'bold', 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False, 'lines.solid_capstyle': 'round',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

VERA = ['A', 'B', 'B-wide', 'B-fixlapse', 'C']
SIMPLER = ['D4', 'D1', 'D2', 'D3', 'D0']
EXT = ['D+shared', 'D+crit', 'D+order', 'D+crit+order', 'D4+crit', 'D3+crit', 'D4+crit+shared']
SHORT = {'C+LKP': 'D', 'D4+crit': 'E'}
LABEL = {'A': 'A   pooled, no session differences', 'B': 'B   separate uni / bi hierarchies',
         'B-wide': 'B-wide   wider lapse prior', 'B-fixlapse': 'B-fixlapse   lapses fixed',
         'C': 'C   shared curve, amplitude offset',
         'C+L': 'C+L   own bimanual lapses', 'C+LK': 'C+LK   C+L, lapses depend on amplitude',
         'C+LP': 'C+LP   own lapses + presence', 'C+LPS': 'C+LPS   C+LP + bimanual slope',
         'C+LKP': 'D = C+LKP', 'C+LKPS': 'C+LKPS   D + bimanual slope',
         'C+Mmax': 'C+Mmax   judges the stronger vibration', 'C+Mdist': 'C+Mdist   judges the distractor',
         'C+LMmax': 'C+LMmax   C+L + stronger vibration', 'C+LPMmax': 'C+LPMmax   C+LP + stronger vibration',
         'C+LPMdist': 'C+LPMdist   C+LP + judges the distractor',
         'D4': 'D4   distractor terms fixed', 'D1': 'D1   lapses fixed',
         'D2': 'D2   D1 and D4 combined', 'D3': 'D3   only the bias varies',
         'D0': 'D0   no session differences',
         'D+crit': 'D+crit   bias follows amplitude', 'D+order': 'D+order   bias drifts over time',
         'D+crit+order': 'D+crit+order   both', 'D+shared': 'D+shared   daily bias shared by hands',
         'D4+crit': 'E = D4+crit', 'D3+crit': 'D3+crit   D3 + bias follows amplitude',
         'D4+crit+shared': 'E+shared   E + daily bias shared by hands'}


def fam_marker(m):
    return 'o' if m in VERA else ('v' if m in SIMPLER else ('D' if m in EXT else 's'))


def style(ax, grid_axis='both'):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, linewidth=0.7, axis=grid_axis); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name, **kw):
    fig.savefig(FIG / f'{name}.pdf', **kw); fig.savefig(FIG / f'{name}.png', dpi=170, **kw); plt.close(fig)


cmp_ = pd.read_csv(RES / 'comparison.csv').set_index('model')
pw = pd.read_csv(RES / 'pairwise_vs_D.csv').set_index('model')       # written by report_tables.py
fam_handles = [plt.Line2D([], [], marker=mk, ls='', color=INK2, ms=5.5, label=lab) for mk, lab in
               [('o', "Vera's models"), ('s', 'variants of C'), ('v', 'simpler versions of D'), ('D', 'extensions of D')]]

# ---- F1: comparison against D, both criteria ----------------------------------------------------------
order = pw.sort_values('trial').index.tolist()
y = {m: i for i, m in enumerate(order)}
XMIN, XMAX = -72, 36
fig, axes = plt.subplots(1, 2, figsize=(7.2, (0.27 * len(order) * 0.8) + 1.5), sharey=True)
for ax, col, se, title in [(axes[0], 'trial', 'trial_se', 'Trial-level LOO'),
                           (axes[1], 'loso', 'loso_se', 'Leave-one-session-out')]:
    for m in order:
        v, s_ = pw.loc[m, col], pw.loc[m, se]
        if pd.isna(v):
            continue
        c = COL.get(m, GREY); z = 3 if m in COL else 2
        if v < XMIN:
            ax.annotate('', xy=(XMIN, y[m]), xytext=(XMIN + 7, y[m]), arrowprops=dict(arrowstyle='->', color=c, lw=1.4))
            ax.text(XMIN + 8, y[m], f'{v:.0f} ± {2 * s_:.0f}'.replace('-', '−'), va='center', fontsize=6.5, color=INK2)
            continue
        if m != REF:
            ax.plot([max(v - 2 * s_, XMIN), v + 2 * s_], [y[m], y[m]], color=c, lw=1.6, alpha=0.75, zorder=z, solid_capstyle='butt')
        ax.plot(v, y[m], marker=fam_marker(m), color=c, ms=6 if m in COL else 5, ls='', mec=SURF, mew=0.8, zorder=z + 1)
    ax.axvline(0, color=INK2, lw=1.0, ls='--', zorder=1)
    ax.set_xlim(XMIN, XMAX); style(ax, 'x')
    ax.set_title(title); ax.set_xlabel('elpd difference to Model D')
axes[0].set_yticks(range(len(order)), [LABEL.get(m, m) for m in order])
for t in axes[0].get_yticklabels():
    if t.get_text().split()[0] in ('B', 'C') or t.get_text().startswith(('D =', 'E =')):
        t.set_fontweight('bold')
axes[0].set_ylim(-0.7, len(order) - 0.3)
fig.legend(handles=fam_handles, loc='lower center', ncol=4, fontsize=6.8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 'F1_comparison')

# ---- F2: fit against complexity ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, (4.1) * 0.8), sharey=True)
YMIN = -62
F2_MODELS = ['A', 'B', 'B-fixlapse', 'C', REF, 'D4', 'D1', 'D2', 'D3', 'D0', 'D+crit', REC, 'D3+crit']
OFF = {'trial': {REF: (6, -9), 'D+crit': (6, 2), REC: (-12, 5), 'D4': (5, -8), 'D1': (5, -8), 'D2': (-4, 5), 'D3': (-4, -11),
                 'D3+crit': (-22, 5), 'B': (5, 3), 'C': (5, 3), 'B-fixlapse': (5, -9)},
       'loso': {REF: (6, -9), 'D+crit': (6, 2), REC: (-12, 5), 'D4': (-6, 5), 'D1': (5, -8), 'D2': (-4, 5), 'D3': (-4, -11),
                'D3+crit': (-12, 5), 'B': (5, 3), 'C': (5, 3), 'B-fixlapse': (5, -9)}}
for ax, col, title in [(axes[0], 'trial', 'Trial-level LOO'), (axes[1], 'loso', 'Leave-one-session-out')]:
    low = []
    for m in [m for m in F2_MODELS if m in pw.index]:
        v = pw.loc[m, col]
        if pd.isna(v):
            continue
        xv = cmp_.loc[m, 'p_loo']; c = COL.get(m, GREY)
        if v < YMIN:
            ax.annotate('', xy=(xv, YMIN + 0.5), xytext=(xv, YMIN + 7), arrowprops=dict(arrowstyle='->', color=c, lw=1.3))
            low.append(f"{SHORT.get(m, m)} {v:.0f}".replace('-', '−'))
            continue
        ax.plot(xv, v, marker=fam_marker(m), color=c, ms=6.5 if m in COL else 5.5, ls='', mec=SURF, mew=0.8, zorder=3)
    if low:
        ax.text(26, YMIN + 3.5, 'below the scale: ' + ', '.join(low), fontsize=6.5, color=INK2, va='center')
    for chain, ls_ in [(['D3', 'D2', 'D1', REF], '-'), (['D3+crit', REC, 'D+crit'], ':')]:
        lad = [m for m in chain if m in pw.index and pw.loc[m, col] >= YMIN]
        if len(lad) > 1:
            ax.plot(cmp_.loc[lad, 'p_loo'], pw.loc[lad, col], color=GREY, lw=1.0, ls=ls_, zorder=1)
    for m in ['B', 'C', REF, 'D4', 'D3', 'D2', 'D1', 'D+crit', REC, 'D3+crit', 'B-fixlapse']:
        if m in pw.index and pd.notna(pw.loc[m, col]) and pw.loc[m, col] >= YMIN:
            ax.annotate(SHORT.get(m, m), (cmp_.loc[m, 'p_loo'], pw.loc[m, col]), xytext=OFF[col].get(m, (5, 3)),
                        textcoords='offset points', fontsize=6.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1.0, ls='--', zorder=0); style(ax)
    ax.set_ylim(YMIN, 24); ax.set_xlim(0, 192)
    ax.set_xlabel('effective number of parameters ($p_\\mathrm{loo}$)'); ax.set_title(title)
axes[0].set_ylabel('elpd difference to Model D')
fig.legend(handles=fam_handles, loc='lower center', ncol=4, fontsize=6.8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, 'F2_complexity')

# ---- F3: lapse posterior predictive check ----------------------------------------------------------------
ppc = pd.read_csv(RES / 'ppc_lapses.csv')
SHOW = [('B', 'B'), ('C', 'C'), (REC, 'E')]
bins = ['6', '12', '16-22', '24-30', '32']
fig, axes = plt.subplots(2, 2, figsize=(7.2, (6.0) * 0.8), sharex=True)
for col_i, hand in enumerate(['left', 'right']):
    for row_i, (check, ylab) in enumerate([('high @ stim<=12', '"high" answers to 6-12 μm stimuli'),
                                           ('low @ stim>=44', '"low" answers to 44-50 μm stimuli')]):
        ax = axes[row_i, col_i]
        o = ppc[(ppc['model'] == 'B') & (ppc['hand'] == hand) & (ppc['check'] == check)].set_index('distractor').loc[bins]
        xpos = np.arange(len(bins))
        lo = np.nan_to_num(beta_dist.ppf(0.025, o['k'], o['n'] - o['k'] + 1), nan=0.0)
        hi = beta_dist.ppf(0.975, o['k'] + 1, o['n'] - o['k'])
        ax.errorbar(xpos - 0.2, 100 * o['observed'], yerr=[100 * (o['observed'] - lo), 100 * (hi - o['observed'])], fmt='s',
                    color=INK, ms=5, elinewidth=1.3, capsize=0, label='observed (95% CI)', zorder=5)
        for j, (m, lab) in enumerate(SHOW):
            p_ = ppc[(ppc['model'] == m) & (ppc['hand'] == hand) & (ppc['check'] == check)].set_index('distractor').loc[bins]
            ax.errorbar(xpos + 0.13 * j, 100 * p_['pred_mean'], yerr=[100 * (p_['pred_mean'] - p_['lo']), 100 * (p_['hi'] - p_['pred_mean'])],
                        fmt='o', color=COL[m], ms=4.5, elinewidth=1.3, capsize=0, label=f'Model {lab} predicts (95%)')
        ax.set_xticks(xpos, [f'{b} μm' for b in bins]); style(ax, 'y')
        ax.set_title(f'{hand} hand, bimanual trials'); ax.set_ylabel(ylab + ' (%)')
        ax.set_ylim(bottom=0)
        if row_i == 1:
            ax.set_xlabel("distractor amplitude of the session")
axes[0, 0].legend(fontsize=6.8, loc='upper left')
fig.tight_layout()
save(fig, 'F3_lapse_ppc')

# ---- F4: PSE shift against distractor amplitude ----------------------------------------------------------
ss = pd.read_csv(RES / 'session_shifts.csv')
grid_amp = np.linspace(6, 32, 60)


def post_draws(m, n=2000):
    post = az.from_netcdf(FITS / f'{m}.nc').posterior.stack(s=('chain', 'draw'))
    return post.isel(s=np.linspace(0, post.sizes['s'] - 1, n).astype(int))


def lapse_pse_term(gh, gl):
    """value of eta at which gh + (1-gh-gl) * invlogit(eta) = 0.5"""
    return np.log((0.5 - gh) / (0.5 - gl))


def typical_shift(m, hand):
    """bimanual - unimanual PSE (um) of a typical session (population means) at each distractor amplitude."""
    p = post_draws(m); cfg = FAMILY.get(m, {})
    mb = p['mu_betas'].sel(groups=hand).values; mg = p['mu_gams'].sel(groups=hand).values
    d = grid_amp[:, None] / X_SIG
    gh_u, gl_u = 0.25 / (1 + np.exp(-mg[0])), 0.25 / (1 + np.exp(-mg[1]))
    pse_u = (lapse_pse_term(gh_u, gl_u) - mb[0]) / mb[1]
    off = mb[0] + mb[2] * d + (p['mu_b3'].sel(groups=hand).values if cfg.get('P') else 0.0)
    if cfg.get('L'):
        mgb = p['mu_gams_bi'].sel(groups=hand).values
        kap = p['kappa'].sel(groups=hand).values if cfg.get('K') else np.zeros_like(mgb)
        gh_b = 0.25 / (1 + np.exp(-(mgb[0] + kap[0] * (d - D_BAR))))
        gl_b = 0.25 / (1 + np.exp(-(mgb[1] + kap[1] * (d - D_BAR))))
    else:
        gh_b, gl_b = gh_u, gl_u
    pse_b = (lapse_pse_term(gh_b, gl_b) - off) / mb[1]
    sh = (pse_b - pse_u) * X_SIG
    return sh.mean(1), np.quantile(sh, 0.025, 1), np.quantile(sh, 0.975, 1)


fig, axes = plt.subplots(1, 2, figsize=(7.2, (3.9) * 0.8), sharey=True)
rng = np.random.default_rng(1)
for ax, hand in zip(axes, ['left', 'right']):
    for m, lab, dx in [('B', 'B: each session', -0.4), (REC, 'E: each session', 0.4)]:
        s_ = ss[(ss['model'] == m) & (ss['hand'] == hand)]
        if len(s_):
            ax.plot(s_['dist_um'] + dx + rng.uniform(-0.25, 0.25, len(s_)), s_['shift'], 'o', color=COL[m], ms=3.6,
                    alpha=0.8, mec='none', label=lab)
    for m, ls, lab in [(REC, '-', 'E: typical session (95%)'), ('C', '--', 'C: typical session')]:
        if (FITS / f'{m}.nc').exists():
            mean, lo, hi = typical_shift(m, hand)
            ax.plot(grid_amp, mean, color=COL[m], ls=ls, lw=1.9, label=lab)
            if m == REC:
                ax.fill_between(grid_amp, lo, hi, color=COL[m], alpha=0.18, lw=0)
    ax.axhline(0, color=INK2, lw=0.9, ls=':'); style(ax)
    ax.set_title(f'{hand} hand'); ax.set_xlabel('distractor amplitude (μm)')
axes[0].set_ylabel('PSE shift, bimanual − unimanual (μm)')
axes[0].legend(fontsize=6.8, loc='lower left')
fig.tight_layout()
save(fig, 'F4_shift_vs_amplitude')

# ---- F5: where D and C differ from B -------------------------------------------------------------------------
vb = pd.read_csv(RES / 'vs_B_by_amplitude.csv').set_index('model')
SHOW5 = [m for m in ['C', REC] if m in vb.index]
nsess = [int(((sess_amp >= lo_) & (sess_amp <= hi_)).sum()) for lo_, hi_ in [(6, 6), (12, 12), (16, 22), (24, 30), (32, 32)]]
fig, axes = plt.subplots(1, 2, figsize=(7.2, (3.5) * 0.8), sharey=True)
for ax, pre, title in [(axes[0], 'trial', 'Trial-level LOO'), (axes[1], 'loso', 'Leave-one-session-out')]:
    w = 0.36
    for j, m in enumerate(SHOW5):
        vals = [vb.loc[m].get(f'{pre} {b}', np.nan) for b in bins]
        ax.bar(np.arange(len(bins)) + (j - 0.5) * w, vals, width=w * 0.94, color=COL[m], label=f'Model {SHORT.get(m, m)} − Model B')
    ax.axhline(0, color=INK2, lw=0.9); style(ax, 'y')
    ax.set_xticks(range(len(bins)), [f'{b} μm\n({n} sessions)' for b, n in zip(bins, nsess)])
    ax.set_xlabel("distractor amplitude of the session"); ax.set_title(title)
axes[0].set_ylabel('elpd difference to B, summed')
axes[0].legend(fontsize=6.8, loc='lower left')
fig.tight_layout()
save(fig, 'F5_vs_B_by_amplitude')

# ---- F6: session bias against time and amplitude --------------------------------------------------------------
if all((FITS / f'{m}.nc').exists() for m in ['D+crit', 'D+order']):
    t_idx = np.arange(nS)
    fig, axs = plt.subplots(3, 1, figsize=(7.2, (7.4) * 0.8), sharex=True, gridspec_kw={'height_ratios': [1, 2.3, 2.3], 'hspace': 0.28})
    ax0 = axs[0]
    ax0.bar(t_idx + 1, sess_amp, color=GREY, width=0.72)
    style(ax0, 'y'); ax0.set_ylabel('distractor (μm)'); ax0.set_yticks([0, 16, 32])
    ax0.set_title('Distractor amplitude of each session, in the order the sessions were run')
    s_ = ss[ss['model'] == REF]
    for i, hand in enumerate(['left', 'right']):
        ax = axs[i + 1]
        h_ = s_[s_['hand'] == hand].reset_index(drop=True)
        ax.errorbar(t_idx + 1, h_['pse_uni'], yerr=[h_['pse_uni'] - h_['pse_uni_lo'], h_['pse_uni_hi'] - h_['pse_uni']], fmt='o',
                    color=GREY, ms=3.8, elinewidth=1.0, capsize=0, label='Model D (no trend), each session', zorder=3)
        for m, ls, lab in [('D+crit', '-', 'D+crit: follows the distractor amplitude'), ('D+order', '--', 'D+order: drifts over time')]:
            p = post_draws(m)
            mb = p['mu_betas'].sel(groups=hand).values; mg = p['mu_gams'].sel(groups=hand).values
            gh_u, gl_u = 0.25 / (1 + np.exp(-mg[0])), 0.25 / (1 + np.exp(-mg[1]))
            shift = (p['lam'].sel(groups=hand).values * (data['sess_d'][:, None] - D_BAR) if m == 'D+crit'
                     else p['tau'].sel(groups=hand).values * ((t_idx[:, None] - (nS - 1) / 2) / 10))
            pse = (lapse_pse_term(gh_u, gl_u) - (mb[0] + shift)) / mb[1] * X_SIG + X_MU
            ax.plot(t_idx + 1, pse.mean(1), color=INK, ls=ls, lw=1.6, label=lab)
        ax.axhline(28, color=INK2, lw=0.8, ls=':')
        ax.text(nS + 0.6, 27.85, 'trained boundary', fontsize=6.5, color=INK2, ha='right', va='top')
        style(ax); ax.set_ylim(21, 39.5)
        ax.set_title(f'{hand} hand: unimanual PSE of each session'); ax.set_ylabel('unimanual PSE (μm)')
    h_, l_ = axs[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.04))
    ticks = list(range(0, nS, 4))
    axs[2].set_xticks([t + 1 for t in ticks], [data['sessions'][t] for t in ticks])
    axs[2].set_xlim(0.3, nS + 0.7)
    axs[2].set_xlabel('session date (month-day, 2023)')
    save(fig, 'F6_bias_time_amplitude', bbox_inches='tight')
print('figures written to', FIG)
