"""Dot-and-whisker summary of the model comparison: elpd difference to the best model, with
+-2 standard errors, for trial-level PSIS-LOO (SE clustered by session) and leave-one-session-out."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(__file__).resolve().parent / 'results'
trial = pd.read_csv(RES / 'loo_trial.csv')
loso = pd.read_csv(RES / 'loso.csv')
VERA = {'A', 'B', 'B-wide', 'C'}
LABELS = {'A': 'A  pooled', 'B': 'B  hierarchical, uni/bi separate', 'B-wide': 'B  with wider lapse-sd prior',
          'C': 'C  amplitude covariate',
          'C+L': 'C + lapses per condition', 'C+LP': 'C + lapses per condition + presence',
          'C+LPS': 'C + lapses per cond. + presence + bi slope',
          'C+LK': 'C + bimanual lapses grow with amplitude', 'C+LKP': 'C + amplitude lapses + presence',
          'C+LKPS': 'C + amplitude lapses + presence + bi slope',
          'C+Mmax': 'C + capture: judge the stronger vibration', 'C+Mdist': 'C + capture: judge the distractor',
          'C+LMmax': 'C + lapses per cond. + capture (stronger)', 'C+LPMmax': 'C + lapses per cond. + presence + capture'}
XMIN = -75
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {True: '#2a78d6', False: '#eb6834'}     # Vera's models / new variants (categorical slots 1, 2)

order = trial.sort_values('elpd_loo')['model'].tolist()     # best at the top
y = {m: i for i, m in enumerate(order)}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True, facecolor=SURF)
panels = [(axes[0], trial, 'diff_vs_best', 'se_diff_session', 'Trial-level LOO (Vera\'s comparison)\nerror bars: ±2 SE, clustered by session'),
          (axes[1], loso, 'diff_vs_best', 'se_diff', 'Leave-one-session-out\nerror bars: ±2 SE over sessions')]
for ax, df, col, secol, title in panels:
    ax.set_facecolor(SURF)
    for _, r in df.iterrows():
        if r['model'] not in y:
            continue
        c = COL[r['model'] in VERA]
        if r[col] - 2 * r[secol] < XMIN:          # off scale (Model A): marker at the edge, value written
            ax.plot(XMIN + 1.5, y[r['model']], marker='<', color=c, markersize=8, linestyle='')
            ax.text(XMIN + 4, y[r['model']], f"{r[col]:.0f} ± {2 * r[secol]:.0f} (off scale)", va='center',
                    fontsize=9, color=INK2)
            continue
        ax.errorbar(r[col], y[r['model']], xerr=2 * r[secol], fmt='o', color=c, markersize=7,
                    elinewidth=2, capsize=0, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=1, linestyle='--')
    ax.set_xlim(XMIN, 10)
    ax.grid(axis='x', color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.set_xlabel('elpd difference to the best model (higher is better)', color=INK2, fontsize=10)
axes[0].set_yticks(range(len(order)), [LABELS.get(m, m) for m in order], color=INK, fontsize=10)
handles = [plt.Line2D([], [], marker='o', linestyle='', color=COL[True], markersize=7, label="Vera's models"),
           plt.Line2D([], [], marker='o', linestyle='', color=COL[False], markersize=7, label='new variants of C')]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle('Model comparison on the Sirius data (12,937 trials, 42 sessions)', color=INK, fontsize=13, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(RES / 'model_comparison.png', dpi=200, facecolor=SURF)
print('saved')
