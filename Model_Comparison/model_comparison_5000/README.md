# Sirius model comparison with 4 × 5000 draws

Prepared 3 Nov 2026 as the follow-up to `model_comparison_2026-11-03/`. The report is
`Sirius_model_comparison_5000_draws.pdf`. The scripts only read Vera's data files and
build scripts from the repository root; nothing outside this folder is changed.

## What is here

| Path | Contents |
|---|---|
| `Sirius_model_comparison_5000_draws.pdf` | the report |
| `Sirius_conclusions_and_model_E.pdf` | short note: the conclusions, how confident they are, and Model E's equations and priors (sources: `report/conclusions.tex`, `conclusions_tables.py`; `check_E_equations.py` confirms the written equations reproduce the fitted model's pointwise LOO to 1e-14) |
| `results5000/comparison.csv` | every model: trial-level LOO, leave-one-session-out (LOSO), p_loo, sampling diagnostics |
| `results5000/pairwise_vs_D.csv` | every model minus Model D, both scores, standard errors from per-session sums |
| `results5000/vs_B_by_amplitude.csv` | every model minus Model B, by the session's distractor amplitude |
| `results5000/effects.csv`, `parameters.csv` | PSE and JND shifts, key parameters (mean and 95% interval) |
| `results5000/session_shifts.csv` | each session's unimanual PSE, bimanual PSE and shift, per model and hand |
| `results5000/ppc_lapses.csv` | posterior predictive check of the lapse-like answers |
| `results5000/pointwise_loo_by_trial.csv`, `pointwise_loso_by_session.csv` | per-trial and per-session scores of every model, so any comparison can be redone without the posteriors |
| `results5000/draws_1000_vs_5000.csv` | the earlier 1000-draw scores next to the new ones |
| `results5000/param_counts.csv` | parameters per model (hand level and per session) |
| `results5000/rt_check.txt`, `extension_checks.txt` | response times of errors; trial history, side bias, within-session drift |
| `results5000/figs/`, `results5000/tex/` | the figures, and the tables and numbers the report reads |
| `report/` | LaTeX sources of the report |

## Models

Vera's A, B and C are built from her own scripts. All other models are variants of C (`models.py`, `FAMILY`):
L (own bimanual lapses), K (bimanual lapses depend on the distractor amplitude), P (presence offset), S (bimanual
slope), M (capture: judges the stronger vibration or the distractor). Model D = C+LKP. Simpler versions of D hold
parameters fixed across sessions: D4 (distractor terms), D1 (lapses), D2 (both), D3 (also the slope), D0 (all).
Extensions of D: `+crit` (session bias follows the distractor amplitude), `+order` (linear drift over the study),
`+shared` (daily bias shared by both hands). **Model E = D4+crit** is the recommended model.

## Re-running

The scripts find the repository two levels up (or set `MP_REPO`). Use an environment with the repo's
requirements plus `nutpie`, `h5netcdf` and `statsmodels`, and point `PY` at its Python:

    PY=/path/to/python ./run_all_5000.sh

This fits every model (4 chains × 5000 draws after 1000 tuning steps, target_accept 0.9, seed 123), runs
leave-one-session-out for each, and rebuilds the tables, figures and PDF; about 5 hours on 2 cores. Single steps:
`fit5000.py <model>`, `loso.py <model> 1 fits5000`, then `compare2.py`, `effects2.py`, `report_tables.py`,
`figures.py`. The posterior files (about 200 MB per model) are not included.

`extension_checks.py` also needs two trial tables built during the first review (every row of the per-session trial
files, and Vera's selected trials with their row index); set `RAW_ALL` and `VERA_REBUILT` to them.
