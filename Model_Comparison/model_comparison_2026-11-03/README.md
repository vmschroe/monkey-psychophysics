# Model comparison for the Sirius bimanual-interference data

Prepared 3 Nov 2026 as a follow-up to a review of Models A–C (2 Nov 2026).
The scripts only read Vera's data files and build scripts from the repository root; nothing outside this folder is changed.

## Models

All models use the same 12,937 trials in the same order. Vera's models are built with her own scripts, unchanged.

| Model | Description |
|---|---|
| A | Vera's Model A: one curve per hand × condition, sessions pooled |
| B | Vera's Model B: hierarchical over sessions, unimanual and bimanual fit as separate groups |
| B-wide | Model B with the between-session sd of the lapse logits widened from HalfNormal(0.5) to HalfNormal(1.5) |
| C | Vera's Model C: hierarchical, one group per hand, shift `b2 * x_dist` |
| C+L | C with lapse rates per condition: bimanual trials get their own lapse mean and session variation |
| C+LP | C+L plus a distractor-presence term `b3` on bimanual trials (session-varying) |
| C+LPS | C+LP plus a bimanual slope change `db1` (session-varying) |
| C+LK | C+L with the bimanual lapse means depending on the session's distractor amplitude (`kappa`) |
| C+LKP | C+LK plus the presence term |
| C+LKPS | C+LKP plus the bimanual slope change |
| C+Mmax | C plus capture: on a fraction `pi` of bimanual trials the monkey judges the stronger of the two vibrations |
| C+Mdist | C plus capture of the distractor: on a fraction `pi` of bimanual trials the monkey judges the distractor |
| C+LMmax | C+L plus capture of the stronger vibration |
| C+LPMmax | C+LP plus capture of the stronger vibration |

Priors for the shared parts are Model C's. The new parts use: `mu_b3 ~ N(0,1)`, `sig_b3 ~ HalfNormal(1)`; `mu_db1 ~ N(0,1)`, `sig_db1 ~ HalfNormal(1)`; bimanual lapse means `~ N(-3, 1.5)` with sds `~ HalfNormal(0.5)`; `kappa ~ N(0,2)`; `logit(pi) ~ N(-2.5, 1.5)`.

## Comparisons

- **Trial-level PSIS-LOO**, as in Vera's reports. The standard error of each difference is given two ways: over trials, as Vera computed it, and clustered by session, which allows for the fact that a model's misfit is shared by the trials of a session.
- **Leave-one-session-out (LOSO)**: the predictive density of a whole held-out session, with its session parameters integrated out. It is computed without refitting: importance sampling over the session parameters, then PSIS over the hyperparameter draws (`loso.py`). Exact refits without one session agree with these estimates to within 0.2 elpd, including sessions with Pareto k up to 1.03 (`exact_loso.py`). Repeating the estimate with a different random seed changes the totals by less than 0.1.
- **Posterior predictive check** of the answers the models disagree about: weak-stimulus "high" and strong-stimulus "low" responses on bimanual trials, by distractor amplitude.

## Files

| File | Purpose |
|---|---|
| `models.py` | model builders (Vera's via her scripts; new variants) and a numpy version of every likelihood |
| `fit_model.py` | fits one model (4 chains × 1000 draws after 1000 tuning, target_accept 0.9, seed 123) and saves its posterior, LOO and diagnostics to `fits/` |
| `run_all.sh` | fits every model in turn (about 1.5 h on 2 cores) |
| `loso.py` | leave-one-session-out for one model: `python loso.py C+LKP` |
| `exact_loso.py` | brute-force check of one session: `python exact_loso.py C+L 7` |
| `compare.py` | LOO and LOSO tables, with differences by distractor amplitude |
| `effects.py` | PSE and JND shifts, key parameters, posterior predictive check |
| `recover_A_lapses.py` | Model A's synthetic study rerun at its real-data values |
| `plot_comparison.py` | the summary figure |
| `export_pointwise.py` | writes the per-trial and per-session values to `results/` |
| `results/` | all tables (`.csv`), the figure, and per-trial / per-session LOO values for every model |

The posterior files (about 0.5 GB) are not included; `run_all.sh` regenerates them. The scripts find the repository two levels up (or set `MP_REPO`). They need the repo's Python requirements plus `h5netcdf`.
