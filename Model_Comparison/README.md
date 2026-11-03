# Model comparison for the Sirius data

Analysis by Claude for Krešimir Josić, 2–3 Nov 2026. Vera's Models A, B and C and variants of Model C are fit to
the same 12,937 trials and scored two ways: trial-level PSIS-LOO, as in Vera's reports, and leave-one-session-out
cross-validation, which asks how well a model predicts a session it has not seen.

## Start here

1. `model_comparison_5000/Sirius_conclusions_and_model_E.pdf`: 6 pages. Which model does best, how confident we
   can be, and the recommended Model E written out exactly (likelihood, priors, posterior summaries).
2. `model_comparison_5000/Sirius_model_comparison_5000_draws.pdf`: the full report (28 models, 4 chains × 5000 draws).
3. `model_comparison_5000/README.md`: the files, the result tables, and how to rerun.
4. `model_comparison_2026-11-03/`: the first comparison (14 models, 4 chains × 1000 draws). The 5000-draw runs
   supersede its numbers, but it holds two checks that were not repeated: exact leave-one-session-out refits
   (`exact_loso.py`) and Model A's synthetic study rerun at its real lapse values (`recover_A_lapses.py`).

## Model names

"Model D" in these documents is C+LKP: Model C with separate, amplitude-dependent bimanual lapse rates and a
distractor-presence term. "Model E" is D with the distractor terms fixed per hand, plus a trend in the session bias
(`D4+crit` in the code). Both names were chosen for this comparison and are unrelated to the `Model_D/` folder
started on 30 Oct.

## Rerunning

The scripts read the data and Vera's build scripts (`Model_A/`, `Model_B/`, `Model_C/`) from the repository root,
two levels up; set `MP_REPO` to use another copy. They need the repo's requirements plus `nutpie`, `h5netcdf` and
`statsmodels`. The posterior files are not included (about 0.5 GB for the 1000-draw runs, about 200 MB per model
for the 5000-draw runs); `run_all.sh` and `run_all_5000.sh` regenerate them, and the per-trial and per-session
scores in `results/` and `results5000/` allow any comparison to be redone without them.
