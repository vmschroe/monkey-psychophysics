# Monkey psychophysics: bimanual perceptual interactions

Bayesian models (PyMC) of a monkey's (Sirius) vibrotactile 2AFC intensity
discrimination task. A stimulus of 6–50 µm is delivered to one hand and the
monkey responds "low" (correct for 6, 12, 18, 24 µm) or "high" (correct for 32,
38, 44, 50 µm). On bimanual trials a distractor is delivered to the other hand
at the same time; its amplitude (6–32 µm) is fixed within a session. The
question is how the distractor shifts the psychometric function.

## Layout

```
Data/            Sirius_Data.pkl (cleaned trials + session summary),
                 Data_Conversion.ipynb (builds it), raw/ (session .xlsx files)
Model_A/ … Model_D/   one folder per model (see below)
Presentations/   ABC/ (full Beamer deck, Models A–C), Short/ (~30 min cut of ABC)
Notes/           research updates kept for reference (typeset + handwritten)
Mathematica/     symbolic work on the optimal decision threshold
```

Each model folder follows roughly the same pattern (letter X = A, B, C, D; not
every model has every file):

| File | Role |
|---|---|
| `DataProcessing_X.py` | turns `Data/Sirius_Data.pkl` into the model's inputs in `Data_X/` (real and synthetic) |
| `Build_Model_X.py` | the PyMC model |
| `Check_Prior_X.py` | prior predictive checks |
| `Main_Analysis_X.py` | real-data fit, diagnostics, plots; saves `Results_X.pkl` |
| `ParamRecov_X.py`, `Convergence_X.py`, `Model_Recov_*.py` | parameter recovery, convergence study, model recovery on synthetic data |
| `PPC_X.py` | posterior predictive checks |
| `*_Report.tex/.pdf` | LaTeX reports; figures come from `Plots_X/` |

Side analyses live in subfolders: `Model_A/Learning_A/` (first vs. second half
of each session) and `Model_B/Exploratory/` (PSE movement within and across
sessions).

## Models

All models use a logistic psychometric function with lapse rates,

    P("high") = gam_h + (1 - gam_h - gam_l) * invlogit(eta)

and differ in how `eta` is built and how sessions are pooled.

- **Model A: pooled additive.** `eta = b0 + b1*x_stim`. Complete pooling over
  sessions; four independent fits by hand (left/right) × condition
  (unimanual/bimanual).
- **Model B: hierarchical additive.** Same partition as A, with partial pooling
  of the parameters over sessions.
- **Model C: hierarchical, distractor as covariate.** `eta = b0 + b1*x_stim +
  b2*x_dist` (`x_dist = 0` on unimanual trials); one fit per hand, partial pooling
  over sessions.
- **Model D: in development** (see `Model_D/Model_D_Spec.txt`).
  `eta = b0[s,j] + b1[s,j]*x_stim + b2[j]*x_dist + b3[j]*x_bi`: session-level
  intercept and slope, distractor terms per hand. Developed under Betancourt's
  principled Bayesian workflow; no fit yet.
  **Not done yet:** the prior checks are a work in progress. The committed
  outputs of `Check_Prior_D.py` (`Prior_Check_D_Results.txt`,
  `Plots_D/Prior_Plots_D/`) are from the 2026-11-17 version, and several
  checks still come out "too wide", so the priors aren't final.

## Reproducing

Install with `pip install -r requirements.txt` (exact versions in
`requirements-lock-linux.txt`). Then:

1. `Data/Data_Conversion.ipynb` (run from `Data/`) → `Data/Sirius_Data.pkl`.
2. `Model_X/DataProcessing_X.py` → `Model_X/Data_X/` (Model A's is
   `Model_A/Data_A/DataProcessing_A.py`). The C and D versions work from any
   directory. The A and B versions still use older relative paths: run them from
   the repo root; A writes with Windows-style paths and B writes to the working
   directory, so their outputs may need moving into `Data_A/` / `Data_B/`.
3. `Model_X/Main_Analysis_X.py` and the recovery/convergence scripts → results
   and `Plots_X/`.
4. Compile reports and decks with `pdflatex` (twice) from their own folders.

Old drafts and early exploratory code were removed on 2026-11-18; they are
recoverable from the git tag `pre-cleanup`.
