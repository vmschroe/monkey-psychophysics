# Presentation outline

Editable outline of `Presentation.tex` and `sections/*.tex`. One `##` heading per slide.
`TODO:` marks the red placeholder boxes. `Figure:` names the image on a figure slide,
`Caption:` is the line under it. Math is written as in LaTeX.
`### Subsection:` marks a subsection (it appears in the table of contents). `[gray] ... [end gray]` marks text shown in light gray, `[blue] ... [end blue]` text in blue, `[red]`/`[amber]` likewise.

---

## 1. Title slide
- Title: Modeling Bimanual Perceptual Interactions in Macaque Monkeys
- Subtitle: Bayesian Hierarchical Modeling
- Author: Vera Schroeder
- Date: October 29, 2026

## 2. Outline
- (automatic table of contents from the section names)

---

# Section: Task and Data

## 3. The task: 2AFC categorization of vibration amplitude
- stimulus $x_\text{stim} \in \{6, 12, 18, 24, 32, 38, 44, 50\}$ μm
  - administered to one finger of one hand (left or right)
- [gray] distractor $x_\text{dist} \in \{0, x_\text{d}\}$ μm
  - $x_\text{d} \in \{6, 12, 18, 24, 32\}$ μm, value fixed for each session
  - administered to the other (non-stimulus) hand
- if present, distractor administered simultaneously with the stimulus [end gray]
- monkey categorizes the stimulus as either "high" or "low" relative to a trained boundary of 28 μm
  - "low" response: correct for $x_\text{stim} < 28$ μm
  - "high" response: correct for $x_\text{stim} > 28$ μm
- correct responses are rewarded with water or juice

## 4. Psychometric function: response $r \sim \text{Bernoulli}(p = \psi)$
  $\psi(x_\text{stim}) = \gamma_h + (1 - \gamma_h - \gamma_l)\,\text{logit}^{-1}(\beta_0 + \beta_1 x_\text{stim})$

- Figure (below the equation, large): Presentation/example_psych_func.png

## 5. Parameters
  $\psi(x_\text{stim}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}(\beta_0 + \beta_1 x_\text{stim}), \qquad \vec{\theta} = (\beta_0, \beta_1, \gamma_h, \gamma_l)$

- $\gamma_h$ (in blue): lapse rate for "high"
  - lower bound on $\mathbf{P}(\text{guess} = \text{"high"})$, even to the weakest stimuli
  - lower (left) asymptote of $\psi$
- $\gamma_l$ (in purple): lapse rate for "low"
  - lower bound on $\mathbf{P}(\text{guess} = \text{"low"})$, even to the strongest stimuli
  - $1-\gamma_l$ = upper (right) asymptote of $\psi$
- $\text{logit}^{-1}$: the logistic sigmoid, $\text{logit}^{-1}(z) = \frac{1}{1+\exp(-z)}$
- PSE: stimulus where responses of "high" and "low" are equally likely
  - $x_\text{stim}$ such that $\psi(x_\text{stim}) = 0.5$
  - represents bias
- JND: how much the stimulus needs to change to be noticed half the time
  - half the distance between $\psi(x_\text{stim}) = 0.25$ and $0.75$
  - represents sensitivity

## 6. The task: one trial
- TikZ diagram (macros \trialhand, \trialpanel in the Presentation.tex preamble), two panels side by side:
  - left, "unimanual trial (no distractor)": stim hand with a blue zigzag probe $x_\text{stim}$;
    dist hand with a straight orange probe, $x_\text{d} = 0$
  - right, "bimanual trial (with distractor)": same stim probe; dist hand with a tight
    orange coil, $x_\text{d}$

## 7. Modeling bimanual interactions
Same bullets as slide 3, highlighting swapped: distractor bullets in [blue] ... [end blue], the rest in gray.
- [gray] stimulus $x_\text{stim} \in \{6, 12, 18, 24, 32, 38, 44, 50\}$ μm
  - administered to one finger of one hand (left or right) [end gray]
- [blue] distractor $x_\text{dist} \in \{0, x_\text{d}\}$ μm
  - $x_\text{d} \in \{6, 12, 18, 24, 32\}$ μm, value fixed for each session
  - administered to the other (non-stimulus) hand
- if present, distractor administered simultaneously with the stimulus [end blue]
- [gray] monkey categorizes the stimulus as either "high" or "low" relative to a trained boundary of 28 μm
  - "low" response: correct for $x_\text{stim} < 28$ μm
  - "high" response: correct for $x_\text{stim} > 28$ μm
- correct responses are rewarded with water or juice [end gray]

## 8. The data
- One monkey (Sirius), 12,937 trials.
- Collected over 42 sessions, averaging approximately 308 trials per session.
- 8 stimulus amplitudes, 6–50 μm; 4 groups: hand × (unimanual, bimanual).
- Each session uses a single distractor amplitude (6–32 μm) on its bimanual trials.

Below the bullets: table of trial counts by stimulus amplitude, generated from Sirius_Data.pkl
by Presentation/Make_Data_Table.py into Presentation/data_table.tex (rerun the script, don't edit by hand):

| $x_\text{stim}$ (μm) | left unimanual | left bimanual | right unimanual | right bimanual | total |
|---|---|---|---|---|---|
| 6 | 408 | 404 | 404 | 402 | 1,618 |
| 12 | 407 | 400 | 408 | 400 | 1,615 |
| 18 | 402 | 400 | 403 | 405 | 1,610 |
| 24 | 402 | 405 | 402 | 406 | 1,615 |
| 32 | 404 | 405 | 405 | 407 | 1,621 |
| 38 | 403 | 408 | 404 | 408 | 1,623 |
| 44 | 406 | 403 | 404 | 405 | 1,618 |
| 50 | 405 | 405 | 406 | 401 | 1,617 |
| total | 3,237 | 3,230 | 3,236 | 3,234 | 12,937 |

## 9. Three models
Table:

| Model | Sessions | Distractor |
|---|---|---|
| A | pooled: one curve per group | uni and bi fit separately |
| B | hierarchical: one curve per group and session | uni and bi fit separately |
| C | hierarchical, one group per hand | covariate: shift $\beta_2 x_\text{dist}$ |

Below the table: For each model: convergence study on synthetic data, then the fit to the real data.

---

# Section: Model A: Sessions Pooled

### Subsection: Model Structure

## 10. Model A: graphical model
- Left: TikZ plate diagram: $\beta_0, \beta_1, \gamma_h, \gamma_l$ → $\psi_i$ ← $x_i$; $\psi_i$ → $r_i$ (observed);
  trial plate ($i = 1,\dots,n$) inside an outer plate labelled "each of the 4 groups"
- Right (legend):
  - circles: random variables
  - shaded: observed
  - rectangles: deterministic
  - $x_i$: fixed covariate
  - outer plate: the same model, fit independently to each group

## 11. Model A: likelihood
For each group (left/right × unimanual/bimanual), fit independently to its $n \approx 3,230$ trials from all 42 sessions pooled:
Likelihood: probability of obtaining responses $\vec{r}$ given parameters $\vec{\theta} = (\beta_0, \beta_1, \gamma_h, \gamma_l)$
$r_i \sim \mathbf{Bernoulli}[\psi(x_i;\vec{\theta})]$ ($r_i = 1$ for "high", $x_i$ standardized), so:

  $\mathbf{P}[r_i = z] = \psi(x_i;\vec{\theta})^{z}\,(1-\psi(x_i;\vec{\theta}))^{1-z}, \qquad z \in \{0, 1\}$

  $L(\vec{\theta};\vec{r}) = \prod_{i=1}^{n} \mathbf{P}[r_i \mid \vec{\theta}]$
  $\qquad = \prod_{i=1}^{n} \psi(x_i;\vec{\theta})^{r_i}\,(1-\psi(x_i;\vec{\theta}))^{1-r_i}$

## 12. Model A: priors
The same priors in every group:

  $\beta_0 \sim \mathcal{N}(0, 6)$
  $\beta_1 \sim \mathcal{N}(4, 2)$
  $\gamma_h = 0.25\, w_h, \quad w_h \sim \text{Beta}(1.22, 4.89)$
  $\gamma_l = 0.25\, w_l, \quad w_l \sim \text{Beta}(1.22, 4.89)$

- $\beta_1$ must be positive for the psychometric function to have a positive slope
- $\gamma$'s have support between 0 and 0.25, with prior mean of 0.05 and 95% range [0.003, 0.141]

### Subsection: Synthetic Data
(numbers from Model_A/Convergence_A_Report)

## 13. Model A on synthetic data: the study
- One fixed set of true parameters $(\beta_0, \beta_1, \gamma_h, \gamma_l)$ per group
- Trials drawn from the real design (the real mix of stimulus levels and groups), responses simulated from the true curves
- 7 sizes: 250 to 16,000 trials; the real data (12,937) lie between the two largest
- 20 replicates of the whole sweep: 140 fits, 80 per size and parameter
- All fits sampled well: $\hat{R} \le 1.006$, bulk ESS $\ge 1,554$

## 14. Model A on synthetic data: the 95% HDIs are calibrated
- Figure: Model_A/Plots_A/Convergence_Plots_A/coverage_vs_size_all_params.png (no caption)

## 15. Model A on synthetic data: $\gamma_l$ is pulled upward
- Figure: Model_A/Plots_A/Convergence_Plots_A/recovery_gam_l_mean_hdi_vs_size.png
- Caption: True $\gamma_l$ (0.005, 0.01) sits in the prior's lower tail, and the data hold only 3–7 such lapses: the estimates are 1 sd too high at small sizes, 0.36 sd at 16,000.

## 16. Model A on synthetic data: what is recoverable
At the size of the real data (about 3,230 trials per group):
- **Recoverable:** PSE, JND, $\beta_0$, $\gamma_h$: unbiased, calibrated HDIs
  - posterior sd about 0.3 μm for the PSE, 0.2 μm for the JND
- **Recoverable, HDIs a little wide:** $\beta_1$ (coverage 0.98–0.99)
- **Not reliably recoverable:** small $\gamma_l$
  - biased upward by 0.4–0.5 sd; by about 1 sd when the curve never reaches its ceiling (left_uni), where $\gamma_l$ trades off with the slope
  - HDIs still cover the truth, but the posterior mean should not be read literally
- **Caveats:** one set of true values; data simulated with one curve per group, so this says nothing about variation between sessions

### Subsection: Real Data
(numbers from Model_A/Fit_A_Report)

## 17. Model A on the real data: group estimates
Table, posterior mean [95% HDI]:

| | PSE (μm) | JND (μm) | $\gamma_h$ | $\gamma_l$ |
|---|---|---|---|---|
| left unimanual | 32.58 [32.10, 33.04] | 3.56 [3.25, 3.89] | 0.002 [0.000, 0.006] | 0.009 [0.000, 0.018] |
| left bimanual | 28.38 [27.79, 28.98] | 4.39 [3.93, 4.83] | 0.041 [0.025, 0.056] | 0.016 [0.005, 0.027] |
| right unimanual | 31.57 [31.05, 32.06] | 4.09 [3.77, 4.41] | 0.003 [0.000, 0.008] | 0.006 [0.000, 0.013] |
| right bimanual | 25.50 [24.89, 26.08] | 4.36 [3.93, 4.81] | 0.045 [0.025, 0.064] | 0.012 [0.004, 0.020] |

Small note under the table: Posterior mean [95% HDI]. 12,937 trials, 4 chains × 5,000 draws: no divergences, $\hat{R} = 1.000$, bulk ESS $\ge 15,213$.

- Without the distractor the PSE lies 3.6–4.6 μm above the trained 28 μm
- With it the PSE moves to the threshold (left) or below it (right)
- $\gamma_l$ estimates lean high (see the synthetic study)

## 18. Model A on the real data: fitted curves
- Figures side by side: Model_A/Plots_A/Fit_Plots_A/curves_left.png, curves_right.png
- Caption: Unimanual in blue, bimanual in red, with the observed proportions pooled over sessions; dashed: the trained threshold of 28 μm.

## 19. Model A on the real data: effect of the distractor on PSE
- Figures stacked, no caption, both cropped in LaTeX from Model_A/Plots_A/Fit_Plots_A/:
  - top ("uni vs bi"): the PSE row of uni_bi_posteriors.png, unimanual and bimanual posteriors overlaid, left and right hand
  - bottom ("bi − uni"): the PSE row of effect_densities.png, the posterior of the difference, with P(>0)
- "Left Hand" / "Right Hand" above the columns; legend line below: Unimanual (blue), Bimanual (red), 95% HDI shaded, mean dotted; green dashed: trained threshold

## 20. Model A on the real data: effect of the distractor on JND
- Figures stacked, no caption, both cropped in LaTeX from Model_A/Plots_A/Fit_Plots_A/:
  - top ("uni vs bi"): the JND row of uni_bi_posteriors.png, unimanual and bimanual posteriors overlaid, left and right hand
  - bottom ("bi − uni"): the JND row of effect_densities.png, the posterior of the difference, with P(>0)
- "Left Hand" / "Right Hand" above the columns; legend line below: Unimanual (blue), Bimanual (red), 95% HDI shaded, mean dotted

## 21. Model A on the real data: effect of the distractor on $\gamma_h$
- Figures stacked, no caption, both cropped in LaTeX from Model_A/Plots_A/Fit_Plots_A/:
  - top ("uni vs bi"): the $\gamma_h$ row of uni_bi_posteriors.png, unimanual and bimanual posteriors overlaid, left and right hand
  - bottom ("bi − uni"): the $\gamma_h$ row of effect_densities.png, the posterior of the difference, with P(>0)
- "Left Hand" / "Right Hand" above the columns; legend line below: Unimanual (blue), Bimanual (red), 95% HDI shaded, mean dotted

## 22. Model A on the real data: effect of the distractor on $\gamma_l$
- Figures stacked, no caption, both cropped in LaTeX from Model_A/Plots_A/Fit_Plots_A/:
  - top ("uni vs bi"): the $\gamma_l$ row of uni_bi_posteriors.png, unimanual and bimanual posteriors overlaid, left and right hand
  - bottom ("bi − uni"): the $\gamma_l$ row of effect_densities.png, the posterior of the difference, with P(>0)
- "Left Hand" / "Right Hand" above the columns; legend line below: Unimanual (blue), Bimanual (red), 95% HDI shaded, mean dotted

## 23. Model A on the real data: the pooled data are reproduced
- Figure: Model_A/Plots_A/Fit_Plots_A/ppc_pooled_by_stimulus.png
- Caption: Pooled over sessions, all 32 group × amplitude proportions lie inside their 95% posterior predictive intervals.

## 24. Model A: conclusions
- Reliable fit; the pooled data are reproduced at every amplitude (elpd_loo = −3344.0, SE 64.5)
- Effect of the distractor (bimanual − unimanual), posterior mean [95% HDI]:
  - **PSE:** lowered, by −4.19 [−4.96, −3.44] μm (left) and −6.07 [−6.85, −5.30] μm (right)
  - **JND:** raised for the left hand, +0.83 [0.27, 1.37] μm; unclear for the right, +0.27 [−0.29, 0.80] μm
  - **$\gamma_h$:** raised, by +0.038 [0.023, 0.055] (left) and +0.042 [0.021, 0.061] (right)
  - **$\gamma_l$:** no clear effect, +0.007 [−0.008, 0.022] (left) and +0.006 [−0.006, 0.017] (right)
- ⇒ A curve for every session: Model B

---

# Section: Model B: Sessions Hierarchical

### Subsection: Model Structure

## 25. Model B: graphical model
- TikZ plate diagram: hyperparameters $\mu_{\beta_0}, \sigma_{\beta_0}, \mu_{\beta_1}, \sigma_{\beta_1}, \mu_{\gamma_h}, \sigma_{\gamma_h}, \mu_{\gamma_l}, \sigma_{\gamma_l}$ → session parameters $\beta_{0,s}, \beta_{1,s}, \gamma_{h,s}, \gamma_{l,s}$ → $\psi_i$ ← $x_i$; $\psi_i$ → $r_i$ (observed)
  - plates: trials $i$ in session $s$, inside sessions $s = 1,\dots,42$, inside "each of the 4 groups"
- Small note below: Circles: random variables (shaded: observed); rectangle: deterministic; $x_i$: fixed covariate. The same model is fit independently to each group.

## 26. Model B: likelihood
For each group (left/right × unimanual/bimanual), fit independently to its $n \approx 3,230$ trials, now with a parameter set for every session $s$:
Likelihood: probability of obtaining responses $\vec{r}$ given parameters $\vec{\theta} = (\vec{\theta}_s)_{s=1}^{42}$, $\vec{\theta}_s = (\beta_{0,s}, \beta_{1,s}, \gamma_{h,s}, \gamma_{l,s})$
$r_i \sim \mathbf{Bernoulli}[\psi(x_i;\vec{\theta}_{s[i]})]$ ($s[i]$: the session of trial $i$), so:

  $\mathbf{P}[r_i = z] = \psi(x_i;\vec{\theta}_{s[i]})^{z}\,(1-\psi(x_i;\vec{\theta}_{s[i]}))^{1-z}, \qquad z \in \{0, 1\}$

  $L(\vec{\theta};\vec{r}) = \prod_{i=1}^{n} \mathbf{P}[r_i \mid \vec{\theta}_{s[i]}]$
  $\qquad = \prod_{s=1}^{42} \prod_{i:\, s[i]=s} \psi(x_i;\vec{\theta}_s)^{r_i}\,(1-\psi(x_i;\vec{\theta}_s))^{1-r_i}$

## 27. Model B: priors
Session parameters drawn from the group's population (non-centered in the code):

  $\beta_{0,s} \sim \mathcal{N}(\mu_{\beta_0}, \sigma_{\beta_0})$, $\quad \gamma_{h,s} = 0.25\, w_{h,s},\ w_{h,s} \sim \text{LogitNormal}(\mu_{\gamma_h}, \sigma_{\gamma_h})$
  $\beta_{1,s} \sim \mathcal{N}(\mu_{\beta_1}, \sigma_{\beta_1})$, $\quad \gamma_{l,s} = 0.25\, w_{l,s},\ w_{l,s} \sim \text{LogitNormal}(\mu_{\gamma_l}, \sigma_{\gamma_l})$

Hyperpriors, the same in every group:

  $\mu_{\beta_0} \sim \mathcal{N}(0, 2)$, $\quad \sigma_{\beta_0} \sim \text{HalfNormal}(2)$
  $\mu_{\beta_1} \sim \text{LogNormal}(\log 4, 0.5)$, $\quad \sigma_{\beta_1} \sim \text{Exponential}(1)$
  $\mu_{\gamma_h}, \mu_{\gamma_l} \sim \mathcal{N}(-3, 1.5)$, $\quad \sigma_{\gamma_h}, \sigma_{\gamma_l} \sim \text{HalfNormal}(0.5)$

- Chosen from the design, checked by prior predictive simulation, not fit to the data
- $\mu_{\beta_0}$: typical PSE within 13–43 μm with 90% probability
- $\mu_{\beta_1} > 0$: median 4, 95% in $[1.5, 10.6]$; $\sigma_{\beta_1}$ keeps 98% of session slopes positive
- Typical lapse rate: median 0.012, 95% in $[0.0007, 0.12]$

### Subsection: Synthetic Data
(numbers from Model_B/Convergence_B_Report)

## 28. Model B on synthetic data: the study
- Fixed true hyperparameters for 4 synthetic groups, chosen to cover high and low values of each parameter (next slide); session parameters drawn from them
- Two sweeps, each run twice with new data:
  - sessions: 5, 10, 20, 40, 80 sessions of 308 trials (tests the hyperparameters)
  - trials: 42 sessions of 20, 40, 80, 160, 320 trials (tests the session parameters)
- The real data (42 sessions of about 308 trials) are closest to the 40-session and 320-trial fits
- All 20 fits sampled well: no divergences, $\hat{R} \le 1.010$, bulk ESS $\ge 579$

## 29. Model B on synthetic data: the true values
One set of fixed values per group, chosen to cover high and low values of each parameter (not taken from the real data):

| group | slope $\mu_{\beta_1}$ | $\mu_{\beta_0}$ | PSE (μm) | typical $\gamma_h$ | typical $\gamma_l$ | $\sigma_{\beta_0}$ |
|---|---|---|---|---|---|---|
| left unimanual | 3 | −1 | 32.6 | 0.033 | 0.0055 | 1.00 |
| left bimanual | 3 | +1 | 22.8 | 0.033 | 0.0050 | 0.50 |
| right unimanual | 6 | −1 | 30.2 | 0.058 | 0.0143 | 0.25 |
| right bimanual | 6 | +1 | 25.3 | 0.058 | 0.0108 | 0.75 |

- The other between-session sds are near 0: $\sigma_{\beta_1}$ 0.002–0.008, $\sigma_{\gamma_h}$ 0.024–0.147, $\sigma_{\gamma_l}$ 0.014–0.103 (logit scale)
- The left-hand curves are shallow: left unimanual never reaches its ceiling at 50 μm, left bimanual never reaches its floor at 6 μm

## 30. Model B on synthetic data: $\mu_{\beta_0}$ vs number of sessions
- Figure (no caption): $\mu_{\beta_0}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 31. Model B on synthetic data: $\mu_{\beta_1}$ vs number of sessions
- Figure (no caption): $\mu_{\beta_1}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 32. Model B on synthetic data: $\mu_{\gamma_h}$ vs number of sessions
- Figure (no caption): $\mu_{\gamma_h}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 33. Model B on synthetic data: $\mu_{\gamma_l}$ vs number of sessions
- Figure (no caption): $\mu_{\gamma_l}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 34. Model B on synthetic data: $\sigma_{\beta_0}$ vs number of sessions
- Figure (no caption): $\sigma_{\beta_0}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_sds_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 35. Model B on synthetic data: $\sigma_{\beta_1}$ vs number of sessions
- Figure (no caption): $\sigma_{\beta_1}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_sds_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 36. Model B on synthetic data: $\sigma_{\gamma_h}$ vs number of sessions
- Figure (no caption): $\sigma_{\gamma_h}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_sds_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 37. Model B on synthetic data: $\sigma_{\gamma_l}$ vs number of sessions
- Figure (no caption): $\sigma_{\gamma_l}$'s row of Model_B/Plots_B/Convergence_Plots_B/hyper_sds_recovery_vs_sessions.png, cut in half and stacked 2 × 2 in LaTeX: left hand on top, right hand below; unimanual left, bimanual right; x tick labels under the lower half

## 38. Model B on synthetic data: session-level recovery
At 320 trials per session (42 sessions, both replicates): RMSE of the posterior mean / mean 95% HDI width / coverage

| | left unimanual | left bimanual | right unimanual | right bimanual |
|---|---|---|---|---|
| PSE | 0.14 / 0.49 / 0.90 | 0.12 / 0.42 / 0.90 | 0.05 / 0.16 / 0.93 | 0.09 / 0.32 / 0.94 |
| JND | 0.014 / 0.15 / 1.00 | 0.034 / 0.17 / 1.00 | 0.013 / 0.08 / 1.00 | 0.010 / 0.10 / 1.00 |
| $\beta_0$ | 0.41 / 1.67 / 0.93 | 0.34 / 1.20 / 0.92 | 0.26 / 1.01 / 0.95 | 0.55 / 1.99 / 0.94 |
| $\beta_1$ | 0.19 / 1.39 / 1.00 | 0.23 / 1.18 / 1.00 | 0.29 / 2.11 / 1.00 | 0.35 / 2.97 / 1.00 |
| $\gamma_h$ | 0.003 / 0.047 / 1.00 | 0.012 / 0.052 / 1.00 | 0.009 / 0.071 / 1.00 | 0.013 / 0.062 / 1.00 |
| $\gamma_l$ | 0.016 / 0.048 / 1.00 | 0.003 / 0.019 / 1.00 | 0.004 / 0.021 / 1.00 | 0.003 / 0.023 / 1.00 |

Small note: PSE and JND in standardized units (1 unit = 14.6 μm).

## 39. Model B on synthetic data: what is recoverable
At the size of the real data (42 sessions of about 308 trials):
- **Recoverable:** per-session PSE and JND; the bias mean $\mu_{\beta_0}$ (±1); the slope mean $\mu_{\beta_1}$ when the slope is shallow (3); the bias spread $\sigma_{\beta_0}$ (0.25–1.0); $\mu_{\gamma_h}$ in most groups
- **Weakly constrained:** $\mu_{\beta_1}$ when the slope is steep (6): the curve rises within one stimulus spacing, so only 2 levels inform it (HDIs 1.8–2.6 wide)
- **Biased in some datasets:** a lapse rate whose side of a shallow curve never reaches its asymptote: $\gamma_l$ too high when the PSE is high, $\gamma_h$ too low when it is low; the higher $\mu_{\gamma_l}$ (−2.8) is also overestimated at 80 sessions
- **Not identified:** the sds with true values near 0 ($\sigma_{\beta_1}$, $\sigma_{\gamma_h}$, $\sigma_{\gamma_l}$); their posteriors reflect the prior
- Session-level HDIs mostly conservative; PSE coverage slightly low (0.90–0.94)

### Subsection: Real Data
(numbers from Model_B/Fit_B_Report)

## 40. Model B on the real data: a typical session
Table, quantities at the group-level means, posterior mean [95% HDI]:

| | PSE (μm) | JND (μm) | $\gamma_h$ | $\gamma_l$ |
|---|---|---|---|---|
| left unimanual | 32.4 [31.6, 33.3] | 3.29 [2.95, 3.63] | 0.002 [0.000, 0.005] | 0.009 [0.000, 0.019] |
| left bimanual | 28.2 [27.2, 29.2] | 3.89 [3.34, 4.46] | 0.023 [0.009, 0.037] | 0.009 [0.001, 0.017] |
| right unimanual | 31.4 [30.3, 32.5] | 3.56 [3.23, 3.89] | 0.003 [0.000, 0.006] | 0.007 [0.000, 0.015] |
| right bimanual | 25.5 [24.5, 26.5] | 3.95 [3.48, 4.44] | 0.036 [0.017, 0.056] | 0.010 [0.003, 0.018] |

Small note: 4 chains × 5,000 draws: no divergences; $\hat{R} = 1.00$ and bulk ESS $\ge 2,210$ for all but 4 raw variables, whose only disagreement is a sign that does not enter the model.

- The sessions differ in bias ($\sigma_{\beta_0} = 0.76$–1.01); left bimanual also in slope ($\sigma_{\beta_1} = 1.27$)
- The between-session lapse sds sit near their prior (not identified)

## 41. Model B on the real data: fitted curves in one session (04-13)
- Figures side by side (no caption): Model_B/Plots_B/Fit_Plots_B/curves_left_04-13.png, curves_right_04-13.png

## 42. Model B on the real data: PSE, left hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the left-hand row of PSE_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the left-hand row of effect_PSE_by_session.png (bimanual − unimanual in each session), with the session labels

## 43. Model B on the real data: PSE, right hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the right-hand row of PSE_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the right-hand row of effect_PSE_by_session.png (bimanual − unimanual in each session), with the session labels
- Key line below (the right-hand rows have no legend in the PNGs): unimanual (blue), bimanual (red): session mean and 95% HDI; bands: typical session; green dashed: trained threshold. bi − uni: filled if the session's 95% HDI excludes 0; orange band: average over sessions

## 44. Model B on the real data: JND, left hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the left-hand row of JND_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the left-hand row of effect_JND_by_session.png (bimanual − unimanual in each session), with the session labels

## 45. Model B on the real data: JND, right hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the right-hand row of JND_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the right-hand row of effect_JND_by_session.png (bimanual − unimanual in each session), with the session labels
- Key line below (the right-hand rows have no legend in the PNGs): unimanual (blue), bimanual (red): session mean and 95% HDI; bands: typical session. bi − uni: filled if the session's 95% HDI excludes 0; orange band: average over sessions

## 46. Model B on the real data: $\gamma_h$, left hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the left-hand row of gamh_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the left-hand row of effect_gamh_by_session.png (bimanual − unimanual in each session), with the session labels

## 47. Model B on the real data: $\gamma_h$, right hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the right-hand row of gamh_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the right-hand row of effect_gamh_by_session.png (bimanual − unimanual in each session), with the session labels
- Key line below (the right-hand rows have no legend in the PNGs): unimanual (blue), bimanual (red): session mean and 95% HDI; bands: typical session. bi − uni: filled if the session's 95% HDI excludes 0; orange band: average over sessions

## 48. Model B on the real data: $\gamma_l$, left hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the left-hand row of gaml_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the left-hand row of effect_gaml_by_session.png (bimanual − unimanual in each session), with the session labels

## 49. Model B on the real data: $\gamma_l$, right hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the right-hand row of gaml_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the right-hand row of effect_gaml_by_session.png (bimanual − unimanual in each session), with the session labels
- Key line below (the right-hand rows have no legend in the PNGs): unimanual (blue), bimanual (red): session mean and 95% HDI; bands: typical session. bi − uni: filled if the session's 95% HDI excludes 0; orange band: average over sessions

## 50. Model B: PSE shift vs distractor amplitude (exploratory)
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_vs_distractor_amplitude.png

## 51. Model B on the real data: the pooled data are reproduced
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/ppc_pooled_by_stimulus.png

## 52. Model B: conclusions
- Reliable fit; no systematic misfit, pooled or within sessions (elpd_loo = −3180.0, SE 64.3)
- Effect of the distractor (bimanual − unimanual), average over the 42 sessions, posterior mean [95% HDI]:
  - **PSE:** lowered, by −4.29 [−5.05, −3.52] μm (left) and −6.06 [−6.80, −5.34] μm (right); negative in 40/42 and 42/42 sessions
  - **JND:** raised for the left hand, +1.03 [0.46, 1.59] μm, driven by a few sessions; unclear for the right, +0.49 [−0.07, 1.03] μm
  - **$\gamma_h$:** raised, by +0.025 [0.009, 0.042] (left) and +0.035 [0.016, 0.054] (right)
  - **$\gamma_l$:** no effect, 0.000 [−0.014, 0.013] (left) and +0.003 [−0.009, 0.015] (right)
- The PSE shift grows with distractor amplitude, by about −2 μm per 10 μm (exploratory; amplitude is confounded with session order)
- ⇒ Put the distractor amplitude into the model: Model C

---

# Section: Model C: Distractor Amplitude as a Covariate

### Subsection: Model Structure

## 53. Model C: including the distractor as a covariate
Psychometric function: $\psi(x_\text{stim}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}($[red] $\beta_0 + \beta_1 x_\text{stim}$ [end red]$)$
Suppose the distractor has an impact on the perceived stimulus as follows: $x_\text{perc} = x_\text{stim} + c\, x_\text{dist}$
Then (all [red], except [amber] terms):
  $\beta_0 + \beta_1 x_\text{perc} = \beta_0 + \beta_1 (x_\text{stim} + c\, x_\text{dist})$
  $\qquad = \beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_1 c$ [end amber] $x_\text{dist}$
  $\qquad = \beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_2$ [end amber] $x_\text{dist}$
Considering $x_\text{dist}$ as a covariate: $\psi(x_\text{stim}, x_\text{dist}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}($[red] $\beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_2$ [end amber] $x_\text{dist}$ [end red]$)$

## 54. Model C: graphical model
- TikZ plate diagram: hyperparameters $\mu, \sigma$ of $\beta_0, \beta_1, \beta_2, \gamma_h, \gamma_l$ → session parameters $\beta_{0,s}, \beta_{1,s}, \beta_{2,s}, \gamma_{h,s}, \gamma_{l,s}$ → $\psi_i$ ← $x_{\text{stim},i}$, $x_{\text{dist},i}$; $\psi_i$ → $r_i$ (observed)
  - plates: trials $i$ in session $s$, inside sessions $s = 1,\dots,42$, inside "each hand"
- Small note below: Circles: random variables (shaded: observed); rectangle: deterministic; $x_{\text{stim},i}$, $x_{\text{dist},i}$: fixed covariates. The same model is fit independently to each hand, with unimanual and bimanual trials together.

## 55. Model C: likelihood
For each hand, fit independently to its $n \approx 6,470$ unimanual and bimanual trials, with the distractor as a covariate $x_{\text{dist},i}$ (amplitude / 14.63; $x_{\text{dist},i} = 0$ on unimanual trials):
Likelihood: probability of obtaining responses $\vec{r}$ given parameters $\vec{\theta} = (\vec{\theta}_s)_{s=1}^{42}$, $\vec{\theta}_s = (\beta_{0,s}, \beta_{1,s}, \beta_{2,s}, \gamma_{h,s}, \gamma_{l,s})$
$r_i \sim \mathbf{Bernoulli}[\psi(x_{\text{stim},i}, x_{\text{dist},i};\vec{\theta}_{s[i]})]$, with $\psi(x_\text{stim}, x_\text{dist};\vec{\theta}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}(\beta_0 + \beta_1 x_\text{stim} + \beta_2 x_\text{dist})$, so:

  $\mathbf{P}[r_i = z] = \psi(x_{\text{stim},i}, x_{\text{dist},i};\vec{\theta}_{s[i]})^{z}\,(1-\psi(x_{\text{stim},i}, x_{\text{dist},i};\vec{\theta}_{s[i]}))^{1-z}, \qquad z \in \{0, 1\}$

  $L(\vec{\theta};\vec{r}) = \prod_{i=1}^{n} \mathbf{P}[r_i \mid \vec{\theta}_{s[i]}]$
  $\qquad = \prod_{s=1}^{42} \prod_{i:\, s[i]=s} \psi(x_{\text{stim},i}, x_{\text{dist},i};\vec{\theta}_s)^{r_i}\,(1-\psi(x_{\text{stim},i}, x_{\text{dist},i};\vec{\theta}_s))^{1-r_i}$

## 56. Model C: priors
Session parameters drawn from the hand's population (non-centered in the code):

  $\beta_{k,s} \sim \mathcal{N}(\mu_{\beta_k}, \sigma_{\beta_k}),\ k = 0, 1, 2$
  $\gamma_{h,s} = 0.25\, w_{h,s},\ w_{h,s} \sim \text{LogitNormal}(\mu_{\gamma_h}, \sigma_{\gamma_h})$; same for $\gamma_{l,s}$

Hyperpriors, the same for both hands; all but those of $\beta_2$ as in Model B:

  $\mu_{\beta_0} \sim \mathcal{N}(0, 2)$, $\quad \sigma_{\beta_0} \sim \text{HalfNormal}(2)$
  $\mu_{\beta_1} \sim \text{LogNormal}(\log 4, 0.5)$, $\quad \sigma_{\beta_1} \sim \text{Exponential}(1)$
  $\mu_{\beta_2} \sim \mathcal{N}(0, 1)$, $\quad \sigma_{\beta_2} \sim \text{HalfNormal}(1)$
  $\mu_{\gamma_h}, \mu_{\gamma_l} \sim \mathcal{N}(-3, 1.5)$, $\quad \sigma_{\gamma_h}, \sigma_{\gamma_l} \sim \text{HalfNormal}(0.5)$

- $\mu_{\beta_2}$ centered at 0: the distractor could push responses either way
- Uni- and bimanual trials share slope and lapse rates: $\beta_2 x_\text{dist}$ only shifts the curve

## 57. Model C: prior predictive check of the distractor effect
- Figure (title cropped): Model_C/Plots_C/Convergence_Plots_C/Prior_Check_C_distractor.png
- Caption: **Model C drawback:** the PSE shift $-\beta_2 x_\text{dist}/\beta_1$ is tied to the distractor amplitude, so weak distractors can only shift the PSE slightly

### Subsection: Synthetic Data
(numbers from Model_C/Convergence_C_Report)

## 58. Model C on synthetic data: the study
- True hyperparameters derived from Model B's fit to the real data, so the synthetic data resemble the real data (e.g. $\mu_{\beta_2} = 0.93$ left, 1.13 right)
- Each simulated session gets one distractor amplitude, drawn from the real sessions'
- The same two sweeps as for Model B, each run twice with new data:
  - sessions: 5, 10, 20, 40, 80 sessions of 308 trials
  - trials: 42 sessions of 20, 40, 80, 160, 320 trials
- 20 fits: 2 divergences in total (in the two smallest datasets), $\hat{R} \le 1.008$, bulk ESS $\ge 532$

## 59. Model C on synthetic data: hand-level means
At 40 sessions: true value and posterior mean [95% HDI], averaged over the two replicates

| | left true | left posterior | right true | right posterior |
|---|---|---|---|---|
| $\mu_{\beta_0}$ | −1.49 | −1.56 [−1.90, −1.24] | −1.06 | −0.94 [−1.29, −0.57] |
| $\mu_{\beta_1}$ | 4.64 | 4.65 [4.16, 5.18] | 4.46 | 4.46 [4.06, 4.86] |
| $\mu_{\beta_2}$ | 0.93 | 0.93 [0.71, 1.18] | 1.13 | 1.11 [0.74, 1.47] |
| $\mu_{\gamma_h}$ | −2.80 | −2.89 [−3.45, −2.41] | −2.41 | −2.60 [−3.10, −2.11] |
| $\mu_{\gamma_l}$ | −3.20 | −3.39 [−4.34, −2.60] | −3.20 | −3.04 [−3.73, −2.37] |

- All five recovered without visible bias; $\mu_{\beta_2}$'s HDI excludes 0
- Lapse means (logit scale) are the least precise

## 60. Model C on synthetic data: between-session sds
At 80 sessions: true value and posterior mean [95% HDI], averaged over the two replicates

| | left true | left posterior | right true | right posterior |
|---|---|---|---|---|
| $\sigma_{\beta_0}$ | 0.76 | 0.74 [0.58, 0.92] | 1.01 | 0.94 [0.76, 1.14] |
| $\sigma_{\beta_1}$ | 0.78 | 0.83 [0.57, 1.09] | 0.39 | 0.17 [0.00, 0.40] |
| $\sigma_{\beta_2}$ | 0.62 | 0.51 [0.36, 0.67] | 0.83 | 0.86 [0.68, 1.06] |
| $\sigma_{\gamma_h}$ | 0.48 | 0.38 [0.00, 0.83] | 0.37 | 0.31 [0.00, 0.68] |
| $\sigma_{\gamma_l}$ | 0.38 | 0.35 [0.00, 0.85] | 0.41 | 0.29 [0.00, 0.69] |

- The sds of $\beta_0$, $\beta_1$, $\beta_2$ are identified (a small $\sigma_{\beta_1}$ is hard to tell from 0)
- The lapse sds are not: their HDIs start at 0 and barely narrow with more data

## 61. Model C on synthetic data: session $\beta_2$ is weak for small distractors
- Figure (no caption, title cropped): Model_C/Plots_C/Convergence_Plots_C/session_b2_true_vs_recovered.png

## 62. Model C on synthetic data: what is recoverable
At the size of the real data (42 sessions of about 308 trials), if the data follow Model C:
- **Recoverable:** per-session PSEs in both conditions and JNDs; all five hand-level means, including the distractor effect $\mu_{\beta_2}$; the between-session sds of $\beta_0$, $\beta_1$ and $\beta_2$
- **Weakly determined:** the individual session coefficients, especially $\beta_2$ in sessions with a small distractor (HDI 2.6 wide at 6 μm, 1.2 at 32 μm), which are pulled toward the hand mean
- **Not identified:** $\sigma_{\gamma_h}$ and $\sigma_{\gamma_l}$; their posteriors reflect the prior
- **Not tested:** what happens when the data break Model C's assumptions (shared lapse rates and slope, a shift proportional to amplitude)

### Subsection: Real Data
(numbers from Model_C/Fit_C_Report)

## 63. Model C on the real data: hand-level estimates
Table, posterior mean [95% HDI]:

| | left | right |
|---|---|---|
| distractor coefficient $\mu_{\beta_2}$ | 0.90 [0.71, 1.09] | 1.18 [0.99, 1.37] |
| PSE, unimanual (μm) | 32.34 [31.51, 33.17] | 31.15 [29.92, 32.40] |
| JND (μm) | 3.75 [3.39, 4.11] | 3.88 [3.59, 4.19] |
| $\gamma_h$ | 0.0091 [0.0033, 0.0147] | 0.0113 [0.0042, 0.0189] |
| $\gamma_l$ | 0.0086 [0.0023, 0.0155] | 0.0085 [0.0029, 0.0143] |
| $\sigma_{\beta_0}$ / $\sigma_{\beta_1}$ / $\sigma_{\beta_2}$ | 0.60 / 0.90 / 0.30 | 1.07 / 0.35 / 0.23 |

Small note: PSE, JND and lapse rates of a typical session (JND and lapse rates are the same on bimanual trials). 4 chains × 5,000 draws: 2 divergences, $\hat{R} = 1.000$, bulk ESS $\ge 3,410$.

- $\mu_{\beta_2} > 0$ for both hands: the distractor lowers the PSE, more the stronger it is
- The sessions vary little around the hand's amplitude line ($\sigma_{\beta_2}$ small)

## 64. Model C on the real data: fitted curves in one session (04-13)
- Figures side by side (no caption): Model_C/Plots_C/Fit_Plots_C/curves_left_04-13.png, curves_right_04-13.png

## 65. Model C on the real data: PSE in each session
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/PSE_by_session.png (unimanual and bimanual per session)

## 66. Model C on the real data: effect of the distractor on PSE
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/effect_PSE_by_session.png

## 67. Model C: the PSE shift grows with distractor amplitude
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/pse_shift_vs_amplitude.png

## 68. Model C on the real data: misfit at the weakest stimuli
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/ppc_pooled_by_stimulus.png

## 69. Model C: observed vs predicted, by distractor amplitude
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/ppc_by_distractor_amplitude.png

## 70. Model C: conclusions
- Reliable fit: 2 divergences in 20,000 draws, $\hat{R} = 1.000$ (elpd_loo = −3217.0, SE 65.7)
- Effect of the distractor (bimanual − unimanual), posterior mean [95% HDI]:
  - **PSE:** lowered, by −4.51 [−5.15, −3.86] μm (left) and −5.74 [−6.39, −5.11] μm (right) on average over sessions; more the stronger the distractor: a typical session shifts −1.23 and −1.66 μm at 6 μm, −6.57 and −8.85 μm at 32 μm
  - **JND, $\gamma_h$, $\gamma_l$:** shared by both conditions, so the model cannot show an effect on them
- The misfit sits where its assumptions meet the data:
  - shared lapse rates: 6 of 32 pooled proportions outside their intervals, all at weak stimuli
  - a shift proportional to amplitude: for the right hand, even a 6 μm distractor shifts responses more than Model C allows
- ⇒ Which model describes the data best? Model comparison

---

# Section: Model Comparison and Recovery

### Subsection: Predictive Performance

## 71. Leave-one-out cross-validation
All three models fit to the same 12,937 trials in the same order, so the difference between two models can be computed trial by trial:

| Model | elpd_loo (SE) | p_loo | difference (SE) |
|---|---|---|---|
| A: sessions pooled | −3344.0 (64.5) | 13.4 | B − A: +164.1 (19.2) |
| B: sessions hierarchical | −3180.0 (64.3) | 178.7 | **best** |
| C: distractor as covariate | −3217.0 (65.7) | 138.6 | C − B: −37.1 (11.9) |

- B beats A by 8.5 standard errors, despite 179 effective parameters against 13
- B beats C by 3.1 standard errors, although C needs fewer effective parameters
- Every Pareto $k$ below 0.7, so the PSIS-LOO estimates are reliable

### Subsection: Model B vs Model A
(numbers from Model_B/Fit_B_Report, "Comparison with Model A")

## 72. Model B vs Model A: the distractor effects agree
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_A_vs_B.png

## 73. Model B vs Model A: pooling inflates the JND
Posterior means: Model A (sessions pooled) against the average of Model B's 42 session estimates

| | PSE A | PSE B | JND A | JND B | $\gamma_h$ A | $\gamma_h$ B | $\gamma_l$ A | $\gamma_l$ B |
|---|---|---|---|---|---|---|---|---|
| left unimanual | 32.58 | 32.45 | 3.56 | 3.31 | 0.002 | 0.002 | 0.009 | 0.010 |
| left bimanual | 28.38 | 28.16 | 4.39 | 4.34 | 0.041 | 0.027 | 0.016 | 0.010 |
| right unimanual | 31.57 | 31.44 | 4.10 | 3.57 | 0.003 | 0.003 | 0.006 | 0.008 |
| right bimanual | 25.50 | 25.38 | 4.36 | 4.06 | 0.045 | 0.038 | 0.012 | 0.011 |

- PSEs agree to within 0.22 μm
- Model A's JNDs are 0.05–0.52 μm larger: averaging curves whose PSEs drift from session to session gives a shallower curve than any single session has
- Model A's bimanual $\gamma_h$ is also higher: it absorbs low-PSE sessions as lapses

### Subsection: Model Recovery: B vs A
(numbers from Model_B/Model_Recov_BA_Report)

## 74. Model recovery: is B's advantage real?
Two questions before reading B's win as session-to-session variation:
1. Can LOO identify the model that generated a dataset of this size and design?
2. Could B win just because it is more flexible? B contains A as the case of no between-session variation.

Design:
- 10 replicates, each with two synthetic datasets on the real trial design (12,937 trials in 42 sessions)
  - **data A:** responses from Model A, true values = A's real-data posterior means
  - **data B:** session parameters drawn from B's real-data hyperparameters
- Both models fit to every dataset; LOO picks the model with the higher elpd_loo; $\Delta = \text{elpd}(\text{B}) - \text{elpd}(\text{A})$

## 75. Model recovery: LOO picks the generating model

| | selects A | selects B |
|---|---|---|
| data from A | 9 | 1 |
| data from B | 0 | 10 |

- **Data from B:** B wins every time by a wide margin, $\Delta$ = 150.2 to 282.4 ($z$ = 8.2 to 11.1); B's p_loo = 167–190
- **Data from A:** A wins 9 of 10, but narrowly, $\Delta$ = −6.8 to +1.6; the one B win is a near-tie ($z$ = 0.36). B mimics A by shrinking its sds (p_loo = 54–84)
- **Real data:** $\Delta$ = 164.1 and p_loo = 178.7, inside the data-from-B range and far outside the data-from-A range

## 76. Model recovery: $\Delta$ in every replicate
- Figure (no caption, title cropped): Model_B/Plots_B/Model_Recov_Plots_BA/elpd_diff_by_replicate.png

## 77. Model recovery: conclusions
- With this design and data size, LOO identifies session-to-session variation reliably: B wins every time with $z > 8$ when the variation is there
- B's flexibility alone does not make it win: on data without session variation it loses or ties ($\Delta \le 1.6$)
- The real data look like the data-from-B replicates in both $\Delta$ and p_loo: the psychometric curves do vary between sessions
- The reverse, confirming that variation is absent, is only weakly answered: the more flexible model loses by just a few elpd points when the simpler one is true
- Limits: 10 replicates per model; one set of B hyperparameters (with smaller between-session sds the gap would shrink)

### Subsection: Model C vs Model B
(numbers from Model_C/Fit_C_Report, "Comparison with Model B")

## 78. Model C vs Model B: where C loses
elpd_loo(C) − elpd_loo(B) summed over the trials at each amplitude; negative values favor Model B

| stimulus (μm) | left bimanual | left unimanual | right bimanual | right unimanual |
|---|---|---|---|---|
| 6 | −6.6 | −3.5 | −5.1 | −2.3 |
| 12 | −4.2 | −3.0 | −1.8 | −2.8 |
| 18 | 0.7 | −1.6 | −3.4 | −0.3 |
| 24 | −1.0 | 0.8 | 1.9 | 0.8 |
| 32 | −0.1 | −3.4 | 0.3 | 1.7 |
| 38 | −0.3 | −1.4 | 0.2 | 1.6 |
| 44 | 0.0 | −3.8 | 0.4 | −1.5 |
| 50 | −0.1 | 0.3 | 0.1 | 0.0 |

- 29.2 of the 37.1 are lost at 6 and 12 μm: one lapse rate for both conditions
- Next: left unimanual at 32 and 44 μm: one slope for both conditions

## 79. Model C vs Model B: session PSE shifts
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/pse_shift_C_vs_B.png

## 80. Model C vs Model B: the weakest distractor
Session PSE shifts (μm): mean, and range of the posterior means, over the sessions at the weakest and strongest distractor

| hand | distractor | Model B | Model C |
|---|---|---|---|
| left | 6 μm (8 sessions) | −1.13 (−3.82 to 0.41) | −1.34 (−1.59 to −1.18) |
| left | 32 μm (13 sessions) | −6.67 (−10.12 to −0.44) | −7.40 (−11.80 to −2.91) |
| right | 6 μm (8 sessions) | −3.91 (−6.74 to −2.06) | −1.75 (−1.88 to −1.64) |
| right | 32 μm (13 sessions) | −7.90 (−11.28 to −3.72) | −8.58 (−10.58 to −6.36) |

- The models agree on the average shift and, for the left hand, on the session shifts ($r$ = 0.91)
- For the right hand at 6 μm, B's shift is more than twice C's: C ties the shift to the amplitude, so a weak distractor must give a small shift
- Suggests an effect of the distractor being present, on top of its amplitude

## 81. Model comparison: conclusions
- **Sessions differ:** B predicts far better than A, and model recovery shows this is not B's flexibility alone. A's pooling inflates the JND and bimanual $\gamma_h$.
- **The models agree on the PSE shift:** about −4.2 to −4.5 μm (left) and −5.7 to −6.1 μm (right) on average.
- **The shift depends on amplitude, but not only:** C confirms the trend inside a model, yet B still predicts better. C's misfit points to
  - lapse rates per condition (the strongest evidence)
  - a distractor-presence term, so the shift is not forced to 0 for a weak distractor
  - possibly a bimanual slope, which would let the distractor change the JND
- ⇒ Model B is the model to use for inference for now

---

# Section: Summary

## 82. Comparison of the models
Table:

| | A | B | C |
|---|---|---|---|
| Sessions | pooled | hierarchical | hierarchical |
| Distractor | separate groups | separate groups | covariate $\beta_2 x_\text{dist}$ |
| elpd_loo (SE) | −3344.0 (64.5) | −3180.0 (64.3) | −3217.0 (65.7) |
| difference from B (SE) | −164.1 (19.2) | best | −37.1 (11.9) |
| effective parameters p_loo | 13.4 | 178.7 | 138.6 |
| PSE shift, left / right (μm) | −4.19 / −6.07 | −4.29 / −6.06 | −4.51 / −5.74 |

Small note: PSE shift: bimanual − unimanual, posterior mean; for B and C averaged over the 42 sessions. Numbers from Fit_A, Fit_B and Fit_C_Report.

- All three models agree on the PSE shift, to within about 0.3 μm for each hand
- B predicts best: sessions differ (B ≫ A), and the shift is not simply proportional to distractor amplitude (B > C)

## 83. Conclusions
- The distractor lowers the PSE: by about 4.2–4.5 μm (left) and 5.7–6.1 μm (right), in all three models.
- It raises the rate of "high" responses to weak stimuli ($\gamma_h$).
- The sessions differ; a hierarchical model is needed.
- The shift grows with distractor amplitude, but a weak distractor still shifts the PSE more than a proportional model allows (right hand).

## 84. Next steps
- Revise Model C: lapse rates per condition, a distractor-presence term, possibly a bimanual slope.
- Model recovery for B against the revised C.
- Further experiments: stimulus levels spaced more closely around the PSE, to pin down steep slopes, and levels further out, so the curves reach their asymptotes and the lapse rates are identified.

---

# Backup (currently commented out in Presentation.tex)

## B1. Backup: Model B sampling diagnostics
- Figure: Model_B/Plots_B/Fit_Plots_B/trace_session_params_left_bi.png
- Caption: TODO: no divergences, $\hat{R} = 1.00$ (Fit_B_Report, "Sampling diagnostics")

## B2. Backup: Model C sampling diagnostics
- Figure: Model_C/Plots_C/Fit_Plots_C/trace_session_params_left.png
- Caption: TODO: 2 divergences in 20,000 draws, $\hat{R} = 1.000$ (Fit_C_Report)

## B3. Backup: Model C session PSEs
- Figure: Model_C/Plots_C/Fit_Plots_C/PSE_by_session.png
- Caption: (none)
