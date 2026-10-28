# Presentation outline

Editable outline of `Presentation.tex` and `sections/*.tex`. One `##` heading per slide.
`TODO:` marks the red placeholder boxes. `Figure:` names the image on a figure slide,
`Caption:` is the line under it. Math is written as in LaTeX.
`### Subsection:` marks a subsection (it appears in the table of contents). `[gray] ... [end gray]` marks text shown in light gray, `[blue] ... [end blue]` text in blue.

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
| C | hierarchical, one group per hand | covariate: shift $\beta_2 d$ |

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
- Fixed true hyperparameters per group (group slopes $\mu_{\beta_1} = 3$ for the left hand, 6 for the right); session parameters drawn from them
- Two sweeps, each run twice with new data:
  - sessions: 5, 10, 20, 40, 80 sessions of 308 trials (tests the hyperparameters)
  - trials: 42 sessions of 20, 40, 80, 160, 320 trials (tests the session parameters)
- The real data (42 sessions of about 308 trials) are closest to the 40-session and 320-trial fits
- All 20 fits sampled well: no divergences, $\hat{R} \le 1.010$, bulk ESS $\ge 579$

## 29. Model B on synthetic data: group means
- Figure (no caption, title cropped): Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png

## 30. Model B on synthetic data: between-session sds
- Figure (no caption, title cropped): Model_B/Plots_B/Convergence_Plots_B/hyper_sds_recovery_vs_sessions.png

## 31. Model B on synthetic data: session-level recovery
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

## 32. Model B on synthetic data: what is recoverable
At the size of the real data (42 sessions of about 308 trials):
- **Recoverable:** per-session PSE and JND; the group means $\mu_{\beta_0}$; $\mu_{\beta_1}$ for the left hand; the bias spread $\sigma_{\beta_0}$; $\mu_{\gamma_h}$ except for left bimanual
- **Weakly constrained:** $\mu_{\beta_1}$ for steep curves (slope 6, right hand): the curve rises within one stimulus spacing, so only 2 levels inform it
- **Biased in some datasets:** a lapse rate whose side of the curve never reaches its asymptote: $\gamma_l$ for left unimanual (too high), $\gamma_h$ for left bimanual (too low)
- **Not identified:** $\sigma_{\beta_1}$, $\sigma_{\gamma_h}$, $\sigma_{\gamma_l}$ when their true values are near 0; their posteriors reflect the prior
- Session-level HDIs mostly conservative; PSE coverage slightly low (0.90–0.94)

### Subsection: Real Data
(numbers from Model_B/Fit_B_Report)

## 33. Model B on the real data: a typical session
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

## 34. Model B on the real data: fitted curves in one session (04-13)
- Figures side by side (no caption): Model_B/Plots_B/Fit_Plots_B/curves_left_04-13.png, curves_right_04-13.png

## 35. Model B on the real data: PSE in each session
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/PSE_uni_bi_by_session.png (unimanual and bimanual per session, both hands)

## 36. Model B on the real data: effect of the distractor on PSE
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_PSE_by_session.png (bimanual − unimanual per session)

## 37. Model B on the real data: JND in each session
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/JND_uni_bi_by_session.png

## 38. Model B on the real data: effect of the distractor on JND
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_JND_by_session.png

## 39. Model B on the real data: $\gamma_h$ in each session
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/gamh_uni_bi_by_session.png

## 40. Model B on the real data: effect of the distractor on $\gamma_h$
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_gamh_by_session.png

## 41. Model B on the real data: $\gamma_l$ in each session
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/gaml_uni_bi_by_session.png

## 42. Model B on the real data: effect of the distractor on $\gamma_l$
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_gaml_by_session.png

## 43. Model B: PSE shift vs distractor amplitude (exploratory)
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_vs_distractor_amplitude.png

## 44. Model B on the real data: the pooled data are reproduced
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/ppc_pooled_by_stimulus.png

## 45. Model B: conclusions
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

## 46. Model C
  $\psi(x, d) = \gamma_h + (1 - \gamma_h - \gamma_l)\,\text{logit}^{-1}(\beta_0 + \beta_1 x + \beta_2 d)$

- $d$: distractor amplitude (0 on unimanual trials); one group per hand.
- Uni- and bimanual trials share slope and lapse rates: the distractor only shifts the curve, by an amount proportional to $d$.

TODO: why: test the amplitude trend from Model B inside a model

## 47. Model C: prior on the distractor effect
- Figure: Model_C/Plots_C/Convergence_Plots_C/Prior_Check_C_distractor.png
- Caption: TODO: weakly informative; a weak distractor can only shift a little (Convergence_C_Report)

## 48. Model C: convergence on synthetic data
- Figure: Model_C/Plots_C/Convergence_Plots_C/hyper_means_recovery_vs_sessions.png
- Caption: TODO: all hand-level means recovered, $\mu_{\beta_2}$ well (Convergence_C_Report, "Conclusions")

## 49. Model C: per-session $\beta_2$ is weak for small distractors
- Figure: Model_C/Plots_C/Convergence_Plots_C/session_b2_true_vs_recovered.png
- Caption: TODO: shrinkage toward the hand mean at 6 μm

## 50. Model C: the shift grows with distractor amplitude
- Figure: Model_C/Plots_C/Fit_Plots_C/pse_shift_vs_amplitude.png
- Caption: TODO: $\mu_{\beta_2}$ and the typical shift at 6 and 32 μm (Fit_C_Report, "Effect of the Distractor")

## 51. Model C misfits at the weakest stimuli
- Figure: Model_C/Plots_C/Fit_Plots_C/ppc_pooled_by_stimulus.png
- Caption: TODO: shared lapse rates: 6 of 32 proportions outside (Fit_C_Report, "Posterior predictive check")

## 52. Model C against Model B
- Figure: Model_C/Plots_C/Fit_Plots_C/pse_shift_C_vs_B.png
- Caption: TODO: LOO favors B by 37 (SE 12); 6 μm shifts too small (Fit_C_Report, Comparison)

## 53. Model C: shift by distractor amplitude, observed vs predicted
- Figure: Model_C/Plots_C/Fit_Plots_C/ppc_by_distractor_amplitude.png
- Caption: TODO: right hand at 6 μm outside: a presence effect (Fit_C_Report, amplitude check)

---

# Section: Summary

## 54. Comparison of the models
Table:

| | A | B | C |
|---|---|---|---|
| Sessions | pooled | hierarchical | hierarchical |
| Distractor | separate groups | separate groups | covariate $\beta_2 d$ |
| elpd_loo | TODO | TODO | TODO |
| PSE shift, left / right (μm) | TODO | TODO | TODO |

TODO: fill from Fit_A/B/C_Report; elpd differences from Fit_B and Fit_C

## 55. Conclusions
- The distractor lowers the PSE: TODO: size, both hands.
- It raises the rate of "high" responses to weak stimuli ($\gamma_h$).
- The sessions differ; a hierarchical model is needed.
- The shift grows with distractor amplitude, TODO: but not proportionally?

## 56. Next steps
- Revise Model C: lapse rates per condition, a distractor-presence term, possibly a bimanual slope.
- Model recovery for B against the revised C.
- TODO: further data or experiments

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
