# Presentation outline (short version)

Editable outline of `Presentation.tex` and `sections/*.tex`, a ~30 min version of Presentation_ABC
for the lab that collected the data. Same sections, subsections and slide titles as Presentation_ABC;
slides not marked are copied unchanged from it, `[trimmed]` marks a shortened slide.
`Figure:` names the image on a figure slide, `Caption:` is the line under it. Math is written as in LaTeX.
`### Subsection:` marks a subsection. `[gray] ... [end gray]` marks text shown in light gray, `[blue] ... [end blue]` text in blue, `[red]`/`[amber]` likewise.

---

## 1. Title slide
- Title: Modeling Bimanual Perceptual Interactions in Macaque Monkeys
- Subtitle: Bayesian Hierarchical Modeling
- Author: Vera Schroeder
- Date: November 2026

## 2. Outline
- (automatic table of contents from the section names)

---

# Section: Task and Data

## 3. The task: 2AFC categorization of vibration amplitude
- stimulus $x_\text{stim} \in \{6, 12, 18, 24, 32, 38, 44, 50\}$ μm
  - administered to one finger of one hand (left or right)
- [gray] distractor $x_\text{dist} \in \{0, x_\text{d}\}$ μm
  - $x_\text{d} \in [6, 32]$ μm, value fixed for each session
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
  - lower bound on $\mathbf{P}(\text{guess} = \text{"high"})$, even to the lowest stimulus amplitudes
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

## 7. The data
[trimmed: the trial-count table is replaced by the amplitude schedule]
- One monkey (Sirius), 12,937 trials, over 42 sessions of about 308 trials
- 4 groups: hand × (unimanual, bimanual)
- Each session uses a single distractor amplitude on its bimanual trials:
- Figure below: Presentation_Short/amplitude_schedule.png (made by Make_Amplitude_Schedule.py from Sirius_Data.pkl)

## 8. Three models
Table:

| Model | Sessions | Distractor |
|---|---|---|
| A | pooled: one curve per group | uni and bi fit separately |
| B | hierarchical: one curve per group and session | uni and bi fit separately |
| C | hierarchical: one curve per hand and session | covariate: shift $\beta_2 x_\text{dist}$ |

Below the table: For each model: convergence study on synthetic data, then the fit to the real data.

---

# Section: Model A: Sessions Pooled

### Subsection: Model Structure

## 9. Model A: graphical model
- Left: TikZ plate diagram: $\beta_0, \beta_1, \gamma_h, \gamma_l$ → $\psi_i$ ← $x_i$; $\psi_i$ → $r_i$ (observed);
  trial plate ($i = 1,\dots,n$) inside an outer plate labelled "each of the 4 groups"
- Right (legend):
  - circles: random variables
  - shaded: observed
  - rectangles: deterministic
  - $x_i$: fixed covariate
  - outer plate: the same model, fit independently to each group

### Subsection: Synthetic Data
(numbers from Model_A/Convergence_A_Report)

## 10. Model A on synthetic data: what is recoverable
[trimmed]
At the size of the real data (about 3,230 trials per group):
- **Recoverable:** PSE, JND, $\gamma_h$, unbiased and with calibrated 95% HDIs
- **Not reliably recoverable:** small $\gamma_l$, biased upward
- **Caveat:** data simulated with one curve per group, so this says nothing about variation between sessions

### Subsection: Real Data
(numbers from Model_A/Fit_A_Report)

## 11. Model A on the real data: fitted curves
- Figures side by side: Model_A/Plots_A/Fit_Plots_A/curves_left.png, curves_right.png
- Caption: Unimanual in blue, bimanual in red, with the observed proportions pooled over sessions; dashed: the trained threshold of 28 μm.

## 12. Model A on the real data: effect of the distractor on PSE
- Figures stacked, no caption, both cropped in LaTeX from Model_A/Plots_A/Fit_Plots_A/:
  - top ("uni vs bi"): the PSE row of uni_bi_posteriors.png, unimanual and bimanual posteriors overlaid, left and right hand
  - bottom ("bi − uni"): the PSE row of effect_densities.png, the posterior of the difference, with P(>0)
- "Left Hand" / "Right Hand" above the columns; legend line below: Unimanual (blue), Bimanual (red), 95% HDI shaded, mean dotted; green dashed: trained threshold

## 13. Model A: conclusions
[trimmed]
- The pooled data are reproduced at every amplitude
- Effect of the distractor (bimanual − unimanual):
  - **PSE:** lowered, by about 4 μm (left) and 6 μm (right)
  - **JND:** raised for the left hand; unclear for the right
  - **$\gamma_h$:** raised, by about 0.04
  - **$\gamma_l$:** no clear effect
- ⇒ A curve for every session: Model B

---

# Section: Model B: Sessions Hierarchical

### Subsection: Model Structure

## 14. Model B: graphical model
- TikZ plate diagram: hyperparameters $\mu_{\beta_0}, \sigma_{\beta_0}, \mu_{\beta_1}, \sigma_{\beta_1}, \mu_{\gamma_h}, \sigma_{\gamma_h}, \mu_{\gamma_l}, \sigma_{\gamma_l}$ → session parameters $\beta_{0,s}, \beta_{1,s}, \gamma_{h,s}, \gamma_{l,s}$ → $\psi_i$ ← $x_i$; $\psi_i$ → $r_i$ (observed)
  - plates: trials $i$ in session $s$, inside sessions $s = 1,\dots,42$, inside "each of the 4 groups"
- Small note below: Circles: random variables (shaded: observed); rectangle: deterministic; $x_i$: fixed covariate. The same model is fit independently to each group.

### Subsection: Synthetic Data
(numbers from Model_B/Convergence_B_Report)

## 15. Model B on synthetic data: what is recoverable
[trimmed]
At the size of the real data (42 sessions of about 308 trials), at values like the real ones:
- **Recoverable:** each session's PSE and JND, and how much they vary between sessions
- **Weakly determined:** the lapse rates, especially session by session
- **Not identified:** how much the lapse rates vary between sessions; the estimate reflects the prior

### Subsection: Real Data
(numbers from Model_B/Fit_B_Report)

## 16. Model B on the real data: fitted curves in one session (04-13)
- Figures side by side (no caption): Model_B/Plots_B/Fit_Plots_B/curves_left_04-13.png, curves_right_04-13.png

## 17. Model B on the real data: PSE, left hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the left-hand row of PSE_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the left-hand row of effect_PSE_by_session.png (bimanual − unimanual in each session), with the session labels

## 18. Model B on the real data: PSE, right hand
- Stacked, no caption, aligned so each session's estimate sits above its effect (cropped in LaTeX from Model_B/Plots_B/Fit_Plots_B/):
  - top: the right-hand row of PSE_uni_bi_by_session.png (unimanual and bimanual in each session)
  - bottom: the right-hand row of effect_PSE_by_session.png (bimanual − unimanual in each session), with the session labels
- Key line below (the right-hand rows have no legend in the PNGs): unimanual (blue), bimanual (red): session mean and 95% HDI; bands: typical session; green dashed: trained threshold. bi − uni: filled if the session's 95% HDI excludes 0; orange band: average over sessions

## 19. Model B: PSE shift vs distractor amplitude (exploratory)
- Figure (no caption, title cropped): Model_B/Plots_B/Fit_Plots_B/effect_vs_distractor_amplitude.png

## 20. Model B: conclusions
[trimmed]
- Reliable fit; no systematic misfit, pooled or within sessions
- Effect of the distractor (bimanual − unimanual), average over sessions:
  - **PSE:** lowered, by about 4 μm (left) and 6 μm (right), in nearly every session
  - **JND:** raised for the left hand, driven by a few sessions (possibly lapses absorbed into the slope)
  - **$\gamma_h$:** raised; **$\gamma_l$:** no effect
- The PSE shift grows with distractor amplitude (exploratory; amplitude is confounded with session order)
- ⇒ Put the distractor amplitude into the model: Model C

---

# Section: Model C: Distractor Amplitude as a Covariate

### Subsection: Model Structure

## 21. Model C: including the distractor as a covariate
Psychometric function: $\psi(x_\text{stim}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}($[red] $\beta_0 + \beta_1 x_\text{stim}$ [end red]$)$
Suppose the distractor has an impact on the perceived stimulus as follows: $x_\text{perc} = x_\text{stim} + c\, x_\text{dist}$
Then (all [red], except [amber] terms):
  $\beta_0 + \beta_1 x_\text{perc} = \beta_0 + \beta_1 (x_\text{stim} + c\, x_\text{dist})$
  $\qquad = \beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_1 c$ [end amber] $x_\text{dist}$
  $\qquad = \beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_2$ [end amber] $x_\text{dist}$
Considering $x_\text{dist}$ as a covariate: $\psi(x_\text{stim}, x_\text{dist}) = \gamma_h + (1-\gamma_h-\gamma_l)\,\text{logit}^{-1}($[red] $\beta_0 + \beta_1 x_\text{stim} +$ [amber] $\beta_2$ [end amber] $x_\text{dist}$ [end red]$)$

## 22. Model C: graphical model
- TikZ plate diagram: hyperparameters $\mu, \sigma$ of $\beta_0, \beta_1, \beta_2, \gamma_h, \gamma_l$ → session parameters $\beta_{0,s}, \beta_{1,s}, \beta_{2,s}, \gamma_{h,s}, \gamma_{l,s}$ → $\psi_i$ ← $x_{\text{stim},i}$, $x_{\text{dist},i}$; $\psi_i$ → $r_i$ (observed)
  - plates: trials $i$ in session $s$, inside sessions $s = 1,\dots,42$, inside "each hand"
- Small note below: Circles: random variables (shaded: observed); rectangle: deterministic; $x_{\text{stim},i}$, $x_{\text{dist},i}$: fixed covariates. The same model is fit independently to each hand, with unimanual and bimanual trials together.

## 23. Model C: prior predictive check of the distractor effect
- Figure (title cropped): Model_C/Plots_C/Convergence_Plots_C/Prior_Check_C_distractor.png
- Caption: **Model C drawback:** the PSE shift $-\beta_2 x_\text{dist}/\beta_1$ is tied to the distractor amplitude, so weak distractors can only shift the PSE slightly

### Subsection: Synthetic Data
(numbers from Model_C/Convergence_C_Report)

### Subsection: Synthetic Data
(numbers from Model_C/Convergence_C_Report)

## 24. Model C on synthetic data: what is recoverable
[trimmed]
At the size of the real data, if the data follow Model C:
- **Recoverable:** each session's PSE in both conditions, and the distractor effect $\beta_2$ of a typical session
- **Weakly determined:** $\beta_2$ in sessions with a weak distractor, which is pulled toward the typical value
- **Not tested:** data that break Model C's assumptions (shared lapse rates and slope, a shift proportional to amplitude)

### Subsection: Real Data
(numbers from Model_C/Fit_C_Report)

## 25. Model C: the PSE shift grows with distractor amplitude
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/pse_shift_vs_amplitude.png

## 26. Model C on the real data: misfit at the lowest stimulus amplitudes
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/ppc_pooled_by_stimulus.png

## 27. Model C: conclusions
[trimmed]
- **PSE:** lowered by the distractor, more the stronger it is: a typical session shifts 1–2 μm at a 6 μm distractor, 6.5–9 μm at 32 μm
- **JND, $\gamma_h$, $\gamma_l$:** shared by both conditions, so the model cannot show an effect on them
- The misfit sits where its assumptions meet the data:
  - shared lapse rates: too few "high" responses predicted to low stimulus amplitudes on bimanual trials
  - a shift proportional to amplitude: on the right hand, even a 6 μm distractor shifts responses more than Model C allows
- ⇒ Which model describes the data best? Model comparison

---

# Section: Model Comparison and Recovery

### Subsection: Predictive Performance

## 28. Leave-one-out cross-validation
[trimmed: elpd_loo and p_loo columns removed]
How well each model predicts each trial when that trial is left out of the fit (higher is better):

| Model | difference from B (SE) |
|---|---|
| A: sessions pooled | −164.1 (19.2) |
| B: sessions hierarchical | **best** |
| C: distractor as covariate | −37.1 (11.9) |

- B beats A clearly, despite many more parameters
- B beats C, although C needs fewer parameters

### Subsection: Model Recovery: B vs A
(numbers from Model_B/Model_Recov_BA_Report)

## 29. Model recovery: LOO picks the generating model
[trimmed: merges "is B's advantage real?" and the conclusions]
Is B's advantage real, or just flexibility? Simulate datasets from each model on the real design, fit both, and see which one LOO picks:

| | selects A | selects B |
|---|---|---|
| data from A | 9 | 1 |
| data from B | 0 | 10 |

- On data from A, B ties or loses: flexibility alone does not make it win
- The real data look like the data from B: sessions really differ

## 30. Model recovery: $\Delta$ in every replicate
- Figure (no caption, title cropped): Model_B/Plots_B/Model_Recov_Plots_BA/elpd_diff_by_replicate.png

### Subsection: Model C vs Model B
(numbers from Model_C/Fit_C_Report, "Comparison with Model B")

## 31. Model C vs Model B: session PSE shifts
- Figure (no caption, title cropped): Model_C/Plots_C/Fit_Plots_C/pse_shift_C_vs_B.png

## 32. Model comparison: conclusions
[trimmed]
- **Sessions differ:** B predicts far better than A, and model recovery shows this is not B's flexibility alone
- **The models agree on the PSE shift:** about 4 μm (left) and 6 μm (right) on average
- **B predicts better than C mainly because of lapses:** C's one lapse rate for both conditions misses the extra "high" responses to low stimulus amplitudes on bimanual trials
- Less clearly: a distractor-presence effect (right hand) and a bimanual slope
- ⇒ Model B is the model to use for inference for now

---

# Section: Summary

## 33. Comparison of the models
[trimmed: elpd and p_loo rows replaced by a ranking]
Table:

| | A | B | C |
|---|---|---|---|
| Sessions | pooled | hierarchical | hierarchical |
| Distractor | separate groups | separate groups | covariate $\beta_2 x_\text{dist}$ |
| Prediction (LOO) | 3rd | **best** | 2nd |
| PSE shift, left / right (μm) | −4.2 / −6.1 | −4.3 / −6.1 | −4.5 / −5.7 |

- All three models agree on the PSE shift
- B predicts best: sessions differ (B ≫ A), and lapse rates differ between conditions (B > C)

## 34. Conclusions
- The distractor lowers the PSE: by about 4.2–4.5 μm (left) and 5.7–6.1 μm (right), in all three models.
- It raises the rate of "high" responses to low stimulus amplitudes on bimanual trials ($\gamma_h$); whether this grows with distractor amplitude is open.
- The sessions differ; a hierarchical model is needed.
- The shift grows with distractor amplitude, and a weak distractor may shift the PSE more than a proportional model allows (right hand).

## 35. Open questions
[new for this deck]
- **The extra "high" responses to low stimulus amplitudes on bimanual trials:** they seem to cluster in sessions with a strong distractor (being checked). What could produce them: responses to the distractor hand, attention, motor errors?
- **Amplitude and time:** the distractor amplitude rose over the sessions, so a growing effect could also be training or adaptation. Was the schedule chosen for a reason? Did anything else change over those months?
- **Left vs right:** the right hand shifts more, even at weak distractors. Is a hand asymmetry expected?

## 36. Next steps
- A new Model D, revising Model C: lapse rates per condition, testing whether the bimanual $\gamma_h$ depends on distractor amplitude (or session order).
- A distractor-presence term only if it improves prediction.
- Model recovery for B against D.
