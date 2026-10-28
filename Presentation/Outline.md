# Presentation outline

Editable outline of `Presentation.tex` and `sections/*.tex`. One `##` heading per slide.
`TODO:` marks the red placeholder boxes. `Figure:` names the image on a figure slide,
`Caption:` is the line under it. Math is written as in LaTeX.
`[gray] ... [end gray]` marks text shown in light gray, `[blue] ... [end blue]` text in blue.

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

## 10. Model A
- One psychometric curve per group (hand × condition), shared by all 42 sessions: 16 parameters.
- Priors: $\beta_0 \sim \mathcal{N}(0, 6)$, $\beta_1 \sim \mathcal{N}(4, 2)$, $\gamma = 0.25\,w$ with $w \sim \text{Beta}$.

TODO: one-line motivation: simplest model, a baseline

## 11. Model A: convergence on synthetic data
- Figure: Model_A/Plots_A/Convergence_Plots_A/coverage_vs_size_all_params.png
- Caption: TODO: takeaway; numbers from Convergence_A_Report, "Conclusions"

## 12. Model A: fit to the real data
- Figure: Model_A/Plots_A/Fit_Plots_A/ppc_pooled_by_stimulus.png
- Caption: TODO: pooled check passes: all 32 proportions inside (Fit_A_Report, "Model Checks")

## 13. Model A: effect of the distractor
- Figure: Model_A/Plots_A/Fit_Plots_A/effect_densities.png
- Caption: TODO: PSE shift, $\gamma_h$ and JND effects (Fit_A_Report, Summary)

## 14. Model A misses the variation between sessions
- Figure: Model_A/Plots_A/Fit_Plots_A/ppc_session_cells.png
- Caption: TODO: within single sessions responses vary more than A predicts (Fit_A_Report, "Model Checks")

---

# Section: Model B: Sessions Hierarchical

## 15. Model B
- A curve for every group *and session*, drawn from group-level distributions (means and between-session sds).
- Design-based priors, checked by prior predictive simulation.

TODO: plate diagram or equations (Convergence_B_Report, "Model Structure")

## 16. Model B: convergence on synthetic data
- Figure: Model_B/Plots_B/Convergence_Plots_B/hyper_means_recovery_vs_sessions.png
- Caption: TODO: what is recovered at the real data size (Convergence_B_Report, "Conclusions")

## 17. Model B: per-session recovery
- Figure: Model_B/Plots_B/Convergence_Plots_B/session_params_error_width_coverage_vs_trials.png
- Caption: TODO: per-session PSE and JND reliable at ~300 trials

## 18. Model B: fit to the real data
- Figure: Model_B/Plots_B/Fit_Plots_B/ppc_pooled_by_stimulus.png
- Caption: TODO: all 32 proportions inside; 99.8% of session cells (Fit_B_Report, "Posterior predictive checks")

## 19. Model B: PSE shift in every session
- Figure: Model_B/Plots_B/Fit_Plots_B/effect_PSE_by_session.png
- Caption: TODO: shift negative in 40/42 (left) and 42/42 (right) sessions (Fit_B_Report, Summary)

## 20. Model B: the shift grows with distractor amplitude (exploratory)
- Figure: Model_B/Plots_B/Fit_Plots_B/effect_vs_distractor_amplitude.png
- Caption: TODO: about −2 μm per 10 μm; confounded with session order

## 21. Model B against Model A
- Figure: Model_B/Plots_B/Fit_Plots_B/effect_A_vs_B.png
- Caption: TODO: same effects; B better by LOO, 164 (SE 19) (Fit_B_Report, "Comparison with Model A")

## 22. Is B's advantage real? Model recovery
- Figure: Model_B/Plots_B/Model_Recov_Plots_BA/elpd_diff_by_replicate.png
- Caption: TODO: B wins on all data from B, ties or loses on data from A (Model_Recov_BA_Report)

---

# Section: Model C: Distractor Amplitude as a Covariate

## 23. Model C
  $\psi(x, d) = \gamma_h + (1 - \gamma_h - \gamma_l)\,\text{logit}^{-1}(\beta_0 + \beta_1 x + \beta_2 d)$

- $d$: distractor amplitude (0 on unimanual trials); one group per hand.
- Uni- and bimanual trials share slope and lapse rates: the distractor only shifts the curve, by an amount proportional to $d$.

TODO: why: test the amplitude trend from Model B inside a model

## 24. Model C: prior on the distractor effect
- Figure: Model_C/Plots_C/Convergence_Plots_C/Prior_Check_C_distractor.png
- Caption: TODO: weakly informative; a weak distractor can only shift a little (Convergence_C_Report)

## 25. Model C: convergence on synthetic data
- Figure: Model_C/Plots_C/Convergence_Plots_C/hyper_means_recovery_vs_sessions.png
- Caption: TODO: all hand-level means recovered, $\mu_{\beta_2}$ well (Convergence_C_Report, "Conclusions")

## 26. Model C: per-session $\beta_2$ is weak for small distractors
- Figure: Model_C/Plots_C/Convergence_Plots_C/session_b2_true_vs_recovered.png
- Caption: TODO: shrinkage toward the hand mean at 6 μm

## 27. Model C: the shift grows with distractor amplitude
- Figure: Model_C/Plots_C/Fit_Plots_C/pse_shift_vs_amplitude.png
- Caption: TODO: $\mu_{\beta_2}$ and the typical shift at 6 and 32 μm (Fit_C_Report, "Effect of the Distractor")

## 28. Model C misfits at the weakest stimuli
- Figure: Model_C/Plots_C/Fit_Plots_C/ppc_pooled_by_stimulus.png
- Caption: TODO: shared lapse rates: 6 of 32 proportions outside (Fit_C_Report, "Posterior predictive check")

## 29. Model C against Model B
- Figure: Model_C/Plots_C/Fit_Plots_C/pse_shift_C_vs_B.png
- Caption: TODO: LOO favors B by 37 (SE 12); 6 μm shifts too small (Fit_C_Report, Comparison)

## 30. Model C: shift by distractor amplitude, observed vs predicted
- Figure: Model_C/Plots_C/Fit_Plots_C/ppc_by_distractor_amplitude.png
- Caption: TODO: right hand at 6 μm outside: a presence effect (Fit_C_Report, amplitude check)

---

# Section: Summary

## 31. Comparison of the models
Table:

| | A | B | C |
|---|---|---|---|
| Sessions | pooled | hierarchical | hierarchical |
| Distractor | separate groups | separate groups | covariate $\beta_2 d$ |
| elpd_loo | TODO | TODO | TODO |
| PSE shift, left / right (μm) | TODO | TODO | TODO |

TODO: fill from Fit_A/B/C_Report; elpd differences from Fit_B and Fit_C

## 32. Conclusions
- The distractor lowers the PSE: TODO: size, both hands.
- It raises the rate of "high" responses to weak stimuli ($\gamma_h$).
- The sessions differ; a hierarchical model is needed.
- The shift grows with distractor amplitude, TODO: but not proportionally?

## 33. Next steps
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
