# Presentation outline (short version)

Editable outline of `Presentation.tex` and `sections/*.tex`. One `##` heading per slide.
`Figure:` names the image on a figure slide, `Takeaway:` is the line under it. Math is written as in LaTeX.
`[gray] ... [end gray]` marks text shown in light gray.

Audience: the neuroscientist whose lab collected the data (knows the experiment, not a mathematician).
About 30 minutes, a research update. Fewer words and numbers than Presentation_ABC.

---

## 1. Title slide
- Title: How a Distractor on One Hand Changes Judgments on the Other
- Subtitle: Research update
- Author: Vera Schroeder
- Date: November 2026

---

# Section: Terms

## 2. Today
1. The terms I use for the data
2. Three models of the data
3. What they show
4. Open questions for you

## 3. Two kinds of trial
- TikZ trial diagram from Presentation_ABC: unimanual trial (no distractor) | bimanual trial (with distractor)
- **Group:** hand × trial type, so four groups: left/right, unimanual/bimanual

## 4. Each session has one distractor amplitude
- Figure: Presentation_Short/amplitude_schedule.png (made by Make_Amplitude_Schedule.py from Sirius_Data.pkl)
- 42 sessions of about 300 trials, from one monkey (Sirius)

## 5. The psychometric curve
- Left: Presentation_ABC/example_psych_func.png
- Right:
  - **PSE** (bias): called "high" half the time
    - [gray] lower PSE = more "high" answers [end gray]
  - **JND** (sensitivity): how fast the curve rises
    - [gray] larger JND = less precise [end gray]
  - **Lapses:** answers the stimulus does not explain
    - [gray] $\gamma_h$ = floor, $\gamma_l$ = gap below the ceiling [end gray]
- Bottom: **Distractor effect** = bimanual − unimanual, for each hand

---

# Section: Models

## 6. Three models
- **Model A:** one curve per group, all sessions pooled
- **Model B:** a curve for every session
  - [gray] sessions may differ, but are assumed to resemble each other [end gray]
- **Model C:** like B, but the distractor's shift is proportional to its amplitude
  - [gray] one curve per hand, with both trial types [end gray]

## 7. How I check a model
1. **Simulated data:** make data with known answers, and check that the model gets them back
2. **Real data:** check that the fitted model reproduces what the monkey did
3. **Compare models:** how well each predicts trials left out of its fit

---

# Section: Findings

## 8. The distractor makes "high" answers more likely
- Figures side by side: Model_A/Plots_A/Fit_Plots_A/curves_left.png, curves_right.png
- Takeaway: The bimanual curve (red) moves left: PSE about 4 μm lower (left hand), 6 μm (right)

## 9. The shift appears in nearly every session
- Figure (title cropped): Model_B/Plots_B/Fit_Plots_B/effect_PSE_by_session.png
- Takeaway: One session per point, in date order (Model B). Below 0: the distractor lowers the PSE

## 10. The shift grows with the distractor's amplitude
- Figure (title cropped): Model_C/Plots_C/Fit_Plots_C/pse_shift_vs_amplitude.png
- Takeaway: One session per point (Model C). But amplitude also rose over time

## 11. A floor under the bimanual curve
- Figures side by side: the Model A curves of slide 8, with orange circles around the weakest stimuli (6 and 12 μm)
- Takeaway: With a distractor, the monkey sometimes answers "high" to the weakest stimuli

## 12. Less clear
- **Sensitivity:** the left-hand JND is slightly larger with a distractor
  - [gray] driven by a few sessions; may be the same extra "high" answers [end gray]
- **Weak distractors:** on the right hand, a 6 μm distractor shifts the PSE more than its amplitude predicts
  - [gray] not yet shown to matter for prediction [end gray]

## 13. Which model predicts best?
- How much worse than Model B each model predicts left-out trials ([gray] difference in log predictive score [end gray])
- TikZ bar chart: Model A 164.1, Model C 37.1, Model B best (numbers from Fit_B_Report and Fit_C_Report)
- B ≫ A: sessions really differ
- B > C: C has one floor for both trial types

## 14. Why Model C falls short
- Figure: the bimanual row of Model_C/Plots_C/Fit_Plots_C/ppc_pooled_by_stimulus.png (cropped in LaTeX)
- Small note: Black: observed. Blue: Model C's prediction.
- Takeaway: On bimanual trials, C predicts too few "high" answers to the weakest stimuli

## 15. Summary
- The distractor lowers the PSE: about 4 μm (left hand) and 6 μm (right), in all three models
- The shift is larger in sessions with a stronger distractor
- With a distractor, the monkey sometimes answers "high" to the weakest stimuli
- Sessions differ, so each session needs its own curve

---

# Section: Next

## 16. Questions for you
1. **The extra "high" answers:** they seem to cluster in sessions with a strong distractor (being checked). Could the monkey sometimes respond to the distractor hand?
2. **Amplitude and time:** the distractor amplitude rose over the sessions, so a growing effect could also be training or adaptation. Was the schedule chosen for a reason? Did anything else change over those months?

## 17. Next steps
- **Model D:** separate lapse rates for unimanual and bimanual trials
- Test whether the bimanual lapse rate depends on the distractor amplitude, or on time
- Check Model D on simulated data before fitting the real data
