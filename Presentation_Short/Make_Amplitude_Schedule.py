#!/usr/bin/env python3
"""
Plot the distractor amplitude used in each session, in date order.

Reads Sirius_Data.pkl and writes Presentation_Short/amplitude_schedule.png for
the slide on the sessions. Each session uses one distractor amplitude on all of
its bimanual trials; the plot shows how that amplitude changed over the sessions.
Run from the repo root:  python Presentation_Short/Make_Amplitude_Schedule.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "Sirius_Data.pkl"), "rb") as f:
    trials = pickle.load(f)["data"]

bi = trials[trials["group"].str.endswith("_bi")]
amp = bi.groupby("sess_idx").agg(date=("sess_date", "first"), amp=("dist_amp", "unique"))
# one distractor amplitude per session
assert all(len(a) == 1 for a in amp["amp"])
amp["amp"] = [a[0] for a in amp["amp"]]
amp = amp.sort_index()

fig, ax = plt.subplots(figsize=(10, 3.4), dpi=200)
ax.plot(range(len(amp)), amp["amp"], color="#B9B8B5", lw=1, zorder=1)
ax.scatter(range(len(amp)), amp["amp"], color="#EB6834", s=34, zorder=2)
ax.set_xticks(range(len(amp)))
ax.set_xticklabels(amp["date"], rotation=90, fontsize=7)
ax.set_yticks([6, 12, 18, 24, 32])
ax.set_ylabel(r"distractor amplitude ($\mu m$)")
ax.set_xlabel("session (date order)")
ax.set_xlim(-0.8, len(amp) - 0.2)
ax.set_ylim(0, 36)
ax.grid(axis="y", color="#E6E5E2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(here, "amplitude_schedule.png"))
print(f"{len(amp)} sessions; wrote amplitude_schedule.png")
