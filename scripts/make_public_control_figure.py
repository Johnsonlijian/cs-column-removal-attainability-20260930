from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
rows = json.loads((DATA_DIR / "public_paired_opensees.json").read_text(encoding="utf-8"))["rows"]
stories = sorted({int(r["stories"]) for r in rows})
colors = {2: "#1f77b4", 4: "#d62728", 8: "#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
censored_label_used = False
ratio_censored_label_used = False
for h in stories:
    g = [r for r in rows if int(r["stories"]) == h]
    k = [r["multiplier"] for r in g]
    intact_censored = [r["intact_limit_reason"] == "max_load_factor_reached" for r in g]
    removed_censored = [r["removed_limit_reason"] == "max_load_factor_reached" for r in g]
    intact = [np.nan if censored else r["intact_limit_load_factor"]
              for r, censored in zip(g, intact_censored)]
    removed = [np.nan if censored else r["removed_limit_load_factor"]
               for r, censored in zip(g, removed_censored)]
    ratio_censored = [a or b for a, b in zip(intact_censored, removed_censored)]
    ratio = [np.nan if censored else r["removed_over_intact"]
             for r, censored in zip(g, ratio_censored)]
    c = colors.get(h, None)
    axes[0].plot(k, intact, "o-", color=c, lw=2.2, label=f"{h}-story intact")
    axes[0].plot(k, removed, "s--", color=c, lw=2.0, alpha=0.9, label=f"{h}-story, target removed")
    axes[1].plot(k, ratio, "o-", color=c, lw=2.2, label=f"{h}-story")

    # A run that hits the configured load ceiling provides a lower bound on
    # the limit factor.  Show it as an open marker and leave it out of the
    # connecting line so the figure cannot be read as an exact finite value.
    for ki, value, censored in zip(k, [r["intact_limit_load_factor"] for r in g], intact_censored):
        if censored:
            label = "censored at load ceiling" if not censored_label_used else "_nolegend_"
            axes[0].plot(ki, value, marker="v", ms=8, mfc="white", mec=c,
                         mew=1.6, linestyle="none", color=c, label=label)
            censored_label_used = True
    for ki, value, censored in zip(k, [r["removed_limit_load_factor"] for r in g], removed_censored):
        if censored:
            label = "_nolegend_"
            axes[0].plot(ki, value, marker="v", ms=8, mfc="white", mec=c,
                         mew=1.6, linestyle="none", color=c, label=label)
            censored_label_used = True
    for ki, censored in zip(k, ratio_censored):
        if censored:
            label = "censored ratio" if not ratio_censored_label_used else "_nolegend_"
            axes[1].plot(ki, 0.98, marker="v", ms=8, mfc="white", mec=c,
                         mew=1.6, linestyle="none", color=c, label=label)
            ratio_censored_label_used = True

axes[0].set_xscale("log")
axes[0].set_xlabel("multiplier k on all other main members")
axes[0].set_ylabel("OpenSees gravity load factor at limit")
axes[0].set_title("Public-frame paired nonlinear control")
axes[0].axhline(12.0, color="0.35", lw=1.0, ls=":", label="configured load ceiling")
axes[0].grid(True, alpha=0.22)
axes[0].legend(fontsize=7.6, ncol=3, frameon=False)
axes[1].set_xscale("log")
axes[1].axhline(1.0, color="0.35", lw=1.2, ls=":")
axes[1].set_xlabel("multiplier k on all other main members")
axes[1].set_ylabel("removed / intact limit factor")
axes[1].set_title("Relative consequence can change direction")
axes[1].grid(True, alpha=0.22)
axes[1].legend(fontsize=7.6, ncol=2, frameon=False)
fig.suptitle(
    "Source-verified SAC/Elkady–Lignos geometry; bilinear-hinge adapter control",
    fontsize=14,
    fontweight="bold",
)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT_DIR / f"fig_public_paired_control.{ext}", dpi=300 if ext == "png" else None)
plt.close(fig)
print("wrote public paired control figure")
