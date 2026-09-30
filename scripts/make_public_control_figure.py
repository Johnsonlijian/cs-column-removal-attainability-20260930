from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
rows = json.loads((DATA_DIR / "public_paired_opensees.json").read_text(encoding="utf-8"))["rows"]
stories = sorted({int(r["stories"]) for r in rows})
colors = {2: "#1f77b4", 4: "#d62728", 8: "#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
for h in stories:
    g = [r for r in rows if int(r["stories"]) == h]
    k = [r["multiplier"] for r in g]
    intact = [r["intact_limit_load_factor"] for r in g]
    removed = [r["removed_limit_load_factor"] for r in g]
    ratio = [r["removed_over_intact"] for r in g]
    c = colors.get(h, None)
    axes[0].plot(k, intact, "o-", color=c, lw=2.2, label=f"{h}-story intact")
    axes[0].plot(k, removed, "s--", color=c, lw=2.0, alpha=0.9, label=f"{h}-story, target removed")
    axes[1].plot(k, ratio, "o-", color=c, lw=2.2, label=f"{h}-story")

axes[0].set_xscale("log")
axes[0].set_xlabel("multiplier k on all other main members")
axes[0].set_ylabel("OpenSees gravity load factor at limit")
axes[0].set_title("Public-frame paired nonlinear control")
axes[0].grid(True, alpha=0.22)
axes[0].legend(fontsize=8, ncol=2, frameon=False)
axes[1].set_xscale("log")
axes[1].axhline(1.0, color="0.35", lw=1.2, ls=":")
axes[1].set_xlabel("multiplier k on all other main members")
axes[1].set_ylabel("removed / intact limit factor")
axes[1].set_title("Relative consequence can change direction")
axes[1].grid(True, alpha=0.22)
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle(
    "Source-verified SAC/Elkady–Lignos geometry; bilinear-hinge adapter control",
    fontsize=14,
    fontweight="bold",
)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT_DIR / f"fig_public_paired_control.{ext}", dpi=300 if ext == "png" else None)
plt.close(fig)
print("wrote public paired control figure")
