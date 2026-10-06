"""Figure for the independent nonlinear cross-check of the certificate.

Panels
------
(a) Pushdown load--displacement curves of the frozen witness frame in the exact
    case, with the certificate's local and complete values marked.
(b) The same for the strengthened (inexact) case, where the certificate reports a
    negative minimum margin and a large local-over-complete gap.
(c) Verification of the solver against the analytical two-story boundary
    ``min(4k, 5)``.

The curves come from ``data/nonlinear_pushdown.json``; the figure stores both a
vector PDF/SVG master and a raster preview.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures"
DATA = json.loads((ROOT / "data" / "nonlinear_pushdown.json").read_text(encoding="utf-8"))
CASE = {c["case"]: c for c in DATA["cases"]}
LANE_STYLE = {
    "first_order_perfectly_plastic": ("#1f77b4", "-", "first order, perfectly plastic"),
    "second_order_perfectly_plastic": ("#d62728", "--", "P-Delta, perfectly plastic"),
    "second_order_degrading": ("#2ca02c", ":", "P-Delta, degrading hinges"),
}

fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9), constrained_layout=True)


def curve(ax, case_name: str, title: str) -> None:
    case = CASE[case_name]
    cert = case["certificate"]
    for row in case["runs"]:
        color, style, label = LANE_STYLE[row["model_class"]]
        points = [h for h in row["history"] if h.get("ok")]
        if not points:
            continue
        disp = np.abs(np.array([p["disp"] for p in points], float))
        lam = np.array([p["lambda"] for p in points], float)
        ax.plot(disp, lam, style, color=color, lw=2.0, label=label)
        limit = row["lambda_limit"]
        ax.plot(disp[-1], lam[-1], "o", ms=5, color=color, mfc="white", mew=1.4)
    ax.axhline(cert["complete"], color="0.25", lw=1.1, ls="-.",
               label=f"certificate complete = {cert['complete']:.3f}")
    ax.axhline(cert["local"], color="0.55", lw=1.1, ls="--",
               label=f"certificate local = {cert['local']:.3f}")
    ax.set_xlabel("pushdown displacement at the cut node (model units)")
    ax.set_ylabel(r"pushdown multiplier $\lambda$")
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=7.4, frameon=False)


curve(axes[0], "witness_baseline_H2B2_removal_s1g2",
      "(a) Exact case: certificate margin $+0.935$")
curve(axes[1], "witness_memberwise_H2B2_removal_s1g2",
      "(b) Inexact case: certificate margin $-8.537$")

analytic = DATA["cases"][0]
rows = []
for name in ("canonical_k1", "canonical_k2", "canonical_k5"):
    case = CASE[name]
    k = float(name.split("k")[1])
    first = next(r for r in case["runs"]
                 if r["model_class"] == "first_order_perfectly_plastic")
    rows.append((k, first["lambda_limit"], min(4.0 * k, 5.0)))
rows.sort()
ks = [r[0] for r in rows]
limits = [r[1] for r in rows]
exact = [r[2] for r in rows]
axes[2].plot(ks, exact, "s-", color="0.35", lw=2.0, ms=6,
             label=r"analytical $\min(4k,5)$")
axes[2].plot(ks, limits, "o", color="#1f77b4", ms=8, mfc="white", mew=1.8,
             label="pushdown, first order")
for k, value, target in rows:
    offset = (10, -4) if k > 1.0 else (10, 6)
    axes[2].annotate(f"{value:.3f}\n({(value / target - 1.0) * 100:+.1f}%)",
                     (k, value), textcoords="offset points", xytext=offset,
                     fontsize=7.2, color="0.2")
axes[2].set_xlabel("member multiplier $k$ of the two-story construction")
axes[2].set_ylabel(r"collapse multiplier $\lambda$")
axes[2].set_title("(c) Solver verification against the analytical boundary",
                  fontsize=10.5)
axes[2].set_xticks(ks)
axes[2].set_xlim(0.85, 5.35)
axes[2].set_ylim(3.55, 5.25)
axes[2].grid(True, alpha=0.22)
axes[2].legend(fontsize=8.0, frameon=False, loc="lower right")

fig.suptitle(
    "Independent second-order plastic-hinge pushdown cross-check of the certificate",
    fontsize=14,
    fontweight="bold",
)
OUT_DIR.mkdir(exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT_DIR / f"fig_pushdown_crosscheck.{ext}",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("wrote pushdown cross-check figure")
_ = analytic
