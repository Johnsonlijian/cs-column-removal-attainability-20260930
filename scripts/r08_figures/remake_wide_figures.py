"""Render the four widest body figures at true print size.

Why this exists
---------------
These figures were authored on canvases between 202 mm and 396 mm wide while the
manuscript includes every body figure at ``[width=0.99\\linewidth]`` = 142.5 mm.
LaTeX therefore shrank them by up to 2.8x, and because a geometric shrink scales
type along with geometry, labels that were authored at 7-8 pt printed at
2.6-5.1 pt. Rescaling the PDF page does not fix this: it preserves relative
sizes.

The only real fix is to lay the figure out for the width it will be printed at,
with point sizes chosen directly for that width. Stacking panels vertically
gives each panel the full 142.5 mm, which is what makes 7-8 pt labels legible
without discarding content.

All four figures are redrawn from the same frozen JSON data as the originals, so
every plotted value is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FIGDIR = PKG / "figures"
sys.path.insert(0, str(HERE))

PRINT_W_IN = 142.5 / 25.4          # 5.610 in

# Labels authored below the base size are lifted by this factor so that the
# smallest printed type stays above 7 pt at the include width.
SMALL_TEXT_BOOST = 1.55


def boost(ax):
    for t in ax.texts:
        t.set_fontsize(t.get_fontsize() * SMALL_TEXT_BOOST)
    lg = ax.get_legend()
    if lg is not None:
        for t in lg.get_texts():
            t.set_fontsize(t.get_fontsize() * SMALL_TEXT_BOOST)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.015,
    # ASCII hyphen for negative ticks: matplotlib's U+2212 is emitted as a
    # separate tiny glyph span, which breaks legibility checks.
    "axes.unicode_minus": False,
})


def save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", FIGDIR / f"{name}.pdf")


# ---------------------------------------------------------------------------
# pushdown cross-check: 3 panels -> 2 stacked rows
# ---------------------------------------------------------------------------
def pushdown() -> None:
    data = json.loads((PKG / "data" / "nonlinear_pushdown.json")
                      .read_text(encoding="utf-8"))
    case = {c["case"]: c for c in data["cases"]}
    style = {
        "first_order_perfectly_plastic": ("#1f77b4", "-",
                                          "first order, perfectly plastic"),
        "second_order_perfectly_plastic": ("#d62728", "--",
                                           "P-Delta, perfectly plastic"),
        "second_order_degrading": ("#2ca02c", ":",
                                   "P-Delta, degrading hinges"),
    }

    fig = plt.figure(figsize=(PRINT_W_IN, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42,
                          wspace=0.24)

    def curve(ax, case_name, title):
        c = case[case_name]
        cert = c["certificate"]
        for row in c["runs"]:
            col, st, lab = style[row["model_class"]]
            pts = [h for h in row["history"] if h.get("ok")]
            if not pts:
                continue
            disp = np.abs(np.array([p["disp"] for p in pts], float))
            lam = np.array([p["lambda"] for p in pts], float)
            ax.plot(disp, lam, st, color=col, lw=1.6, label=lab)
            ax.plot(disp[-1], lam[-1], "o", ms=4, color=col, mfc="white",
                    mew=1.2)
        ax.axhline(cert["complete"], color="0.25", lw=1.0, ls="-.",
                   label=f"complete = {cert['complete']:.3f}")
        ax.axhline(cert["local"], color="0.55", lw=1.0, ls="--",
                   label=f"local = {cert['local']:.3f}")
        ax.set_xlabel("pushdown displacement at the cut node")
        ax.set_ylabel(r"pushdown multiplier $\lambda$")
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, loc="lower right", handlelength=1.8)

    curve(fig.add_subplot(gs[0, 0]), "witness_baseline_H2B2_removal_s1g2",
          r"(a) Exact case, margin $+0.935$")
    curve(fig.add_subplot(gs[0, 1]), "witness_memberwise_H2B2_removal_s1g2",
          r"(b) Inexact case, margin $-8.537$")

    rows = []
    for name in ("canonical_k1", "canonical_k2", "canonical_k5"):
        c = case[name]
        k = float(name.split("k")[1])
        first = next(r for r in c["runs"]
                     if r["model_class"] == "first_order_perfectly_plastic")
        rows.append((k, first["lambda_limit"], min(4.0 * k, 5.0)))
    rows.sort()
    ks = [r[0] for r in rows]

    ax = fig.add_subplot(gs[1, :])
    ax.plot(ks, [r[2] for r in rows], "s-", color="0.35", lw=1.8, ms=5,
            label=r"analytical $\min(4k,5)$")
    ax.plot(ks, [r[1] for r in rows], "o", color="#1f77b4", ms=7, mfc="white",
            mew=1.6, label="pushdown, first order")
    for k, value, target in rows:
        off = (8, -14) if k > 1.0 else (8, 4)
        ax.annotate(f"{value:.3f} ({(value / target - 1.0) * 100:+.1f}%)",
                    (k, value), textcoords="offset points", xytext=off,
                    fontsize=7.2, color="0.2")
    ax.set_xlabel("member multiplier $k$ of the two-story construction")
    ax.set_ylabel(r"collapse multiplier $\lambda$")
    ax.set_title("(c) Solver verification against the analytical boundary")
    ax.set_xticks(ks)
    ax.set_xlim(0.80, 5.45)
    ax.set_ylim(3.50, 5.35)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.8)

    save(fig, "fig_pushdown_crosscheck")


# ---------------------------------------------------------------------------
# public control: 2 panels -> 2 stacked rows
# ---------------------------------------------------------------------------
def public_control() -> None:
    data = json.loads((PKG / "data" / "public_paired_opensees.json")
                      .read_text(encoding="utf-8"))
    rows = data["rows"]

    fig = plt.figure(figsize=(PRINT_W_IN, 5.4))
    gs = fig.add_gridspec(2, 1, hspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    def censored(r):
        return r.get("intact_limit_reason") == "max_load_factor_reached"

    heights = sorted({r["stories"] for r in rows})
    cols = plt.cm.viridis(np.linspace(0.12, 0.82, len(heights)))
    for h, col in zip(heights, cols):
        sel = sorted([r for r in rows if r["stories"] == h],
                     key=lambda r: r["multiplier"])
        x = [r["multiplier"] for r in sel]
        y = [r["removed_limit_load_factor"] for r in sel]
        cs = [censored(r) for r in sel]
        ax0.plot(x, y, "-", color=col, lw=1.5, label=f"{h}-story, removed")
        for xi, yi, ci in zip(x, y, cs):
            ax0.plot(xi, yi, "o", ms=4.5, color=col,
                     mfc="white" if ci else col, mew=1.2)
        yi2 = [r["intact_limit_load_factor"] for r in sel]
        ax0.plot(x, yi2, "--", color=col, lw=1.1, alpha=0.75,
                 label=f"{h}-story, intact")
    ax0.set_xlabel("multiplier on the other main members, $k$")
    ax0.set_ylabel(r"limit load factor $\lambda$")
    ax0.set_title("(a) Source-traceable SAC/Elkady--Lignos control runs "
                  "(open marker: censored at the load ceiling)")
    ax0.grid(True, alpha=0.22)
    ax0.legend(frameon=False, ncol=3, handlelength=1.6)

    for h, col in zip(heights, cols):
        sel = sorted([r for r in rows if r["stories"] == h],
                     key=lambda r: r["multiplier"])
        xs = [r["multiplier"] for r in sel]
        ys = [r["removed_over_intact"] for r in sel]
        cs = [censored(r) for r in sel]
        good_x = [a for a, c in zip(xs, cs) if not c]
        good_y = [a for a, c in zip(ys, cs) if not c]
        ax1.plot(good_x, good_y, "o-", color=col, lw=1.5, ms=4.5,
                 label=f"{h}-story")
        for a, c in zip(xs, cs):
            if c:
                ax1.plot(a, 0.0, "v", ms=5, mfc="white", mec=col, mew=1.2)
    ax1.set_xlabel("multiplier on the other main members, $k$")
    ax1.set_ylabel("removed-to-intact ratio")
    ax1.set_title("(b) Valid paired ratios; open triangles mark a censored pair")
    ax1.grid(True, alpha=0.22)
    ax1.legend(frameon=False, ncol=3, handlelength=1.6)

    save(fig, "fig_public_paired_control")


# ---------------------------------------------------------------------------
# capacity path: 2 treatments -> 2 stacked panels, grouped bars
# ---------------------------------------------------------------------------
POS_ORDER = ["ground|exterior", "ground|interior_bay",
             "interior_story|exterior", "interior_story|interior_bay",
             "roof|exterior", "roof|interior_bay"]
POS_LABEL = {
    "ground|exterior": "ground\nexterior",
    "ground|interior_bay": "ground\ninterior bay",
    "interior_story|exterior": "interior\nexterior",
    "interior_story|interior_bay": "interior\ninterior bay",
    "roof|exterior": "roof\nexterior",
    "roof|interior_bay": "roof\ninterior bay",
}


def capacity_path() -> None:
    data = json.loads((PKG / "data" / "capacity_path_analysis.json")
                      .read_text(encoding="utf-8"))
    treat = data["treatments"]

    fig = plt.figure(figsize=(PRINT_W_IN, 4.9))
    gs = fig.add_gridspec(2, 1, hspace=0.60)
    specs = [
        ("componentwise_strengthening",
         "(a) Memberwise strengthening"),
        ("geometric_mean_one_redistribution",
         "(b) Bounded log-balanced redistribution"),
    ]
    for i, (key, title) in enumerate(specs):
        ax = fig.add_subplot(gs[i, 0])
        by = treat[key]["by_position_class"]
        keys = [k for k in POS_ORDER if k in by]
        vals = [by[k]["gap_fraction"] * 100.0 for k in keys]
        ns = [by[k]["eligible_instances"] for k in keys]
        x = np.arange(len(keys))
        colours = ["#B4472B" if "ground" in k else
                   ("#1F5F7A" if "interior" in k else "#2E8B6E") for k in keys]
        bars = ax.bar(x, vals, color=colours, width=0.62)
        for xi, v, n, bar in zip(x, vals, ns, bars):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.0)
            ax.text(bar.get_x() + bar.get_width() / 2, 0.4, f"n={n}",
                    ha="center", va="bottom", fontsize=6.4, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([POS_LABEL.get(k, k) for k in keys], fontsize=7.2)
        ax.set_ylabel("new-gap fraction (%)")
        ax.set_ylim(0, max(vals) * 1.28 + 0.5)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.22)

    save(fig, "fig_capacity_path")


# ---------------------------------------------------------------------------
# strengthening / transition figure: 2x2 -> taller 2x2 at print width
# ---------------------------------------------------------------------------
def transition() -> None:
    data = json.loads((PKG / "data" / "final_capacity_pattern_gate.json")
                      .read_text(encoding="utf-8"))
    # Three full-width rows: the 2x2 layout forced the colourbar of the
    # left-hand heat map under the right-hand panel, which collided with the
    # next row's axis labels.
    fig = plt.figure(figsize=(PRINT_W_IN, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.45, 1.0], hspace=0.52)

    Hs = [2, 4, 6, 8, 12]
    Bs = [1, 2, 3, 4]
    groups = data["groups"]

    # (a) analytical boundary
    ax = fig.add_subplot(gs[0, 0])
    ks = np.linspace(1.0, 6.0, 200)
    ax.plot(ks, 4 * ks, "-", color="#B4472B", lw=1.6, label="local, $4k$")
    ax.plot(ks, np.minimum(4 * ks, 5.0), "-", color="#1F5F7A", lw=1.6,
            label="complete, $\\min(4k,5)$")
    ax.axvline(1.25, color="#C9A227", lw=1.1)
    ax.annotate("break at $k=1.25$", xy=(1.25, 5.0), xytext=(1.75, 12.0),
                fontsize=8.0, color="#C9A227",
                arrowprops=dict(arrowstyle="-", color="#C9A227", lw=0.9))
    ax.set_xlabel("capacity multiplier $k$")
    ax.set_ylabel("first-order capacity")
    ax.set_title("(a) Analytical two-story boundary")
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    ax.grid(True, alpha=0.22)

    # (b) heat map: frame-level fraction exact at every removal, and the
    #     transition counts, over the frozen stratification grid.
    ax = fig.add_subplot(gs[1, 0])
    z = np.full((len(Hs), len(Bs)), np.nan)
    for i, H in enumerate(Hs):
        for j, B in enumerate(Bs):
            for key, val in groups.items():
                if f"H{H}|B{B}" in key:
                    base = val["baseline_clean_frames"]
                    frames = val["frames"]
                    z[i, j] = (100.0 * base / frames) if frames else np.nan
                    break
    im = ax.imshow(z, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(Bs)))
    ax.set_xticklabels([f"$B={b}$" for b in Bs])
    ax.set_yticks(range(len(Hs)))
    ax.set_yticklabels([f"$H={h}$" for h in Hs])
    ax.set_title("(b) Baseline-clean frames, % of the 12 generated draws "
                 "in each cell (empty cells are the $B=1$ stratum)")
    for i in range(len(Hs)):
        for j in range(len(Bs)):
            if not np.isnan(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.0f}", ha="center", va="center",
                        fontsize=9.0,
                        color="white" if z[i, j] > 55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("baseline-clean (%)")

    # (c) archived margin-sample diagnostic
    ax = fig.add_subplot(gs[2, 0])
    samples = data.get("margin_sample", {}).get("records", [])
    if samples:
        mg = np.array([s["margin"] for s in samples], float)
        gp = np.array([s["gap"] for s in samples], float)
        pos = mg >= 0
        ax.semilogy(mg[pos], np.maximum(gp[pos], 1e-10), "o", ms=3.0,
                    color="#1F5F7A", alpha=0.55, label="margin $\\geq 0$")
        ax.semilogy(mg[~pos], np.maximum(gp[~pos], 1e-10), "o", ms=3.0,
                    color="#B4472B", alpha=0.55, label="margin $< 0$")
        ax.axhline(1e-10, color="0.6", lw=0.9, ls=":")
        ax.text(-24.0, 2.2e-10, "plotting floor $10^{-10}$", fontsize=8.0,
                color="0.35")
    ax.axvline(0.0, color="#C9A227", lw=1.1)
    ax.set_xlabel("minimum interval margin")
    ax.set_ylabel("local minus complete")
    ax.set_title("(c) Archived 6,000-record margin-sample diagnostic: "
                 "sign of the margin against the realised gap")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax.grid(True, alpha=0.20, which="both")

    save(fig, "fig_strengthening_phase")


if __name__ == "__main__":
    pushdown()
    public_control()
    capacity_path()
    transition()
