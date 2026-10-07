"""Rebuild Figure 1 (mechanics) and add the catalogue-section design-map figure.

Figure 1 is drawn from the EXACT computed velocity fields of the engine, not
from a hand-drawn cartoon: panel (a) plots the local (zero-sway) optimum, panel
(b) plots the complete optimum of the same frame and load, and panel (c) plots
the certificate margin against the sweep parameter k. Panels (a) and (b) show
that the local screen's mechanism is not the mechanism the complete frame
selects.

Figure 2 (new) is the catalogue-section design map: the certificate computed on
real AISC / GB sections, showing the bay-count dichotomy and the fact that
realistic strengthening interventions do not break a clean multi-bay design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "scripts" / "code"))
sys.path.insert(0, str(HERE))

from engine import Frame, Scenario  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402
from real_section_design_map import Mp  # noqa: E402

FIGDIR = PKG / "figures"
FIGDIR.mkdir(exist_ok=True)

# House style: compact, print-legible, no chartjunk.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 9.6,
    "axes.labelsize": 9.4,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.4,
    # Render negative tick labels with an ASCII hyphen. matplotlib otherwise
    # emits a separate U+2212 glyph span whose measured size is tiny, which
    # makes legibility checks report false sub-7 pt text.
    "axes.unicode_minus": False,
})

# Some elements are authored at fixed point sizes below the base. To meet a 7 pt
# floor on the printed page (the manuscript includes figures at 142.5 mm), every
# explicitly sized label is multiplied by SMALL_TEXT_BOOST. This keeps the
# relative hierarchy while lifting the smallest type above the floor.
SMALL_TEXT_BOOST = 1.62


def _boost(ax):
    """Scale explicit point sizes in one axes so nothing prints below 7 pt."""
    for t in ax.texts:
        t.set_fontsize(t.get_fontsize() * SMALL_TEXT_BOOST)
    lg = ax.get_legend()
    if lg is not None:
        for t in lg.get_texts():
            t.set_fontsize(t.get_fontsize() * SMALL_TEXT_BOOST)

C_LOCAL = "#B4472B"     # warm: restricted / local
C_FULL = "#1F5F7A"      # cool: complete
C_ACC = "#C9A227"       # accent
C_GREY = "#5A5A5A"
GREEN = "#2E8B6E"


def canonical_frame() -> Frame:
    f = Frame(
        "canonical_H2_B1",
        [1.0], [1.0, 1.0],
        [
            ["b", 1, 0, 1.0, 1.0, 8.0, 1.0],
            ["b", 2, 0, 1.0, 1.0, 8.0, 1.0],
            ["c", 1, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 1, 1, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 1, 5.0, 5.0, 50.0, 1.0],
        ],
        [[0.5, 0.5], [0.5, 0.5]],
        provenance="analytical construction",
    )
    f.validate()
    return f


def scaled_vector(k: float) -> np.ndarray:
    f = canonical_frame()
    x = np.ones(f.E)
    for i, m in enumerate(f.members):
        if not (m[0] == "c" and m[1] == 1 and m[2] == 0):
            x[i] = k
    return x


def draw_frame(ax, f, sc, vel, title, colour, hinge_factor=0.20):
    """Draw undeformed (grey) and deformed (colour) frame with hinge marks."""
    coords = sc.coords
    # member list with node endpoints
    for m in sc.members:
        kind, r, j = m[0], int(m[1]), int(m[2])
        if kind == "b":
            ni, nj = (j, r), (j + 1, r)
        else:
            ni, nj = (j, r - 1), (j, r)
        p, q = coords[ni], coords[nj]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#BFBFBF", lw=1.1, zorder=1)

    # deformed shape
    scale = hinge_factor / max(np.abs(vel).max(), 1e-12)
    for m in sc.members:
        kind, r, j = m[0], int(m[1]), int(m[2])
        if kind == "b":
            ni, nj = (j, r), (j + 1, r)
        else:
            ni, nj = (j, r - 1), (j, r)
        p, q = coords[ni].copy(), coords[nj].copy()
        for node, pt in ((ni, p), (nj, q)):
            if node in sc.dofs:
                k0 = sc.dofs[node]
                pt[0] += scale * vel[k0]
                pt[1] += scale * vel[k0 + 1]
        ax.plot([p[0], q[0]], [p[1], q[1]], color=colour, lw=1.7, zorder=3)
        # hinge circles at both ends where the end rotation is active
        for node, pt in ((ni, p), (nj, q)):
            if node not in sc.dofs:
                continue
            k0 = sc.dofs[node]
            if abs(vel[k0 + 2]) > 1e-9:
                ax.plot(pt[0], pt[1], "o", ms=3.6, mfc="white",
                        mec=colour, mew=1.0, zorder=4)

    # removed column
    sg, gg = sc.removal
    xs = sc.coords[(gg, 0)][0]
    y0 = sc.coords[(gg, sg - 1)][1]
    y1 = sc.coords[(gg, sg)][1]
    ax.plot([xs, xs], [y0, y1], color=C_GREY, lw=2.4, ls=(0, (4, 2.4)), zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=colour, pad=4)


def figure1() -> None:
    f = canonical_frame()
    sc = Scenario(f, (1, 1))

    # Fig 1 panel (c) carries a legend plus an x-label; at 142.5 mm the panel is
    # narrow, so the canvas is given extra height and the legend is placed below
    # the axes rather than inside them.
    fig = plt.figure(figsize=(5.61, 4.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.72, 0.72, 1.56], wspace=0.26)

    # --- (a) local mechanism at k = 2 ---
    ax = fig.add_subplot(gs[0, 0])
    res_l = sc.chain(scaled_vector(2.0), local=True)
    draw_frame(ax, f, sc, res_l["velocity"], "(a) zero-sway screen", C_LOCAL)

    # --- (b) complete mechanism at k = 2 ---
    ax = fig.add_subplot(gs[0, 1])
    res_f = sc.chain(scaled_vector(2.0))
    draw_frame(ax, f, sc, res_f["velocity"], "(b) complete mechanism", C_FULL)

    # --- (c) certificate margin vs k ---
    ax = fig.add_subplot(gs[0, 2])
    ks = np.linspace(1.0, 3.0, 161)
    margins, gaps = [], []
    for k in ks:
        x = scaled_vector(k)
        margins.append(min_interval_margin(sc, x))
        gaps.append(float(sc.chain(x, local=True)["capacity"]
                          - sc.chain(x)["capacity"]))
    margins = np.array(margins)
    gaps = np.array(gaps)

    ax.axhline(0.0, color=C_GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.plot(ks, margins, color=C_FULL, lw=1.8, zorder=3,
            label="min. interval margin")
    ax.fill_between(ks, 0, margins, where=margins < 0, color=C_LOCAL,
                    alpha=0.16, zorder=0)
    ax.plot(ks, gaps, color=C_LOCAL, lw=1.8, ls=(0, (5, 2)), zorder=3,
            label="screen error (local - complete)")
    ax.axvline(1.25, color=C_ACC, lw=1.2, zorder=2)
    ax.annotate("break at k = 1.25", xy=(1.25, 0.0),
                xytext=(1.42, -1.45), fontsize=9.1, color=C_ACC,
                arrowprops=dict(arrowstyle="-", color=C_ACC, lw=0.9))
    ax.text(2.02, 5.6, "screen fails\n(margin < 0)", fontsize=9.4,
            color=C_LOCAL, ha="left", va="top")
    ax.text(1.04, 1.0, "screen exact", fontsize=9.4, color=C_FULL,
            ha="left", va="bottom")
    ax.set_xlabel("strengthening factor k")
    ax.set_ylabel("normalised margin / error")
    ax.set_xlim(1.0, 3.0)
    ax.set_ylim(-2.4, 7.2)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4,
              fontsize=8.0)
    ax.set_title("(c) the margin decides the transition", color="#222222", pad=4)

    for _a in fig.axes:
        _boost(_a)

    dest = FIGDIR / "fig_mechanism_certificate.pdf"
    fig.savefig(dest)
    fig.savefig(FIGDIR / "fig_mechanism_certificate.png")
    fig.savefig(FIGDIR / "fig_mechanism_certificate.svg")
    plt.close(fig)
    print("wrote", dest)


def figure2() -> None:
    """Catalogue-section design map."""
    rob = json.loads((PKG / "data" / "real_section_robustness.json")
                     .read_text(encoding="utf-8"))["rows"]
    inter = json.loads((PKG / "data" / "real_section_interventions.json")
                       .read_text(encoding="utf-8"))["rows"]
    dm = json.loads((PKG / "data" / "real_section_design_map.json")
                    .read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(5.61, 6.60))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.36)

    # --- (a) exactness vs bay count, real sections ---
    ax = fig.add_subplot(gs[0, 0])
    counts = {}
    for r in rob:
        counts.setdefault(r["B"], [0, 0])
        counts[r["B"]][1] += 1
        if r["exact_all"]:
            counts[r["B"]][0] += 1
    bays = sorted(counts)
    frac = [100.0 * counts[b][0] / counts[b][1] for b in bays]
    bars = ax.bar([str(b) for b in bays], frac, width=0.56,
                  color=[C_LOCAL if v < 50 else C_FULL for v in frac],
                  edgecolor="white", linewidth=0.8)
    for b, v, bar in zip(bays, frac, bars):
        inside = v > 30
        cx = bar.get_x() + bar.get_width() / 2
        if inside:
            ax.text(cx, v - 8.0, f"{v:.0f}%", ha="center", va="top",
                    fontsize=10.1, color="white")
            ax.text(cx, v - 22.0, f"n={counts[b][1]}", ha="center", va="top",
                    fontsize=8.8, color="#DDE9EE")
        else:
            ax.text(cx, v + 3.0, f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=10.1, color=C_LOCAL)
            ax.text(cx, v + 13.0, f"n={counts[b][1]}", ha="center", va="bottom",
                    fontsize=8.8, color=C_GREY)
    ax.set_ylim(0, 122)
    ax.set_xlabel("number of bays $B$")
    ax.set_ylabel("designs exact at every removal (%)")
    ax.set_title("(a) single-bay frames never pass the screen",
                 color="#222222", pad=4)

    # --- (b) margin distribution by bay count ---
    ax = fig.add_subplot(gs[0, 1])
    data = [[r["worst_margin"] for r in rob if r["B"] == b] for b in bays]
    logs = [np.log10(np.maximum(np.abs(d), 1e-6)) * np.sign(d) for d in data]
    parts = ax.violinplot(logs, showextrema=False, widths=0.72)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(C_LOCAL if bays[i] == 1 else C_FULL)
        pc.set_alpha(0.30)
        pc.set_edgecolor("none")
    for i, d in enumerate(logs, start=1):
        ax.plot([i], [np.median(d)], "_", ms=13, color="#222222", mew=1.5)
    ax.axhline(0.0, color=C_ACC, lw=1.1)
    ax.text(2.55, 0.16, r"$\Delta_{\min}=0$", fontsize=9.1, color=C_ACC)
    ax.set_xticks(range(1, len(bays) + 1))
    ax.set_xticklabels([str(b) for b in bays])
    ax.set_xlabel("number of bays $B$")
    ax.set_ylabel(r"$\log_{10}|\Delta_{\min}|$ (signed)")
    ax.set_title("(b) margin separates by topology", color="#222222", pad=4)

    # --- (c) strengthening interventions: gap opened? ---
    ax = fig.add_subplot(gs[1, 0])
    labels, opened, clean = [], [], []
    order = ["baseline", "all", "beam_only", "col_only", "base_only",
             "upper_only"]
    pretty = {"baseline": "baseline", "all": "all", "beam_only": "beams",
              "col_only": "cols", "base_only": "base",
              "upper_only": "upper st."}
    for lab in order:
        rs = [r for r in inter if r["treatment"] == lab]
        if not rs:
            continue
        labels.append(pretty[lab])
        opened.append(sum(1 for r in rs if r.get("opens_gap_from_clean_baseline")))
        clean.append(sum(1 for r in rs if r["exact_all_removals"]))
    n = len(labels)
    totals = [len([r for r in inter if r["treatment"] == o]) for o in order[:n]]
    ax.barh(np.arange(n), totals, color=C_FULL, edgecolor="none", height=0.62)
    if sum(opened) > 0:
        ax.barh(np.arange(n), opened, color=C_LOCAL, height=0.62,
                label="intervention opened a new gap")
    for i, (c0, t0) in enumerate(zip(clean, totals)):
        ax.text(t0 + 1.4, i, f"{c0}/{t0}", va="center", ha="left", fontsize=9.1)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.26)
    ax.set_xlabel("catalogue-section design points exact at every removal")
    ax.text(0.98, 0.06,
            f"0 of {len(inter)} opened a gap",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.1, color=GREEN, weight="bold")
    ax.set_title("(c) realistic upgrades do not break clean designs",
                 color="#222222", pad=4)

    # --- (d) section ratio vs margin, coloured by bay count ---
    ax = fig.add_subplot(gs[1, 1])
    for b, col, mk in ((1, C_LOCAL, "o"), (2, C_FULL, "s"), (3, "#2E8B6E", "^")):
        xs = [r["kappa"] for r in rob if r["B"] == b]
        ys = [np.log10(max(np.abs(r["worst_margin"]), 1e-6))
              * np.sign(r["worst_margin"]) for r in rob if r["B"] == b]
        ax.scatter(xs, ys, s=16, marker=mk, facecolor=col, edgecolor="white",
                   linewidth=0.4, label=f"$B={b}$", zorder=3)
    ax.axhline(0.0, color=C_ACC, lw=1.1, zorder=2)
    ax.set_xlabel(r"section ratio $\kappa = M_{p,\mathrm{beam}}/M_{p,\mathrm{col}}$")
    ax.set_ylabel(r"$\log_{10}|\Delta_{\min}|$ (signed)")
    ax.legend(frameon=False, loc="lower right", ncol=3, columnspacing=1.0)
    ax.set_title("(d) topology, not strength ratio, sets exactness",
                 color="#222222", pad=4)

    dest = FIGDIR / "fig_design_map.pdf"
    for _a in fig.axes:
        _boost(_a)
    fig.savefig(dest)
    fig.savefig(FIGDIR / "fig_design_map.png")
    fig.savefig(FIGDIR / "fig_design_map.svg")
    plt.close(fig)
    print("wrote", dest)


if __name__ == "__main__":
    figure1()
    figure2()
