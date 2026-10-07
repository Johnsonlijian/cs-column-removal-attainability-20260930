"""Robustness of the design-map pattern to (a) the gravity demand model and
(b) the story skeleton.

Stage 2 found that every single-bay (B=1) catalogue-section frame is inexact at
some removal, while multi-bay frames with uniform capacity per story are exact
at every removal. That was measured under one demand model. A reviewer will ask
whether it is an artefact of that model, so this script repeats the sweep under
four demand shapes and four story skeletons and reports the pattern.

Demand shapes (all keep every nodal load positive, which the model requires):
  ribbon      parenthesised tributary ribbon (the design model)
  uniform     equal load on every column line and floor
  top_heavy   load increasing linearly with height
  bottom_heavy load decreasing linearly with height

Skeletons: uniform, and linearly graded story heights / bay spans.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "scripts" / "code"))
sys.path.insert(0, str(HERE))

from engine import Frame, Scenario  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402
from real_section_design_map import SECTIONS, Mp  # noqa: E402

FY = 345.0
GRAVITY = 6.0
TRIB = 1.0
BETA = 1.15

PAIRS = [
    ("W18X40", "W14X109"),
    ("W21X50", "W14X132"),
    ("W24X62", "W14X176"),
    ("HN400X200", "HW350X350"),
]


def demands(H, B, span, shape):
    """Return nodal load matrix (H x B+1) and the beam moment demand D_b."""
    w = GRAVITY * TRIB
    D_b = BETA * w * span ** 2 / 8.0
    lines = np.r_[span / 2, np.full(max(B - 1, 0), span), span / 2]
    base = lines / span  # tributary share per column line, sums to B+1
    if shape == "ribbon":
        prof = np.ones(H)
    elif shape == "uniform":
        prof = np.ones(H)
        base = np.ones(B + 1)
    elif shape == "top_heavy":
        prof = np.linspace(0.4, 1.6, H)
    elif shape == "bottom_heavy":
        prof = np.linspace(1.6, 0.4, H)
    else:
        raise ValueError(shape)
    loads = np.outer(prof, base) * (w * span)
    return loads, D_b


def build(H, B, span, height, beam, column, shape, skeleton):
    Mp_b, Mp_c = Mp(beam), Mp(column)
    if skeleton == "uniform":
        spans = [span] * B
        heights = [height] * H
    elif skeleton == "graded":
        spans = list(np.linspace(0.8 * span, 1.2 * span, B))
        heights = list(np.linspace(1.2 * height, 0.8 * height, H))
    else:
        raise ValueError(skeleton)

    loads, D_b = demands(H, B, span, shape)
    P_col = 2.0 * (GRAVITY * TRIB * span / 2.0) * np.arange(H, 0, -1)
    D_c = P_col * height / 8.0

    members = []
    for r in range(1, H + 1):
        for g in range(B):
            members.append(["b", r, g, Mp_b, Mp_b, 8.0 * Mp_b, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    f = Frame(f"robust_{beam}_{column}_H{H}_B{B}", spans, heights, members,
              loads.tolist(),
              provenance=f"catalogue sections; demand={shape}; skeleton={skeleton}")
    f.validate()
    return f, D_b, float(np.mean(D_c))


def scan(f):
    worst = np.inf
    worst_rem = None
    n_bad = 0
    for removal in f.scenarios():
        sc = Scenario(f, removal)
        x = np.ones(f.E)
        m = float(min_interval_margin(sc, x))
        if m < worst:
            worst, worst_rem = m, list(removal)
        if m < -1.0e-8:
            n_bad += 1
    return {"worst_margin": worst, "worst_removal": worst_rem,
            "n_inexact": n_bad, "n_removals": len(f.scenarios()),
            "exact_all": n_bad == 0}


def main() -> None:
    rows = []
    for shape in ("ribbon", "uniform", "top_heavy", "bottom_heavy"):
        for skeleton in ("uniform", "graded"):
            for H in (2, 4, 8):
                for B in (1, 2, 3):
                    for beam, column in PAIRS:
                        f, D_b, D_c = build(H, B, 6.0, 4.0, beam, column,
                                            shape, skeleton)
                        res = scan(f)
                        res.update({
                            "demand_shape": shape, "skeleton": skeleton,
                            "H": H, "B": B, "beam": beam, "column": column,
                            "kappa": Mp(beam) / Mp(column),
                            "rho": D_c / (D_b + D_c),
                        })
                        rows.append(res)

    out = {"note": "robustness of the B=1/B>=2 exactness pattern",
           "rows": rows}
    dest = PKG / "data" / "real_section_robustness.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)

    print("\n=== exactness by bay count, pooled over demand shape / skeleton / "
          "H / section pair ===")
    for B in (1, 2, 3):
        sub = [r for r in rows if r["B"] == B]
        ok = sum(1 for r in sub if r["exact_all"])
        print(f"  B={B}: {ok}/{len(sub)} exact at every removal "
              f"({100.0 * ok / len(sub):.1f}%)")

    print("\n=== per demand shape ===")
    for shape in ("ribbon", "uniform", "top_heavy", "bottom_heavy"):
        for B in (1, 2, 3):
            sub = [r for r in rows if r["demand_shape"] == shape and r["B"] == B]
            ok = sum(1 for r in sub if r["exact_all"])
            print(f"  {shape:<13s} B={B}: {ok:2d}/{len(sub):2d} exact")

    print("\n=== per skeleton ===")
    for sk in ("uniform", "graded"):
        for B in (1, 2, 3):
            sub = [r for r in rows if r["skeleton"] == sk and r["B"] == B]
            ok = sum(1 for r in sub if r["exact_all"])
            print(f"  {sk:<8s} B={B}: {ok:2d}/{len(sub):2d} exact")

    print("\n=== any B>=2 frame that is inexact? ===")
    bad = [r for r in rows if r["B"] >= 2 and not r["exact_all"]]
    if not bad:
        print("  none")
    for r in bad[:20]:
        print(f"  {r['demand_shape']:<13s} {r['skeleton']:<8s} H={r['H']} B={r['B']} "
              f"{r['beam']}/{r['column']} kappa={r['kappa']:.3f} "
              f"n_inexact={r['n_inexact']}/{r['n_removals']} worst={r['worst_margin']:.2f}")


if __name__ == "__main__":
    main()
