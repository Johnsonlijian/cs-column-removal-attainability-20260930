"""Test the sharp classification rule exhaustively and try to falsify it.

Measured on catalogue sections:
  B >= 2, uniform capacity per story  ->  every removal exact   (0/392)
  B = 1, ground-story removal         ->  always inexact        (24/24)
  B = 1, upper-story removal          ->  usually inexact       (64/88)

Before this can be stated as a result it must survive a much wider sweep than
the one that produced it: more heights, more bays, extreme capacity ratios,
extreme spans and heights, and both demand shapes. This script runs that sweep
and reports every counterexample it finds, with the case printed so it can be
examined.
"""
from __future__ import annotations

import itertools
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
from real_section_design_map import Mp  # noqa: E402

GRAVITY = 6.0
TRIB = 1.0
BETA = 1.15

# A deliberately wide ratio ladder, spanning weak beams to very strong beams.
BEAMS = ["W12X26", "W16X31", "W18X40", "W21X50", "W24X62", "W27X84", "W30X90"]
COLS = ["W14X74", "W14X109", "W14X211", "HW400X400"]


def build(H, B, span, height, beam, column, shape):
    Mp_b, Mp_c = Mp(beam), Mp(column)
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(max(B - 1, 0), span), span / 2] / span
    if shape == "ribbon":
        loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB * span)
    else:                                   # uniform nodal load
        loads = np.outer(np.ones(H), np.ones(B + 1)) * (GRAVITY * TRIB * span)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            members.append(["b", r, g, Mp_b, Mp_b, 8.0 * Mp_b, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    f = Frame(f"sharp_H{H}_B{B}", spans, heights, members, loads.tolist(),
              provenance="sharp-rule falsification sweep")
    f.validate()
    return f


def main() -> None:
    rows = []
    hs = (1, 2, 3, 4, 6, 8, 10, 12)
    bs = (1, 2, 3, 4, 5)
    shapes = ("ribbon", "uniform")
    for shape, H, B, beam, column in itertools.product(shapes, hs, bs, BEAMS, COLS):
        f = build(H, B, 6.0, 4.0, beam, column, shape)
        for removal in f.scenarios():
            s, g = removal
            sc = Scenario(f, removal)
            x = np.ones(f.E)
            margin = float(min_interval_margin(sc, x))
            rows.append({
                "shape": shape, "H": H, "B": B, "s": s, "g": g,
                "beam": beam, "column": column,
                "kappa": Mp(beam) / Mp(column),
                "inexact": margin < -1e-8, "margin": margin,
                "ground": s == 1, "exterior": g in (0, B),
            })
    dest = PKG / "data" / "sharp_rule_sweep.json"
    dest.write_text(json.dumps({"records": rows}, indent=2), encoding="utf-8")
    print("wrote", dest)
    n = len(rows)
    print(f"\nremovals tested: {n}")
    print(f"design points  : {n and len({(r['shape'],r['H'],r['B'],r['beam'],r['column']) for r in rows})}")

    print("\n=== CLAIM 1: B>=2 and uniform capacity -> every removal exact ===")
    sub = [r for r in rows if r["B"] >= 2]
    bad = [r for r in sub if r["inexact"]]
    print(f"  {len(sub)} removals, {len(bad)} counterexamples")
    for r in bad[:5]:
        print("   COUNTEREXAMPLE:", r)

    print("\n=== CLAIM 2: B=1 and ground removal -> inexact ===")
    sub = [r for r in rows if r["B"] == 1 and r["ground"]]
    bad = [r for r in sub if not r["inexact"]]
    print(f"  {len(sub)} removals, {len(bad)} counterexamples")
    for r in bad[:5]:
        print("   COUNTEREXAMPLE:", r)

    print("\n=== CLAIM 3: B=1, upper removal -> inexact ===")
    sub = [r for r in rows if r["B"] == 1 and not r["ground"]]
    bad = [r for r in sub if not r["inexact"]]
    print(f"  {len(sub)} removals, {len(bad)} counterexamples "
          f"({100.0*len(bad)/max(len(sub),1):.1f}% exact, so CLAIM 3 is FALSE)")

    print("\n=== summary table: inexact rate by (B, story) ===")
    print(f"  {'B':>3s} {'story':>7s} {'n':>6s} {'inexact':>8s} {'rate':>8s}")
    for B in sorted({r["B"] for r in rows}):
        for gr, lab in ((True, "ground"), (False, "upper")):
            sub = [r for r in rows if r["B"] == B and r["ground"] == gr]
            if not sub:
                continue
            k = sum(1 for r in sub if r["inexact"])
            print(f"  {B:>3d} {lab:>7s} {len(sub):>6d} {k:>8d} "
                  f"{100.0*k/len(sub):>7.1f}%")

    print("\n=== does kappa (beam/column ratio) matter at B=1? ===")
    sub = [r for r in rows if r["B"] == 1]
    inex = [r["kappa"] for r in sub if r["inexact"]]
    ex = [r["kappa"] for r in sub if not r["inexact"]]
    if inex and ex:
        print(f"  inexact kappa range: {min(inex):.3f} .. {max(inex):.3f}")
        print(f"  exact   kappa range: {min(ex):.3f} .. {max(ex):.3f}  (overlapping)")


if __name__ == "__main__":
    main()
