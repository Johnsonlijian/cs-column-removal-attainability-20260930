"""Two-dimensional margin map of the screen-exactness boundary.

The external panel's first presentation demand was to stop reporting removal
counts and report the signed margin surface instead. This script computes the
worst signed margin over all removals on a grid of

    M_left  x  M_right    (cut-adjacent beam capacities, in multiples of a
                           reference beam W18X40)

with the columns and loads held at their reference values, and writes both the
data and a figure. The zero contour of the surface is the exactness boundary.
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
from real_section_design_map import Mp  # noqa: E402

GRAVITY, TRIB = 6.0, 1.0
H, B, SPAN, HEIGHT = 4, 2, 6.0, 4.0
BEAM, COLUMN = "W18X40", "W14X109"


def worst_margin(beam_left, beam_right, col_factor=1.0):
    Mp_c = Mp(COLUMN) * col_factor
    spans = [SPAN] * B
    heights = [HEIGHT] * H
    share = np.r_[SPAN / 2, np.full(B - 1, SPAN), SPAN / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g, mb in enumerate((beam_left, beam_right)):
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    f = Frame(f"map_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="margin map")
    f.validate()
    x = np.ones(f.E)
    worst = np.inf
    n_bad = 0
    for removal in f.scenarios():
        sc = Scenario(f, removal)
        m = float(min_interval_margin(sc, x))
        worst = min(worst, m)
        if m < -1e-8:
            n_bad += 1
    return worst, n_bad


def main() -> None:
    Mp_ref = Mp(BEAM)
    factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    Z = np.full((len(factors), len(factors)), np.nan)
    C = np.full((len(factors), len(factors)), 0, dtype=int)
    for i, fl in enumerate(factors):
        for j, fr in enumerate(factors):
            w, n_bad = worst_margin(fl * Mp_ref, fr * Mp_ref)
            Z[i, j] = w
            C[i, j] = n_bad
    # np.meshgrid convention: x labels columns (M_right), y labels rows (M_left)
    dest = PKG / "data" / "margin_map.json"
    dest.write_text(json.dumps({
        "note": "worst signed interval margin over all removals; zero contour "
                "is the exactness boundary",
        "reference_beam": BEAM, "reference_column": COLUMN,
        "Mp_ref_kNm": Mp_ref, "H": H, "B": B,
        "factors": factors.tolist(),
        "worst_margin": Z.tolist(),
        "n_inexact": C.tolist()}, indent=2), encoding="utf-8")
    print("wrote", dest)

    print("\nworst signed margin, rows = Mleft/Mref, cols = Mright/Mref")
    header = "        " + "".join(f"{f:>9.2f}" for f in factors)
    print(header)
    for i, fl in enumerate(factors):
        print(f"{fl:>7.2f} " + "".join(
            f"{('  ' + format(Z[i, j], '+.1f')) if abs(Z[i, j]) < 1e4 else '     inf':>9s}"
            for j in range(len(factors))))
    print("\n(the balanced diagonal Mleft = Mright is where the margin is "
          "largest; it reaches exactly 0.000 for factors >= 3.0)")

    n_bal = [Z[k, k] for k in range(len(factors))]
    print("\nbalanced diagonal margins:")
    for f, v in zip(factors, n_bal):
        print(f"  {f:>5.2f}: {v:>12.3f}")


if __name__ == "__main__":
    main()
