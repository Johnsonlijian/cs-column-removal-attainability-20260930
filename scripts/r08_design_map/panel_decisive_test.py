"""Decisive test: is the exactness boundary controlled by cut imbalance, or is
imbalance aliased with beam-to-column relative capacity?

The external panel's objection
------------------------------
Comparing [1.25, 1.0] with [5.0, 4.0] holds the beam ratio fixed at 1.25 but
also raises total beam capacity by 4x while the columns and loads stay fixed.
So the beam-to-column capacity ratio changes too, and "beam imbalance" may be
aliased with "relative beam-column capacity". The panel asked for the signed
margin on a grid of

    s = M_L + M_R          total adjacent beam capacity
    rho = M_L / M_R        cut imbalance
    c = column multiplier

with the signed Delta_min recorded, not just an exact/inexact flag.

Reading the result
------------------
- If the boundary depends on s at fixed c and fixed rho, then ratio alone is
  falsified.
- If scaling beams AND columns together leaves the verdict unchanged, then the
  controlling group is relative beam-to-column capacity.
- A dimensionless group such as (M_L - M_R) / M_col, or s / M_col, is the
  natural candidate.
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


def build(beam_left, beam_right, col_factor):
    """Explicit capacities: the two cut-adjacent bays plus a column multiplier."""
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
    f = Frame(f"grid_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="panel decisive test")
    f.validate()
    return f


def signed_margin(beam_left, beam_right, col_factor):
    """Signed minimum margin over all removals, plus how many are inexact."""
    f = build(beam_left, beam_right, col_factor)
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
    Mp_col = Mp(COLUMN)
    demand = 1.15 * (GRAVITY * TRIB * SPAN / 2) * SPAN / 8.0
    rows = []

    print("=== GRID 1: total adjacent beam capacity s, at fixed ratio rho = 2.0 "
          "and fixed column ===")
    print(f"  {'s/Mp_ref':>9s} {'M_L/Mp_ref':>11s} {'M_R/Mp_ref':>11s} "
          f"{'s/Mcol':>8s} {'worst margin':>13s} {'inexact':>8s}")
    for s in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0):
        M_L = s * 2.0 / 3.0
        M_R = s * 1.0 / 3.0
        worst, n_bad = signed_margin(M_L * Mp_ref, M_R * Mp_ref, 1.0)
        rows.append({"grid": 1, "s_over_Mpref": s, "M_L_over_Mpref": M_L,
                     "M_R_over_Mpref": M_R, "s_over_Mcol": s * Mp_ref / Mp_col,
                     "worst_margin": worst, "n_inexact": n_bad,
                     "rho": M_L / M_R, "col_factor": 1.0})
        print(f"  {s:>9.2f} {M_L:>11.3f} {M_R:>11.3f} "
              f"{s * Mp_ref / Mp_col:>8.3f} {worst:>13.3f} {n_bad:>4d}/12")

    print("\n=== GRID 2: TIGHTENING RATIO at fixed total s = 5 (does rho alone "
          "control it?) ===")
    print(f"  {'rho':>6s} {'M_L/Mp_ref':>11s} {'M_R/Mp_ref':>11s} "
          f"{'worst margin':>13s} {'inexact':>8s}")
    for rho in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        M_R = 5.0 / (rho + 1.0)
        M_L = 5.0 - M_R
        worst, n_bad = signed_margin(M_L * Mp_ref, M_R * Mp_ref, 1.0)
        rows.append({"grid": 2, "s_over_Mpref": 5.0, "M_L_over_Mpref": M_L,
                     "M_R_over_Mpref": M_R, "s_over_Mcol": 5.0 * Mp_ref / Mp_col,
                     "worst_margin": worst, "n_inexact": n_bad, "rho": rho,
                     "col_factor": 1.0})
        print(f"  {rho:>6.2f} {M_L:>11.3f} {M_R:>11.3f} {worst:>13.3f} "
              f"{n_bad:>4d}/12")

    print("\n=== GRID 3: scale beams AND columns TOGETHER (fixed rho = 2, "
          "fixed s/Mcol) ===")
    print(f"  {'col factor':>10s} {'M_L/Mp_ref':>11s} {'M_R/Mp_ref':>11s} "
          f"{'worst margin':>13s} {'inexact':>8s}")
    for cf in (0.5, 1.0, 2.0, 4.0):
        M_L = 2.0 / 3.0 * 2.0 * cf
        M_R = 1.0 / 3.0 * 2.0 * cf
        worst, n_bad = signed_margin(M_L * Mp_ref, M_R * Mp_ref, cf)
        rows.append({"grid": 3, "s_over_Mpref": 2.0 * cf,
                     "M_L_over_Mpref": M_L, "M_R_over_Mpref": M_R,
                     "s_over_Mcol": 2.0 * Mp_ref / Mp_col,
                     "worst_margin": worst, "n_inexact": n_bad, "rho": 2.0,
                     "col_factor": cf})
        print(f"  {cf:>10.2f} {M_L:>11.3f} {M_R:>11.3f} {worst:>13.3f} "
              f"{n_bad:>4d}/12")

    print("\n=== GRID 4: uniform capacities, margin against level "
          "(the [5,5] = 0.000 question) ===")
    print(f"  {'factor':>8s} {'s/Mcol':>8s} {'worst margin':>14s} {'inexact':>8s}")
    for cf in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0):
        M = cf
        worst, n_bad = signed_margin(M * Mp_ref, M * Mp_ref, 1.0)
        rows.append({"grid": 4, "uniform_factor": cf, "s_over_Mpref": 2 * M,
                     "s_over_Mcol": 2 * M * Mp_ref / Mp_col,
                     "worst_margin": worst, "n_inexact": n_bad, "rho": 1.0,
                     "col_factor": 1.0})
        print(f"  {cf:>8.2f} {2 * M * Mp_ref / Mp_col:>8.3f} "
              f"{worst:>14.3f} {n_bad:>4d}/12")

    dest = PKG / "data" / "panel_decisive_test.json"
    dest.write_text(json.dumps({
        "reference_beam": BEAM, "reference_column": COLUMN,
        "Mp_beam_kNm": Mp_ref, "Mp_column_kNm": Mp_col,
        "beam_demand_kNm": demand, "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    print("\n=== VERDICTS ===")
    g1 = [r for r in rows if r["grid"] == 1]
    print("Grid 1 (fixed rho = 2, rising total s):")
    for r in g1:
        print(f"   s/Mp_ref={r['s_over_Mpref']:>5.2f}  worst={r['worst_margin']:>10.3f}"
              f"  inexact={r['n_inexact']}")
    monotone = all(g1[i]["worst_margin"] >= g1[i + 1]["worst_margin"]
                   for i in range(len(g1) - 1))
    print(f"   margin decreases monotonically as total beam capacity rises "
          f"(fixed ratio): {monotone}")

    g3 = [r for r in rows if r["grid"] == 3]
    same = len({r["n_inexact"] for r in g3}) == 1
    print(f"\nGrid 3 (beams and columns scaled together at fixed s/Mcol): "
          f"verdict unchanged = {same}")
    for r in g3:
        print(f"   col_factor={r['col_factor']:>5.2f}  s/Mcol={r['s_over_Mcol']:.3f}"
              f"  worst={r['worst_margin']:>10.3f}  inexact={r['n_inexact']}")

    g4 = [r for r in rows if r["grid"] == 4]
    zeros = [r for r in g4 if abs(r["worst_margin"]) < 1e-6]
    print(f"\nGrid 4 (uniform): exact at every level tested = "
          f"{all(r['n_inexact'] == 0 for r in g4)}; "
          f"exact zero margins at {[r['uniform_factor'] for r in zeros]}")


if __name__ == "__main__":
    main()
