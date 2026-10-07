"""Which functional of the two cut-adjacent beam capacities controls exactness?

Candidate controls, all of which fit the data so far:
  R  ratio        M_left / M_right
  D  difference   M_left - M_right
  r  ratio to demand   (M_left - M_right) / (demand moment)
  A  absolute level    M_left

The distinguishing experiments are lines of constant R and lines of constant D:

  along constant R  ([1.25,1.0], [2.5,2.0], [5.0,4.0], [10,8]) the data show
                    exact, ?, inexact, ? -> if exactness depends on R alone the
                    whole line must behave the same, which it does not;
  along constant D  ([1.25,1.0], [2.25,2.0], [5.25,5.0], [10.25,10]) D = 0.25 for
                    all, so if D is the control the whole line must behave the
                    same.

This script runs both lines and a two-dimensional map, then reports which
functional best separates exact from inexact.
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


def build(beam_factors_num):
    """Beam capacities as explicit multipliers of the reference beam."""
    Mp_b, Mp_c = Mp(BEAM), Mp(COLUMN)
    spans = [SPAN] * B
    heights = [HEIGHT] * H
    share = np.r_[SPAN / 2, np.full(B - 1, SPAN), SPAN / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = Mp_b * beam_factors_num[g]
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    f = Frame(f"control_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="control-variable study")
    f.validate()
    return f


def scan(beam_factors_num):
    f = build(beam_factors_num)
    worst = np.inf
    worst_rem = None
    n_bad = 0
    for removal in f.scenarios():
        sc = Scenario(f, removal)
        x = np.ones(f.E)
        m = float(min_interval_margin(sc, x))
        if m < worst:
            worst, worst_rem = m, list(removal)
        if m < -1e-8:
            n_bad += 1
    return {"beam_factors": list(beam_factors_num), "worst_margin": worst,
            "n_inexact": n_bad, "n_removals": len(f.scenarios()),
            "exact_all": n_bad == 0, "worst_removal": worst_rem}


def main() -> None:
    Mp_ref = Mp(BEAM)
    # The demand moment on a floor beam in this model, in kN m.
    demand = 1.15 * (GRAVITY * TRIB * SPAN / 2) * SPAN / 8.0

    rows = []

    print("=== line of constant RATIO R = 1.25 ===")
    print(f"  {'factors':<16s} {'R':>6s} {'D/Mp_ref':>9s} {'D/demand':>9s} "
          f"{'inexact':>9s} {'worst':>11s}")
    for pair in ([1.25, 1.0], [2.5, 2.0], [5.0, 4.0], [10.0, 8.0]):
        r = scan(pair)
        R = pair[0] / pair[1]
        D = (pair[0] - pair[1]) * Mp_ref
        rows.append({**r, "R": R, "D_over_Mpref": D / Mp_ref,
                     "D_over_demand": D / demand, "line": "const_R"})
        print(f"  {str(pair):<16s} {R:>6.3f} {D/Mp_ref:>9.4f} "
              f"{D/demand:>9.4f} {r['n_inexact']:>4d}/{r['n_removals']:<4d} "
              f"{r['worst_margin']:>11.3f}")

    print("\n=== line of constant DIFFERENCE D = 0.25 Mp_ref ===")
    print(f"  {'factors':<16s} {'R':>6s} {'D/Mp_ref':>9s} {'D/demand':>9s} "
          f"{'inexact':>9s} {'worst':>11s}")
    for pair in ([1.25, 1.0], [2.25, 2.0], [5.25, 5.0], [10.25, 10.0]):
        r = scan(pair)
        R = pair[0] / pair[1]
        D = (pair[0] - pair[1]) * Mp_ref
        rows.append({**r, "R": R, "D_over_Mpref": D / Mp_ref,
                     "D_over_demand": D / demand, "line": "const_D"})
        print(f"  {str(pair):<16s} {R:>6.3f} {D/Mp_ref:>9.4f} "
              f"{D/demand:>9.4f} {r['n_inexact']:>4d}/{r['n_removals']:<4d} "
              f"{r['worst_margin']:>11.3f}")

    print("\n=== two-dimensional map: left factor x right factor ===")
    lefts = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    rights = [1.0, 2.0, 4.0]
    header = "L vs R".rjust(7)
    print("  " + header + "".join(f"{rr:>10.2f}" for rr in rights))
    for L in lefts:
        cells = []
        for Rr in rights:
            r = scan([L, Rr])
            rows.append({**r, "R": L / Rr,
                         "D_over_Mpref": (L - Rr),
                         "D_over_demand": (L - Rr) * Mp_ref / demand,
                         "line": "map"})
            cells.append(f"{r['n_inexact']:>4d}/{r['n_removals']:<4d}")
        print(f"  {L:>7.2f}" + "".join(f"{c:>10s}" for c in cells))

    dest = PKG / "data" / "control_variable_study.json"
    dest.write_text(json.dumps({
        "reference_beam": BEAM, "reference_column": COLUMN,
        "Mp_ref_kNm": Mp_ref, "beam_demand_kNm": demand,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    exact = [r for r in rows if r["exact_all"]]
    inex = [r for r in rows if not r["exact_all"]]
    print("\n=== which functional separates them? ===")
    for name, key in (("ratio R", lambda r: r["R"]),
                      ("difference / Mp_ref", lambda r: r["D_over_Mpref"]),
                      ("difference / demand", lambda r: r["D_over_demand"])):
        ev = [key(r) for r in exact]
        iv = [key(r) for r in inex]
        lo_i, hi_i = min(iv), max(iv)
        print(f"  {name:<22s} exact range [{min(ev):.4f}, {max(ev):.4f}] | "
              f"inexact range [{lo_i:.4f}, {hi_i:.4f}] -> "
              f"{'OVERLAP (not controlling)' if max(ev) > lo_i else 'SEPARATES'}")


if __name__ == "__main__":
    main()
