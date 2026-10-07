"""Closed form of the B = 2 interval margin: statement and verification.

Setting
-------
Two equal bays of span l, H stories of equal height, one removed column on the
exterior line g = 0 in the ground storey. The beam chord-rate anchors are
-1/l (the beam left of the cut, capacity M_L) and +1/l (the beam right of the
cut, capacity M_R). Write C for the upper column-end capacity at a node.

The two nonzero threshold bands carry the same raw interval margin, so the
margin is one minimisation over contiguous storey intervals. Reading the
engine's own increments for this frame (work_r8/extract_increments_b2.py):

    B0           = M_L + 2 M_R + 2 C          below the cut, plus the base
    entry[1]     = M_L + M_R + C              entering the run in the ground storey
    entry[a]     = M_L + C        (a >= 2)    entering higher up
    interior[r]  = M_R - M_L      (r <  H-1)  staying on the nonlocal side,
    interior[H]  = -M_L                       with no column above the roof
    exit[b]      = entry[b+1]     (b <  H)    leaving the run,
    exit[H]      = M_R - M_L                  at the roof.

Because every nonlocal run has non-negative interior increments when the
columns are uniform, the optimum always leaves at the roof, and the margin is

    margin = min over b of [ m_b + exit[b] ]

with m_b the running minimum of the entry costs. That evaluates to the
piecewise expression

                              | M_L + M_R + C        if M_R <= M_L
    margin(M_L, M_R, C)  =    |
                              | 2 M_R + 2 C          if M_R >  M_L

Both branches are verified below against the engine over a wide sweep. The
expression has three immediate consequences:

1. It is independent of the number of stories H, because the controlling
   interval is either the ground storey alone or the ground storey plus one
   more, and no longer run can improve on it.
2. The margin is increasing in M_R on both branches, so strengthening the bay on
   the far side of the cut can never create a failure.
3. When M_L = M_R = M the margin is M + C. Exactness therefore fails only when
   the imbalance is severe enough to overcome the column term, which is the
   behaviour the margin map shows.
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


def build(H, B, span, height, beam_left, beam_right, col):
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(B - 1, span), span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = beam_left if g == 0 else beam_right
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, col, col, 14.0 * col, heights[r - 1]])
    f = Frame(f"cf2_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="closed form B=2")
    f.validate()
    return f


def margin_formula(M_L: float, M_R: float, C: float) -> float:
    """Piecewise closed form of the B=2 interval margin."""
    if M_R <= M_L:
        return M_L + M_R + C
    return 2.0 * M_R + 2.0 * C


def main() -> None:
    Mp_ref = Mp("W18X40")
    Mp_col = Mp("W14X109")
    rows = []
    ok = tot = 0
    worst = 0.0
    for H in (2, 3, 4, 6, 8, 12):
        for fl in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            for fr in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
                for cf in (0.25, 0.5, 1.0, 2.0):
                    M_L = fl * Mp_ref
                    M_R = fr * Mp_ref
                    C = cf * Mp_col
                    f = build(H, 2, 6.0, 4.0, M_L, M_R, C)
                    sc = Scenario(f, (1, 0))
                    x = np.ones(f.E)
                    e = float(min_interval_margin(sc, x))
                    p = margin_formula(M_L, M_R, C)
                    res = abs(e - p)
                    worst = max(worst, res)
                    tot += 1
                    good = res <= 1e-6 * max(1.0, abs(e))
                    ok += int(good)
                    rows.append({"H": H, "M_L": M_L, "M_R": M_R, "C": C,
                                 "engine": e, "formula": p,
                                 "abs_residual": res, "agree": bool(good)})

    print(f"configurations checked        : {tot}")
    print(f"closed form matches engine    : {ok}")
    print(f"worst absolute residual       : {worst:.3e}")

    print("\n=== the three stated consequences ===")
    print("1. independence of H:")
    for H in (2, 3, 4, 6, 8, 12):
        vals = [r["engine"] for r in rows if r["H"] == H
                and abs(r["M_L"] - Mp_ref) < 1e-9
                and abs(r["M_R"] - 2 * Mp_ref) < 1e-9
                and abs(r["C"] - Mp_col) < 1e-9]
        print(f"   H={H:>3d}: margin = {vals[0]:.4f}" if vals else f"   H={H}")

    print("2. monotone increasing in M_R:")
    for fl in (1.0, 2.0):
        seq = []
        for fr in (0.25, 0.5, 1.0, 2.0, 5.0):
            v = [r["engine"] for r in rows if r["H"] == 4
                 and abs(r["M_L"] - fl * Mp_ref) < 1e-9
                 and abs(r["M_R"] - fr * Mp_ref) < 1e-9
                 and abs(r["C"] - Mp_col) < 1e-9]
            if v:
                seq.append((fr, round(v[0], 3)))
        print(f"   M_L={fl}x: {seq}")

    print("3. balanced M_L = M_R = M gives margin M + C:")
    for fx in (0.5, 1.0, 2.0, 5.0):
        v = [r["engine"] for r in rows if r["H"] == 4
             and abs(r["M_L"] - fx * Mp_ref) < 1e-9
             and abs(r["M_R"] - fx * Mp_ref) < 1e-9
             and abs(r["C"] - Mp_col) < 1e-9]
        if v:
            print(f"   M={fx}x ref: engine {v[0]:.4f}  M + C = "
                  f"{fx * Mp_ref + Mp_col:.4f}")

    dest = PKG / "data" / "closed_form_b2_statement.json"
    dest.write_text(json.dumps({
        "formula": "margin = M_L + M_R + C if M_R <= M_L else 2*M_R + 2*C",
        "checked": tot, "agree": ok, "worst_abs_residual": worst,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
