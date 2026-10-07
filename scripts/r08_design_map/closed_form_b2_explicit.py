"""Explicit interval form of the B = 2 margin, verified against the engine.

What is established
-------------------
For the two-bay frame the metric has two nonzero threshold bands carrying the
same raw interval margin, so the margin is a single minimisation over contiguous
storey intervals. The increments of that chain have closed form:

    B0            = M_L + 2 M_R + 2 C
    entry[1]      = M_L + M_R + C
    entry[a]      = M_L + C                  for a >= 2
    interior[H-1] = 2 M_R - M_L              (ground storey increment)
    interior[r]   = M_R - M_L                for 1 <= r < H-1
    interior[H]   = -M_L                     (roof increment, no column above)
    exit[b]       = entry[b+1]               for b < H
    exit[H]       = M_R - M_L                (roof exit)

Hence the margin is the explicit finite minimum

    margin = min over 1 <= b <= H of [ m_b + exit[b] ],
    m_1 = entry[1],   m_b = min(m_{b-1} + interior[b-1], entry[b]),

which is O(H) and contains no iteration to convergence, no linear program and no
chain solve. What this script verifies is precisely that statement: that the
increments above reproduce the engine's own increments, and that the resulting
minimum equals the engine's reported margin.

What is NOT established
-----------------------
A single closed-form expression in (M_L, M_R, C) that removes the minimisation
over b. Two candidates were tested and failed: M_L + M_R + C (correct only for
M_R <= M_L) and 2 M_R + 2 C (correct only for M_R > M_L). The reason is visible
in the increments: interior[H-1] = 2 M_R - M_L differs from the other interior
increments, so the optimum can leave at the ground storey or one storey higher
depending on how M_L compares with 2 M_R, and which one wins depends on H as
well. The minimisation is therefore kept explicit rather than guessed.
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

GRAVITY, TRIB = 6.0, 1.0


def build(H, span, height, beam_left, beam_right, col):
    B = 2
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, span, span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = beam_left if g == 0 else beam_right
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, col, col, 14.0 * col, heights[r - 1]])
    f = Frame(f"ex_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="explicit interval form")
    f.validate()
    return f


def increments_closed_form(H, M_L, M_R, C):
    """The increments, in closed form, as stated in the module docstring."""
    B0 = M_L + 2.0 * M_R + 2.0 * C
    entry = np.empty(H)
    entry[0] = M_L + M_R + C
    if H > 1:
        entry[1:] = M_L + C
    interior = np.empty(H)
    interior[:H - 1] = M_R - M_L
    interior[H - 1] = -M_L
    # the ground-storey increment is special
    if H >= 1:
        interior[0] = 2.0 * M_R - M_L
    exit_cost = np.empty(H)
    exit_cost[:H - 1] = entry[1:]
    exit_cost[H - 1] = M_R - M_L
    return B0, entry, interior, exit_cost


def margin_from_increments(H, M_L, M_R, C):
    B0, entry, interior, exit_cost = increments_closed_form(H, M_L, M_R, C)
    m = np.empty(H)
    m[0] = entry[0]
    for b in range(1, H):
        m[b] = min(m[b - 1] + interior[b - 1], entry[b])
    return float(np.min(m + exit_cost))


def main() -> None:
    Mp_ref = Mp("W18X40")
    Mp_col = Mp("W14X109")
    rows = []
    ok = tot = 0
    worst = 0.0
    inc_ok = inc_tot = 0
    inc_worst = 0.0
    for H in (2, 3, 4, 6, 8, 12):
        for fl in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            for fr in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
                for cf in (0.25, 0.5, 1.0, 2.0):
                    M_L, M_R, C = fl * Mp_ref, fr * Mp_ref, cf * Mp_col
                    f = build(H, 6.0, 4.0, M_L, M_R, C)
                    sc = Scenario(f, (1, 0))
                    x = np.ones(f.E)
                    engine = float(min_interval_margin(sc, x))
                    pred = margin_from_increments(H, M_L, M_R, C)
                    res = abs(engine - pred)
                    worst = max(worst, res)
                    tot += 1
                    ok += int(res <= 1e-6 * max(1.0, abs(engine)))

                    # increment-by-increment check against the engine, using the
                    # first nonzero-width band (both nonzero bands agree here)
                    band = next(bd for bd in sc.bands if bd[1] - bd[0] > 0)
                    V, B0e, _ = sc._potentials(x, band[2], band[3])
                    B0c, entry_c, interior_c, exit_c = \
                        increments_closed_form(H, M_L, M_R, C)
                    d_inc = max(
                        abs(B0e - B0c),
                        float(np.max(np.abs(entry_c[1:] -
                                            (V[:-1, 0, 1] - V[:-1, 0, 0])))),
                        float(np.max(np.abs(interior_c -
                                            (V[:, 1, 1] - V[:, 0, 0])))),
                        float(np.max(np.abs(exit_c -
                                            (V[:, 1, 0] - V[:, 0, 0])))))
                    inc_worst = max(inc_worst, d_inc)
                    inc_tot += 1
                    inc_ok += int(d_inc <= 1e-6)
                    rows.append({"H": H, "M_L": M_L, "M_R": M_R, "C": C,
                                 "engine": engine, "explicit": pred,
                                 "abs_residual": res,
                                 "increment_residual": d_inc})

    print(f"configurations checked                     : {tot}")
    print(f"closed-form increments match the engine     : {inc_ok}/{inc_tot}"
          f"  (worst residual {inc_worst:.3e})")
    print(f"explicit interval minimum matches the engine: {ok}/{tot}"
          f"  (worst residual {worst:.3e})")

    print("\n=== what the increments show about the two failed formulas ===")
    H = 4
    examples = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (5.0, 4.0), (0.5, 1.0)]
    print(f"  {'M_L':>8s} {'M_R':>8s} {'entry2':>9s} {'int0':>9s} "
          f"{'int1':>9s} {'exit2':>9s} {'margin':>10s}")
    for fl, fr in examples:
        M_L, M_R = fl * Mp_ref, fr * Mp_ref
        B0, e, it, ex = increments_closed_form(H, M_L, M_R, Mp_col)
        mg = margin_from_increments(H, M_L, M_R, Mp_col)
        print(f"  {M_L:>8.2f} {M_R:>8.2f} {e[1]:>9.2f} {it[0]:>9.2f} "
              f"{it[1]:>9.2f} {ex[1]:>9.2f} {mg:>10.3f}")

    dest = PKG / "data" / "closed_form_b2_explicit.json"
    dest.write_text(json.dumps({
        "statement": "B=2 margin is an explicit O(H) minimum over storey "
                     "intervals with closed-form increments; no reduced "
                     "single-expression form in (M_L,M_R,C) was found",
        "checked": tot, "agree": ok,
        "worst_residual": worst,
        "increment_checks": inc_tot, "increment_agreements": inc_ok,
        "increment_worst_residual": inc_worst,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
