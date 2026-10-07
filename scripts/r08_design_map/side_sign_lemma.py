"""The side-sign lemma: is the interval margin monotone in the two cut beams?

A referee panel objected that the side-dependent monotonicity is supported only
by 864 capacity ladders and demanded the algebraic reason. This script tests the
underlying sign structure directly.

Reasoning to test
-----------------
Each joint contributes a minimum-cut weight

    psi = min( A0 + Cm*y + Cp*yp ,  A1 + Cm*(1-y) + Cp*(1-yp) ),

a concave, positively homogeneous function of the beam capacities, and the
story potentials V_r(a,b) are sums of these. The interval margin is a fixed
linear combination of the V's, so it is also concave in the capacities. The
reported margin is a minimum over intervals of these concave functions, which is
in general neither convex nor concave, so monotonicity has to be established
rather than assumed.

What is measured here
---------------------
For each removal and each interval, the slope of the interval margin with
respect to M_det and to M_int separately, obtained by finite differences on the
exact certificate. If the sign of the M_int slope is nonnegative for every
interval in every configuration, the intact-side monotonicity of the minimum
follows immediately. If the M_det slope can be positive, the detached-side
nonmonotonicity is confirmed rather than assumed.
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
from real_section_design_map import Mp  # noqa: E402

GRAVITY, TRIB = 6.0, 1.0


def build(H, B, spans, heights, M_det, M_int, C):
    loads = np.zeros((H, B + 1))
    for j in range(B + 1):
        left = spans[j - 1] / 2 if j > 0 else 0.0
        right = spans[j] / 2 if j < B else 0.0
        loads[:, j] = (left + right) * GRAVITY * TRIB
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = M_det if g == 0 else M_int
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, C, C, 14.0 * C, heights[r - 1]])
    f = Frame(f"sign_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="side-sign lemma")
    f.validate()
    return f


def interval_margins(H, B, spans, heights, M_det, M_int, C, removal):
    """Every interval margin (band, a, b) from the certificate's own increments."""
    f = build(H, B, spans, heights, M_det, M_int, C)
    sc = Scenario(f, removal)
    x = np.ones(f.E)
    out = []
    for lo, hi, t, q in sc.bands:
        if hi - lo <= 0:
            continue
        V, B0, _ = sc._potentials(x, t, q)
        entry = np.empty(H)
        entry[0] = B0
        if H > 1:
            entry[1:] = V[:-1, 0, 1] - V[:-1, 0, 0]
        interior = V[:, 1, 1] - V[:, 0, 0]
        exit_cost = V[:, 1, 0] - V[:, 0, 0]
        # reconstruct each individual interval value e_a + sum d + f_b
        for a in range(1, H + 1):
            acc = entry[a - 1]
            for b in range(a, H + 1):
                if b > a:
                    acc += interior[b - 2]
                out.append((q, a, b, acc + exit_cost[b - 1]))
    return out


def main() -> None:
    beam = Mp("W18X40")
    column = Mp("W14X109")
    h = 1e-6

    md_pos = md_neg = mi_pos = mi_neg = mi_zero = 0
    worst_mi_negative = 0.0
    worst_md_positive = 0.0
    worst_md_negative = 0.0
    rows = []

    for H, B, spans in ((2, 2, [6.0, 6.0]), (4, 2, [6.0, 6.0]),
                        (4, 3, [5.0, 6.0, 7.0]), (6, 2, [6.0, 6.0])):
        heights = [4.0] * H
        removals = [(1, 0), (1, B // 2), (H, 0)]
        for removal in removals:
            for fl, fr, cf in itertools.product((0.5, 1.0, 2.0),
                                                (0.5, 1.0, 2.0),
                                                (0.5, 1.0, 2.0)):
                base = interval_margins(H, B, spans, heights, fl * beam,
                                        fr * beam, cf * column, removal)
                d_det = interval_margins(H, B, spans, heights,
                                         (fl + h) * beam, fr * beam,
                                         cf * column, removal)
                d_int = interval_margins(H, B, spans, heights, fl * beam,
                                         (fr + h) * beam, cf * column, removal)
                for (q, a, b, v0), (_, _, _, v1), (_, _, _, v2) in zip(
                        base, d_det, d_int):
                    s_det = (v1 - v0) / (h * beam)
                    s_int = (v2 - v0) / (h * beam)
                    if s_det > 1e-9:
                        md_pos += 1
                        worst_md_positive = max(worst_md_positive, s_det)
                    elif s_det < -1e-9:
                        md_neg += 1
                        worst_md_negative = min(worst_md_negative, s_det)
                    if s_int > 1e-9:
                        mi_pos += 1
                    elif s_int < -1e-9:
                        mi_neg += 1
                        worst_mi_negative = min(worst_mi_negative, s_int)
                    else:
                        mi_zero += 1
                    rows.append({"H": H, "B": B, "removal": list(removal),
                                 "q": q, "a": a, "b": b,
                                 "slope_det": s_det, "slope_int": s_int})

    tot = len(rows)
    print(f"interval margins sampled: {tot}\n")
    print("=== slope with respect to the INTACT-side beam ===")
    print(f"  positive : {mi_pos}")
    print(f"  negative : {mi_neg}   (worst {worst_mi_negative:.3e})")
    print(f"  zero     : {mi_zero}")
    print("  -> the intact-side monotonicity of the minimum follows if the")
    print("     negative count is 0 or confined to inactive intervals")

    print("\n=== slope with respect to the DETACHED-side beam ===")
    print(f"  positive : {md_pos}   (worst +{worst_md_positive:.3e})")
    print(f"  negative : {md_neg}   (worst {worst_md_negative:.3e})")
    print("  -> a nonzero positive count means the interval form is genuinely")
    print("     two-sided in M_det, so the minimum need not be monotone")

    neg = [r for r in rows if r["slope_int"] < -1e-9]
    if neg:
        print(f"\n=== intervals with a negative intact-side slope "
              f"({len(neg)}) ===")
        for r in neg[:8]:
            print(f"   H={r['H']} B={r['B']} removal={r['removal']} "
                  f"q={r['q']} a={r['a']} b={r['b']} slope={r['slope_int']:.4f}")

    dest = PKG / "data" / "side_sign_lemma.json"
    dest.write_text(json.dumps({
        "intervals_sampled": tot,
        "intact_positive": mi_pos, "intact_negative": mi_neg,
        "intact_zero": mi_zero,
        "detached_positive": md_pos, "detached_negative": md_neg,
        "worst_intact_negative": worst_mi_negative,
        "worst_detached_positive": worst_md_positive}, indent=2),
        encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
