"""Mirror relation between the two interior bands, and the dual-band form.

The two-band reduction test showed that the interior bands carry DIFFERENT
values in 2050 of 2304 configurations, so both are needed. It also showed a
clean pattern: swapping the two cut-adjacent capacities swaps the two band
values exactly.

    L=0.5 R=1.0  ->  [ 430.905, -11.040]
    L=1.0 R=0.5  ->  [ -11.040, 430.905]

If that identity holds exactly, the interior margin has a precise form:

    margin = min( V(L, R), V(R, L) )

where V is a single band functional. The exterior margin is the corresponding
single value. This script verifies the mirror identity and then tests whether
the band functional V itself reduces to a closed form in the two capacities and
the column term, including whether a simple max/min expression fits.
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


def build(H, B, spans, heights, beam_left, beam_right, column):
    loads = np.zeros((H, B + 1))
    for j in range(B + 1):
        left = spans[j - 1] / 2 if j > 0 else 0.0
        right = spans[j] / 2 if j < B else 0.0
        loads[:, j] = (left + right) * GRAVITY * TRIB
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = beam_left if g == 0 else beam_right
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, column, column, 14.0 * column,
                            heights[r - 1]])
    f = Frame(f"mir_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="mirror identity test")
    f.validate()
    return f


def band_values(H, B, spans, heights, M_L, M_R, C, g):
    """Both bandwise minima for the interior removal (1, g)."""
    f = build(H, B, spans, heights, M_L, M_R, C)
    sc = Scenario(f, (1, g))
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
        m = np.empty(H)
        m[0] = entry[0]
        for b in range(1, H):
            m[b] = min(m[b - 1] + interior[b - 1], entry[b])
        out.append(float(np.min(m + exit_cost)))
    # negative band first, positive band last
    return out[0], out[-1], float(min_interval_margin(sc, x))


def main() -> None:
    beam = Mp("W18X40")
    column = Mp("W14X109")
    rows = []
    mirror_ok = mirror_tot = 0
    worst_mirror = 0.0
    min_ok = min_tot = 0
    worst_min = 0.0

    print("=== mirror identity: band values swap when L and R swap ===")
    print(f"  {'L':>6s} {'R':>6s} {'bands(L,R)':>26s} {'bands(R,L)':>26s} "
          f"{'ok':>4s}")
    for H, B, spans in ((2, 2, [6.0, 6.0]), (4, 2, [6.0, 6.0]),
                        (4, 3, [5.0, 6.0, 7.0])):
        heights = [4.0] * H
        mids = sorted({1, B // 2})
        for g in mids:
            for fl, fr in ((0.5, 1.0), (1.0, 0.5), (2.0, 1.0),
                           (1.0, 2.0), (3.0, 1.0), (1.0, 3.0)):
                a_neg, a_pos, a_margin = band_values(
                    H, B, spans, heights, fl * beam, fr * beam, column, g)
                b_neg, b_pos, b_margin = band_values(
                    H, B, spans, heights, fr * beam, fl * beam, column, g)
                err = max(abs(a_neg - b_pos), abs(a_pos - b_neg))
                worst_mirror = max(worst_mirror, err)
                mirror_tot += 1
                ok = err <= 1e-6 * max(1.0, abs(a_neg))
                mirror_ok += int(ok)
                if H == 2 and B == 2:
                    print(f"  {fl:>6.2f} {fr:>6.2f} "
                          f"{f'[{a_neg:.3f}, {a_pos:.3f}]':>26s} "
                          f"{f'[{b_neg:.3f}, {b_pos:.3f}]':>26s} "
                          f"{str(ok):>4s}")
                # margin must be the min of the two bands
                pred = min(a_neg, a_pos)
                e2 = abs(pred - a_margin)
                worst_min = max(worst_min, e2)
                min_tot += 1
                min_ok += int(e2 <= 1e-6 * max(1.0, abs(pred)))
                rows.append({"H": H, "B": B, "g": g, "L": fl, "R": fr,
                             "band_neg": a_neg, "band_pos": a_pos,
                             "margin": a_margin,
                             "mirror_error": err, "min_error": e2})

    print(f"\n  mirror identity holds: {mirror_ok}/{mirror_tot} "
          f"(worst error {worst_mirror:.3e})")
    print(f"  margin equals min of the two bands: {min_ok}/{min_tot} "
          f"(worst error {worst_min:.3e})")

    print("\n=== candidate closed forms for the band functional V(L,R,C) ===")
    cands = {
        "L + R + C": lambda L, R, C: L + R + C,
        "2*L + C": lambda L, R, C: 2 * L + C,
        "2*max(L,R) + 2*C": lambda L, R, C: 2 * max(L, R) + 2 * C,
        "max(L,R) + 2*C": lambda L, R, C: max(L, R) + 2 * C,
        "L + 2*R + C": lambda L, R, C: L + 2 * R + C,
    }
    for name, fn in cands.items():
        ok = tot = 0
        res = []
        for r in rows:
            for band, tag in ((r["band_neg"], "neg"), (r["band_pos"], "pos")):
                pass
        # test against the MARGIN, which is min of the two bands
        for r in rows:
            L, R, C = r["L"] * beam, r["R"] * beam, column
            pred = fn(L, R, C)
            tot += 1
            e = abs(pred - r["margin"])
            ok += int(e <= 1e-6 * max(1.0, abs(r["margin"])))
            res.append(round(e, 2))
        print(f"  {name:<20s} {ok}/{tot}   sample residuals {res[:4]}")

    dest = PKG / "data" / "interior_mirror.json"
    dest.write_text(json.dumps({
        "mirror_checks": mirror_tot, "mirror_pass": mirror_ok,
        "worst_mirror_error": worst_mirror,
        "min_checks": min_tot, "min_pass": min_ok,
        "worst_min_error": worst_min, "rows": rows}, indent=2),
        encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
