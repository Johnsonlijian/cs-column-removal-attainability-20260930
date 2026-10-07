"""Does the interior two-band case reduce to a single band?

Interior removals have two nonzero threshold bands, symmetrically placed at
(-1/l, 0) and (0, +1/l). The measured bandwise increments for the equal-span
case are IDENTICAL in the two bands, and both give the same interval minimum.
If that identity holds generally, the interior case reduces to one band exactly
as the exterior case does, and the certification cost halves again.

The identity is not obvious. Band q=1 (negative) and band q=0 (positive) use
opposite sign conventions, so a left/right capacity imbalance could in
principle favour one band over the other. This script tests the identity over a
wide sweep of capacities, spans, storeys and heights, and reports every case
where the two bandwise minima differ, together with which band wins and by how
much.
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
    f = Frame(f"red_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="two-band reduction test")
    f.validate()
    return f


def band_minima(sc: Scenario, x: np.ndarray):
    """(bandwise minimum, increment tuple) for each nonzero-width band."""
    H = sc.f.H
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
        out.append({"q": q, "t": float(t), "width": float(hi - lo),
                    "min_interval": float(np.min(m + exit_cost)),
                    "entry": entry.copy(), "interior": interior.copy(),
                    "exit": exit_cost.copy()})
    return out


def main() -> None:
    beam = Mp("W18X40")
    column = Mp("W14X109")
    rows = []
    same_min = same_inc = tot = 0
    worst_min_gap = 0.0
    worst_inc_gap = 0.0

    for B in (2, 3, 4):
        for H in (2, 3, 4, 6):
            for span_mode in ("equal", "unequal"):
                spans = ([6.0] * B if span_mode == "equal"
                         else list(np.linspace(4.0, 8.0, B)))
                heights = [4.0] * H
                for fl, fr, cf in itertools.product(
                        (0.5, 1.0, 2.0, 4.0), (0.5, 1.0, 2.0, 4.0),
                        (0.5, 1.0, 2.0)):
                    for g in range(1, B):
                        f = build(H, B, spans, heights, fl * beam,
                                  fr * beam, cf * column)
                        sc = Scenario(f, (1, g))
                        x = np.ones(f.E)
                        bands = band_minima(sc, x)
                        if len(bands) < 2:
                            continue
                        tot += 1
                        b0, b1 = bands[0], bands[-1]
                        dmin = abs(b0["min_interval"] - b1["min_interval"])
                        dinc = max(
                            float(np.max(np.abs(b0["entry"] - b1["entry"]))),
                            float(np.max(np.abs(b0["interior"] -
                                               b1["interior"]))),
                            float(np.max(np.abs(b0["exit"] - b1["exit"]))))
                        worst_min_gap = max(worst_min_gap, dmin)
                        worst_inc_gap = max(worst_inc_gap, dinc)
                        same_min += int(dmin <= 1e-8)
                        same_inc += int(dinc <= 1e-8)
                        rows.append({
                            "B": B, "H": H, "span_mode": span_mode, "g": g,
                            "left_factor": fl, "right_factor": fr,
                            "col_factor": cf,
                            "band_mins": [b["min_interval"] for b in bands],
                            "min_gap": dmin, "inc_gap": dinc})

    print("=== do the two interior bands give the same value? ===")
    print(f"  interior configurations tested        : {tot}")
    print(f"  bandwise minima identical             : {same_min}/{tot} "
          f"(worst gap {worst_min_gap:.3e})")
    print(f"  full increment vectors identical      : {same_inc}/{tot} "
          f"(worst gap {worst_inc_gap:.3e})")

    diff = [r for r in rows if r["min_gap"] > 1e-8]
    if diff:
        print(f"\n  cases where the bands DIFFER: {len(diff)}")
        for r in diff[:10]:
            print(f"    B={r['B']} H={r['H']} g={r['g']} "
                  f"L={r['left_factor']} R={r['right_factor']} "
                  f"C={r['col_factor']}  mins={r['band_mins']} "
                  f"gap={r['min_gap']:.4f}")
    else:
        print("\n  no case found in which the two bands differ")

    print("\n=== consequence for the margin ===")
    print("  if the identity holds, the interior margin equals the value of a "
          "single\n  band, so the two-band lemma is a presentational refinement "
          "rather than\n  an extra computational step.")

    dest = PKG / "data" / "two_band_reduction.json"
    dest.write_text(json.dumps({
        "interior_configurations": tot,
        "bandwise_minima_identical": same_min,
        "increment_vectors_identical": same_inc,
        "worst_min_gap": worst_min_gap, "worst_inc_gap": worst_inc_gap,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
