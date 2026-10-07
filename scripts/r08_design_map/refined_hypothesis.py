"""Refined hypothesis: exactness depends on symmetry ACROSS the cut, not on
uniformity per story.

The symmetry attack showed that varying capacity BAY-WISE breaks exactness
immediately, while varying it STORY-WISE, skewing the load, making spans unequal
or changing storey heights never does. That points at a much more specific
condition than "uniform capacity per story":

    the two beam chord-rate anchors adjacent to the removed column line must
    match, i.e. the capacity on the left of the cut must equal the capacity on
    the right of the cut.

This script tests that hypothesis directly and separates the contributions:

  beam_left vs beam_right   the suspected controlling pair
  column_left vs column_right
  absolute magnitude        does the level matter, or only the ratio?
  position of asymmetry     one bay away vs adjacent to the cut
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


def build(H, B, span, height, beam, column, beam_factors, column_factors):
    Mp_b, Mp_c = Mp(beam), Mp(column)
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(B - 1, span), span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = Mp_b * beam_factors[g]
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            mc = Mp_c * column_factors[g]
            members.append(["c", r, g, mc, mc, 14.0 * mc, heights[r - 1]])
    f = Frame(f"refine_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="refined hypothesis")
    f.validate()
    return f


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
        if m < -1e-8:
            n_bad += 1
    return worst, worst_rem, n_bad, len(f.scenarios())


def main() -> None:
    H, B, span, height = 4, 2, 6.0, 4.0
    beam, column = "W18X40", "W14X109"
    rows = []

    def rec(tag, bf, cf):
        f = build(H, B, span, height, beam, column, bf, cf)
        worst, rem, n_bad, n = scan(f)
        rows.append({"tag": tag, "beam_factors": bf, "column_factors": cf,
                     "worst_margin": worst, "worst_removal": rem,
                     "n_inexact": n_bad, "n_removals": n})
        return n_bad, n, worst

    base_bf = [1.0, 1.0]
    base_cf = [1.0, 1.0, 1.0]

    print("=== A. beam capacity differing between the two bays ===")
    print(f"  {'beam factors':<18s} {'inexact':>10s} {'worst margin':>14s}")
    for ratio in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        bf = [ratio, 1.0]
        n_bad, n, worst = rec(f"beam_{ratio}", bf, base_cf)
        print(f"  {str(bf):<18s} {n_bad:>4d}/{n:<4d} {worst:>14.3f}")

    print("\n=== B. column capacity differing between the two lines ===")
    for ratio in (1.0, 1.5, 2.0, 3.0):
        cf = [ratio, 1.0, 1.0]
        n_bad, n, worst = rec(f"col_{ratio}", base_bf, cf)
        print(f"  {str(cf):<18s} {n_bad:>4d}/{n:<4d} {worst:>14.3f}")

    print("\n=== C. both sides raised equally (homogeneous in the cut sense) ===")
    for ratio in (1.0, 2.0, 5.0):
        bf = [ratio, ratio]
        n_bad, n, worst = rec(f"both_{ratio}", bf, base_cf)
        print(f"  {str(bf):<18s} {n_bad:>4d}/{n:<4d} {worst:>14.3f}")

    print("\n=== D. does the absolute level matter, or only the imbalance? ===")
    for lo, hi in ((1.0, 1.0), (2.0, 2.0), (5.0, 5.0),
                   (1.0, 1.25), (4.0, 5.0), (0.5, 0.625)):
        bf = [hi, lo]
        n_bad, n, worst = rec(f"level_{lo}_{hi}", bf, base_cf)
        imb = hi / lo
        print(f"  bf={str(bf):<14s} imbalance={imb:<5.3f} {n_bad:>3d}/{n:<3d} "
              f"worst={worst:>11.3f}")

    print("\n=== E. B=3: asymmetry adjacent to the cut vs one bay away ===")
    for tag, bf in (("uniform", [1.0, 1.0, 1.0]),
                    ("far bay raised", [2.0, 1.0, 1.0]),
                    ("middle bay raised", [1.0, 2.0, 1.0]),
                    ("right bay raised", [1.0, 1.0, 2.0])):
        f = build(4, 3, span, height, beam, column, bf, [1.0] * 4)
        worst, rem, n_bad, n = scan(f)
        rows.append({"tag": f"B3_{tag}", "beam_factors": bf,
                     "column_factors": [1.0] * 4, "worst_margin": worst,
                     "worst_removal": rem, "n_inexact": n_bad, "n_removals": n})
        print(f"  {tag:<18s} {n_bad:>3d}/{n:<3d} worst={worst:>11.3f} "
              f"at {rem}")

    dest = PKG / "data" / "refined_hypothesis.json"
    dest.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    print("\n=== CONTROLLING VARIABLE ===")
    sym = [r for r in rows if r["beam_factors"][0] == r["beam_factors"][1]
           or len(set(r["beam_factors"])) == 1]
    print(f"  designs with equal beam capacity on both cut-adjacent bays: "
          f"{len(sym)}; inexact among them: "
          f"{sum(1 for r in sym if r['n_inexact'])}")
    asym = [r for r in rows if len(set(r["beam_factors"])) > 1]
    print(f"  designs with unequal beam capacity between bays: {len(asym)}; "
          f"inexact among them: {sum(1 for r in asym if r['n_inexact'])}")


if __name__ == "__main__":
    main()
