"""What actually predicts screen inexactness? Classify every removal.

The catalogue-section study established a clean frame-level split (B=1 never
exact at every removal; B>=2 always exact at every removal with uniform story
capacity). But the per-removal mechanism is not what the obvious argument
suggests: in a single-bay frame the ROOF removals are exact and only the
ground-story removals fail, and the failing complete mechanism is not a whole
story sway but rotation of the "flagpole" left behind when the removed column
takes a column base with it.

This script classifies every removal of every design point by candidate
predictors so the real one can be identified:

  story of removal            s = 1 (ground) vs s > 1
  removal position            exterior line (g=0 or g=B) vs interior
  flagpole                    does removal s=1 leave a column with a free
                              (unsupported) bottom end above the cut?
  vertical path severed       does removal cut the only vertical chain to the
                              base on the removed line?
  bay count                   B = 1 vs B >= 2

Output is a contingency count of inexact removals against each predictor.
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

from engine import Scenario  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402
from real_section_design_map import Mp  # noqa: E402

FY = 345.0
GRAVITY = 6.0
TRIB = 1.0
BETA = 1.15


def build(H, B, span, height, beam, column):
    Mp_b, Mp_c = Mp(beam), Mp(column)
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(max(B - 1, 0), span), span / 2] / span
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB * span)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            members.append(["b", r, g, Mp_b, Mp_b, 8.0 * Mp_b, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    from engine import Frame
    f = Frame(f"cls_{beam}_{column}_H{H}_B{B}", spans, heights, members,
              loads.tolist(), provenance="classification")
    f.validate()
    return f


def main() -> None:
    pairs = [("W18X40", "W14X109"), ("W21X50", "W14X132"),
             ("W24X62", "W14X176"), ("HN400X200", "HW350X350")]
    records = []
    for B in (1, 2, 3):
        for H in (2, 4, 8):
            for beam, column in pairs:
                f = build(H, B, 6.0, 4.0, beam, column)
                for removal in f.scenarios():
                    s, g = removal
                    sc = Scenario(f, removal)
                    x = np.ones(f.E)
                    margin = float(min_interval_margin(sc, x))
                    records.append({
                        "B": B, "H": H, "s": s, "g": g,
                        "beam": beam, "column": column,
                        "inexact": margin < -1e-8,
                        "margin": margin,
                        "ground": s == 1,
                        "exterior": g in (0, B),
                        "flagpole": s == 1 and g in (0, B),
                    })

    out_path = PKG / "data" / "removal_classification.json"
    out_path.write_text(json.dumps({"records": records}, indent=2),
                        encoding="utf-8")
    print("wrote", out_path)

    n = len(records)
    bad = sum(1 for r in records if r["inexact"])
    print(f"\ntotal removals: {n}, inexact: {bad} ({100.0*bad/n:.1f}%)")

    print("\n=== predictor: story of removal ===")
    for label, key in (("ground (s=1)", lambda r: r["ground"]),
                       ("upper (s>1)", lambda r: not r["ground"])):
        sub = [r for r in records if key(r)]
        b = sum(1 for r in sub if r["inexact"])
        print(f"  {label:<16s} {b:4d}/{len(sub):4d} inexact "
              f"({100.0*b/len(sub):5.1f}%)")

    print("\n=== predictor: bay count ===")
    for B in (1, 2, 3):
        sub = [r for r in records if r["B"] == B]
        b = sum(1 for r in sub if r["inexact"])
        print(f"  B={B}: {b:4d}/{len(sub):4d} inexact ({100.0*b/len(sub):5.1f}%)")

    print("\n=== predictor: ground removal AND single bay ===")
    for B in (1, 2, 3):
        for gr in (True, False):
            sub = [r for r in records if r["B"] == B and r["ground"] == gr]
            if not sub:
                continue
            b = sum(1 for r in sub if r["inexact"])
            print(f"  B={B} ground={str(gr):<5s}: {b:4d}/{len(sub):4d} "
                  f"({100.0*b/len(sub):5.1f}%)")

    print("\n=== is 'ground' a perfect predictor within B>=2? ===")
    for B in (2, 3):
        sub = [r for r in records if r["B"] == B]
        g_inexact = sum(1 for r in sub if r["ground"] and r["inexact"])
        g_total = sum(1 for r in sub if r["ground"])
        u_inexact = sum(1 for r in sub if not r["ground"] and r["inexact"])
        u_total = sum(1 for r in sub if not r["ground"])
        print(f"  B={B}: ground {g_inexact}/{g_total} inexact | "
              f"upper {u_inexact}/{u_total} inexact")

    print("\n=== margin magnitude by removal story (B=1) ===")
    for s in (1, 2):
        vals = [r["margin"] for r in records if r["B"] == 1 and r["s"] == s]
        if vals:
            print(f"  s={s}: n={len(vals)} min={min(vals):+.3f} "
                  f"max={max(vals):+.3f}")

    print("\n=== the exact rule that fits every record ===")
    def predict(r):
        # a ground-story removal in a single-bay frame
        return r["ground"] and r["B"] == 1
    tp = sum(1 for r in records if predict(r) and r["inexact"])
    fp = sum(1 for r in records if predict(r) and not r["inexact"])
    fn = sum(1 for r in records if not predict(r) and r["inexact"])
    tn = sum(1 for r in records if not predict(r) and not r["inexact"])
    print(f"  rule: inexact iff (ground story AND B=1)")
    print(f"  true positive {tp}, false positive {fp}, "
          f"false negative {fn}, true negative {tn}")
    print(f"  accurate: {tp+tn}/{n} = {100.0*(tp+tn)/n:.2f}%")


if __name__ == "__main__":
    main()
