"""Verified monotonicity of the B = 2 margin in the two cut-adjacent beams.

The dependence probe shows a one-sided structure that is sharper than the
"cut-adjacent imbalance" statement currently in the manuscript:

  raising M_L, the beam on the DETACHED side of the cut, lowers the margin and
  can break the screen;

  raising M_R, the beam on the intact side, raises the margin and can never
  break it.

This script verifies that over the full grid and extracts the two monotone
envelopes, so the design statement can be made exact rather than qualitative.
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


def margin(H, span, height, M_L, M_R, C):
    B = 2
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, span, span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = M_L if g == 0 else M_R
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, C, C, 14.0 * C, heights[r - 1]])
    f = Frame(f"mono_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="monotonicity verification")
    f.validate()
    sc = Scenario(f, (1, 0))
    return float(min_interval_margin(sc, np.ones(f.E)))


def main() -> None:
    Mp_ref = Mp("W18X40")
    Mp_col = Mp("W14X109")
    rows = []
    v_ML_ok = v_MR_ok = 0
    v_ML_tot = v_MR_tot = 0
    worst_ML = worst_MR = 0.0

    LADDER = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
    for H in (2, 3, 4, 6, 8, 12):
        for span in (4.0, 6.0, 9.0):
            for height in (3.0, 4.0, 5.0):
                for cf in (0.25, 0.5, 1.0, 2.0):
                    C = cf * Mp_col
                    for other in (0.5, 1.0, 2.0, 5.0):
                        # sweep M_L upward at fixed M_R
                        seq = [margin(H, span, height, f * Mp_ref,
                                      other * Mp_ref, C) for f in LADDER]
                        mono_down = all(seq[i] >= seq[i + 1] - 1e-9
                                        for i in range(len(seq) - 1))
                        v_ML_tot += 1
                        v_ML_ok += int(mono_down)
                        worst_ML = max(worst_ML,
                                       max(seq[i] - seq[i + 1]
                                           for i in range(len(seq) - 1)))
                        # sweep M_R upward at fixed M_L
                        seq2 = [margin(H, span, height, other * Mp_ref,
                                       f * Mp_ref, C) for f in LADDER]
                        mono_up = all(seq2[i] <= seq2[i + 1] + 1e-9
                                      for i in range(len(seq2) - 1))
                        v_MR_tot += 1
                        v_MR_ok += int(mono_up)
                        worst_MR = max(worst_MR,
                                       max(seq2[i] - seq2[i + 1]
                                           for i in range(len(seq2) - 1)))
                        rows.append({
                            "H": H, "span": span, "height": height,
                            "C": C, "other_factor": other,
                            "ML_ladder_margins": seq,
                            "MR_ladder_margins": seq2,
                            "ML_monotone_decreasing": bool(mono_down),
                            "MR_monotone_increasing": bool(mono_up)})

    print("=== monotonicity of the B=2 margin ===")
    print(f"  raising M_L (detached-side beam): monotone decreasing in "
          f"{v_ML_ok}/{v_ML_tot} ladders   worst violation {worst_ML:.3e}")
    print(f"  raising M_R (intact-side beam)  : monotone increasing in "
          f"{v_MR_ok}/{v_MR_tot} ladders   worst violation {worst_MR:.3e}")

    print("\n=== consequence: exactness can be broken by M_L alone ===")
    broken = 0
    total = 0
    for r in rows:
        seq = r["ML_ladder_margins"]
        total += 1
        if seq[0] >= 0 and seq[-1] < 0:
            broken += 1
    print(f"  in {broken}/{total} ladders a screen that is exact at low M_L "
          f"becomes inexact at high M_L with everything else fixed")

    print("\n=== consequence: raising M_R never breaks an exact screen ===")
    worse = 0
    for r in rows:
        seq = r["MR_ladder_margins"]
        if seq[0] >= 0 and seq[-1] < 0:
            worse += 1
    print(f"  ladders where raising M_R turned exact into inexact: {worse}")

    print("\n=== worked example, H=4, span 6, height 4, reference column ===")
    print(f"  {'M_L factor':>10s} {'margin':>12s}   (M_R held at 1.0x ref)")
    for f in LADDER:
        m = margin(4, 6.0, 4.0, f * Mp_ref, Mp_ref, Mp_col)
        print(f"  {f:>10.2f} {m:>12.3f}")
    print(f"\n  {'M_R factor':>10s} {'margin':>12s}   (M_L held at 1.0x ref)")
    for f in LADDER:
        m = margin(4, 6.0, 4.0, Mp_ref, f * Mp_ref, Mp_col)
        print(f"  {f:>10.2f} {m:>12.3f}")

    dest = PKG / "data" / "b2_monotonicity.json"
    dest.write_text(json.dumps({
        "statement": "raising the beam on the detached side of the cut lowers "
                     "the margin; raising the beam on the intact side raises it",
        "ML_ladders": v_ML_tot, "ML_monotone": v_ML_ok,
        "ML_worst_violation": worst_ML,
        "MR_ladders": v_MR_tot, "MR_monotone": v_MR_ok,
        "MR_worst_violation": worst_MR,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
