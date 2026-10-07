"""Does the B = 2 margin depend only on the beam LEFT of the cut?

The anchor dump shows that for the removal (1,0) the surviving beams have
anchors +1/l (the beam left of the cut, whose far end is on the detached column
line) and 0 (the beam right of the cut, whose far end is on an intact line).
There is therefore exactly ONE nonzero threshold band, of width 1/l, and not two
as the earlier reasoning assumed.

That changes the structure decisively: with a single band the margin is a single
interval minimum, and the beam to the RIGHT of the cut enters only through the
interior increments. This script measures the dependence directly:

  vary M_R at fixed M_L, H, C   -> does the margin move at all?
  vary M_L at fixed M_R, H, C   -> does the margin move?
  vary H                        -> what is the H dependence?

If the margin is a function of M_L (and H) alone, then the observed
"imbalance" effect in the design map is really an effect of the beam on the
detached side, and M_R acts only as a modifier. That would replace the earlier
"cut-adjacent imbalance" statement with a sharper and more useful one.
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
    f = Frame(f"dep_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="dependence probe")
    f.validate()
    return f


def margin(H, M_L, M_R, C):
    f = build(H, 6.0, 4.0, M_L, M_R, C)
    sc = Scenario(f, (1, 0))
    x = np.ones(f.E)
    band = next(bd for bd in sc.bands if bd[1] - bd[0] > 0)
    V, B0, _ = sc._potentials(x, band[2], band[3])
    interior = V[:, 1, 1] - V[:, 0, 0]
    entry = np.empty(H)
    entry[0] = B0
    entry[1:] = V[:-1, 0, 1] - V[:-1, 0, 0]
    exit_cost = V[:, 1, 0] - V[:, 0, 0]
    m = np.empty(H)
    m[0] = entry[0]
    for b in range(1, H):
        m[b] = min(m[b - 1] + interior[b - 1], entry[b])
    # structural quantities, useful for reading the formula
    return (float(min_interval_margin(sc, x)), float(B0), entry.tolist(),
            interior.tolist(), exit_cost.tolist(), f.E)


def main() -> None:
    Mp_ref = Mp("W18X40")
    Mp_col = Mp("W14X109")
    rows = []
    H = 4

    print("=== vary M_R at fixed M_L, C, H ===")
    print(f"  {'M_L':>9s} {'M_R':>9s} {'margin':>12s} {'interior[0..2]':>28s}")
    for MLf in (0.5, 1.0, 1.5, 2.0, 5.0):
        for MRf in (0.5, 1.0, 2.0, 3.0, 5.0):
            mg, B0, e, it, ex, E = margin(H, MLf * Mp_ref, MRf * Mp_ref, Mp_col)
            rows.append({"H": H, "M_L_factor": MLf, "M_R_factor": MRf,
                         "margin": mg, "B0": B0, "entry": e,
                         "interior": it, "exit": ex})
            print(f"  {MLf*Mp_ref:>9.2f} {MRf*Mp_ref:>9.2f} {mg:>12.3f} "
                  f"{[round(v,2) for v in it[:3]]!s:>28s}")

    print("\n=== margin as a function of M_L at fixed M_R (does ordering matter?) ===")
    print(f"  {'M_L factor':>10s} {'M_R=0.5':>12s} {'M_R=1.0':>12s} "
          f"{'M_R=2.0':>12s} {'M_R=5.0':>12s}")
    for MLf in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
        vals = []
        for MRf in (0.5, 1.0, 2.0, 5.0):
            mg, *_ = margin(H, MLf * Mp_ref, MRf * Mp_ref, Mp_col)
            vals.append(mg)
        print(f"  {MLf:>10.2f} " + "".join(f"{v:>12.3f}" for v in vals))

    print("\n=== is the margin independent of M_R for large M_L? ===")
    for MLf in (2.0, 3.0, 5.0):
        vals = [margin(H, MLf * Mp_ref, MRf * Mp_ref, Mp_col)[0]
                for MRf in (0.25, 0.5, 1.0, 2.0, 5.0)]
        spread = max(vals) - min(vals)
        print(f"  M_L={MLf:>4.1f}x: margins {[round(v,2) for v in vals]}  "
              f"spread {spread:.3f}")

    print("\n=== H dependence at the balanced point ===")
    for Hh in (2, 3, 4, 5, 6, 8, 12):
        mg, B0, e, it, ex, _ = margin(Hh, Mp_ref, Mp_ref, Mp_col)
        print(f"  H={Hh:>3d}: margin {mg:>10.3f}  interior={[round(v,2) for v in it]}")

    dest = PKG / "data" / "b2_dependence.json"
    dest.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
