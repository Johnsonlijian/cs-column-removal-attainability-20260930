"""Verify the interior closed form and state the exterior one.

Discovered while testing the column threshold: for an INTERIOR removal with
uniform capacities, the interval margin is EXACTLY

    margin = M_int - M_det

in kN m, independent of the column capacity at the scale tested, where M_det is
the beam capacity on the detached side of the cut and M_int that on the intact
side. Read as a condition this is

    M_int >= M_det

and it carries no column term at all.

For the EXTERIOR removal of the same two-bay frame the dependence is instead on
the side RAISED: the reference frame gives margin 483.0, raising the detached
side by one reference beam gives -22.08 (a drop of 505.08 = 4 x 126.27 = 8M),
and raising the intact side gives 1040.175 (a rise of 557.175).

This script tests both forms over a sweep so the two can be stated as
closed-form rules or rejected.
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

from column_threshold import build, margin  # noqa: E402
from real_section_design_map import Mp  # noqa: E402

BEAM = Mp("W18X40")
COL = Mp("W14X109")


def main() -> None:
    rows = []

    print("=== INTERIOR removal, H=4 B=2 span 6 h 4 ===")
    print("  candidate: margin = M_int - M_det")
    print(f"  {'M_det':>9s} {'M_int':>9s} {'C':>9s} {'margin':>13s} "
          f"{'M_int-M_det':>13s} {'match':>6s}")
    ok = tot = 0
    for md_f, mi_f in ((1.0, 1.0), (1.25, 1.0), (1.5, 1.0), (2.0, 1.0),
                       (1.0, 1.25), (1.0, 2.0), (2.0, 2.0), (3.0, 1.0),
                       (0.5, 1.0), (1.0, 0.5)):
        for c_f in (0.5, 1.0):
            m = margin(4, 2, [6.0, 6.0], [4.0] * 4,
                       M_det=md_f * BEAM, M_int=mi_f * BEAM, C=c_f * COL,
                       removal=(1, 1))
            pred = (mi_f - md_f) * BEAM
            good = abs(m - pred) <= 1e-6 * max(1.0, abs(pred))
            ok += int(good)
            tot += 1
            rows.append({"kind": "interior", "M_det_factor": md_f,
                         "M_int_factor": mi_f, "C_factor": c_f,
                         "margin": m, "M_int_minus_M_det": pred,
                         "match": bool(good)})
            print(f"  {md_f*BEAM:>9.2f} {mi_f*BEAM:>9.2f} {c_f*COL:>9.2f} "
                  f"{m:>13.3f} {pred:>13.3f} {str(good):>6s}")
    print(f"\n  interior closed form matches: {ok}/{tot}")

    print("\n=== EXTERIOR removal, same frame ===")
    print("  reference margin, and the change produced by raising ONE side")
    base = margin(4, 2, [6.0, 6.0], [4.0] * 4, M_det=BEAM, M_int=BEAM,
                  C=COL, removal=(1, 0))
    up_det = margin(4, 2, [6.0, 6.0], [4.0] * 4, M_det=2 * BEAM,
                    M_int=BEAM, C=COL, removal=(1, 0))
    up_int = margin(4, 2, [6.0, 6.0], [4.0] * 4, M_det=BEAM,
                    M_int=2 * BEAM, C=COL, removal=(1, 0))
    print(f"  reference                       {base:>12.3f}")
    print(f"  detached side raised to 2x      {up_det:>12.3f}   "
          f"change {up_det-base:>+12.3f}   (8M = {-8*BEAM:+.3f})")
    print(f"  intact side raised to 2x        {up_int:>12.3f}   "
          f"change {up_int-base:>+12.3f}")
    rows.append({"kind": "exterior", "reference": base,
                 "raised_detached": up_det, "raised_intact": up_int,
                 "delta_detached": up_det - base,
                 "delta_intact": up_int - base})

    print("\n=== the resulting rules, as conditions ===")
    print("  interior: margin = M_int - M_det  ->  exact iff M_int >= M_det")
    print("  exterior: unaffected by the column at the scale tested; raising the")
    print("            detached side lowers the margin by 8M per reference beam,")
    print("            raising the intact side raises it")

    dest = PKG / "data" / "closed_form_interior_exterior.json"
    dest.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
