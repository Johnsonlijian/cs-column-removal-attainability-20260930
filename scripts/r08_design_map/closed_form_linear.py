"""The closed form: the interval margin is an integer linear function.

Found by reading the numbers rather than by fitting. At the reference geometry
(H=4, B=2, span 6 m, storey height 4 m, capacities in kN m) the margin is

    interior removal (1,1):   margin = -8 M_det - 4 M_int + C
    exterior removal (1,0):   margin = -8 M_det + 4 M_int + C

where M_det is the beam on the detached side of the cut, M_int the beam on the
intact side, and C the column end capacity. The coefficients are integers and
the interior/exterior difference is the SIGN of the M_int coefficient.

Working in units of the reference beam, with m = M_det/M_ref and n = M_int/M_ref
and c = C/M_ref, this reads

    interior:  margin / M_ref = -8 m - 4 n + c
    exterior:  margin / M_ref = -8 m + 4 n + c

which gives immediate design rules. For the interior removal the screen is exact
iff c >= 8m + 4n; for the exterior removal iff c >= 8m - 4n. Both are single
linear inequalities in three capacities, so the exactness set is a halfspace in
each case and the whole certificate collapses to one inequality.

This is a strong claim and an earlier round of this work concluded the opposite
(that no reduced closed form exists, after two candidate forms failed on a
1,944-configuration sweep). That conclusion was wrong, and the reason is now
clear: the candidates tested were nonlinear in the capacities, while the truth
is linear. This script tests the linear forms over a wide sweep, including other
geometries, so the claim can be rejected if it is a coincidence of the reference
frame.
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

from column_threshold import margin  # noqa: E402
from real_section_design_map import Mp  # noqa: E402

BEAM = Mp("W18X40")
COL = Mp("W14X109")


def main() -> None:
    rows = []
    print("reference beam = %.3f kNm ; reference column = %.3f kNm"
          % (BEAM, COL))
    print("capacities expressed in kN m; prediction in kN m\n")

    # ---- the reference geometry: sweep all three capacities --------------
    print("=== H=4 B=2 span 6 h 4 ===")
    print(f"  {'removal':>8s} {'M_det':>9s} {'M_int':>9s} {'C':>10s} "
          f"{'margin':>13s} {'-8Md-4Mi+C':>13s} {'-8Md+4Mi+C':>13s}")
    inter_ok = ext_ok = tot = 0
    for removal, tag in (((1, 1), "interior"), ((1, 0), "exterior")):
        for md_f in (0.5, 1.0, 1.5, 2.0, 3.0):
            for mi_f in (0.5, 1.0, 1.5, 2.0, 3.0):
                for c_f in (0.5, 1.0, 2.0, 4.0):
                    m = margin(4, 2, [6.0, 6.0], [4.0] * 4,
                               M_det=md_f * BEAM, M_int=mi_f * BEAM,
                               C=c_f * COL, removal=removal)
                    p_i = -8 * md_f * BEAM - 4 * mi_f * BEAM + c_f * COL
                    p_e = -8 * md_f * BEAM + 4 * mi_f * BEAM + c_f * COL
                    pred = p_i if removal == (1, 1) else p_e
                    good = abs(m - pred) <= 1e-6 * max(1.0, abs(pred))
                    tot += 1
                    if removal == (1, 1):
                        inter_ok += int(good)
                    else:
                        ext_ok += int(good)
                    rows.append({"removal": list(removal), "tag": tag,
                                 "M_det_factor": md_f, "M_int_factor": mi_f,
                                 "C_factor": c_f, "margin": m,
                                 "pred_interior": p_i, "pred_exterior": p_e,
                                 "match": bool(good)})
                    if md_f in (1.0, 2.0) and mi_f in (1.0, 2.0) \
                            and c_f in (1.0, 2.0):
                        print(f"  {tag:>8s} {md_f*BEAM:>9.2f} "
                              f"{mi_f*BEAM:>9.2f} {c_f*COL:>10.2f} "
                              f"{m:>13.3f} {p_i:>13.3f} {p_e:>13.3f}")

    n_int = len([r for r in rows if r["tag"] == "interior"])
    n_ext = len([r for r in rows if r["tag"] == "exterior"])
    print(f"\n  interior form -8Md-4Mi+C matches {inter_ok}/{n_int}")
    print(f"  exterior form -8Md+4Mi+C matches {ext_ok}/{n_ext}")

    # ---- does the form survive a different geometry? ---------------------
    print("\n=== the same coefficients at other geometries? ===")
    print("  If the coefficients are geometric constants they should persist;")
    print("  if they change, they are functions of the geometry and the forms")
    print("  above are specific to the reference frame.")
    for H, B, spans, heights, tag in (
            (4, 2, [6.0, 6.0], [4.0] * 4, "H4 B2 s6 h4"),
            (2, 2, [6.0, 6.0], [4.0] * 2, "H2 B2 s6 h4"),
            (6, 2, [6.0, 6.0], [4.0] * 6, "H6 B2 s6 h4"),
            (4, 2, [9.0, 9.0], [4.0] * 4, "H4 B2 s9 h4"),
            (4, 2, [6.0, 6.0], [3.0] * 4, "H4 B2 s6 h3")):
        for removal, lab in (((1, B // 2), "int"), ((1, 0), "ext")):
            base = margin(H, B, spans, heights, M_det=BEAM, M_int=BEAM,
                          C=COL, removal=removal)
            d_det = margin(H, B, spans, heights, M_det=2 * BEAM,
                           M_int=BEAM, C=COL, removal=removal) - base
            d_int = margin(H, B, spans, heights, M_det=BEAM,
                           M_int=2 * BEAM, C=COL, removal=removal) - base
            d_c = margin(H, B, spans, heights, M_det=BEAM, M_int=BEAM,
                         C=2 * COL, removal=removal) - base
            print(f"  {tag:<14s} {lab}: base={base:>10.2f}  "
                  f"dM_det={d_det/BEAM:>+8.3f}  dM_int={d_int/BEAM:>+8.3f}  "
                  f"dC={d_c/COL:>+8.3f}   (in units of ref beam / column)")

    dest = PKG / "data" / "closed_form_linear.json"
    dest.write_text(json.dumps({
        "claim": "margin = -8 M_det - 4 M_int + C (interior); "
                 "margin = -8 M_det + 4 M_int + C (exterior); "
                 "coefficients in units of the reference capacities",
        "interior_matches": inter_ok, "interior_total": n_int,
        "exterior_matches": ext_ok, "exterior_total": n_ext,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
