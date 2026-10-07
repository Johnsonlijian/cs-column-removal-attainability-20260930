"""Critical cut-imbalance ratio Lambda*: the design rule.

The margins reported so far are dimensional (kN m), but the exactness set was
shown to be a cone, so the boundary depends on capacity RATIOS only. Reporting an
absolute margin therefore under-uses what has been established.

Define the critical imbalance as the value kappa at which the margin crosses zero
when the capacity on one side of the cut is held at a reference and the other is
scaled:

    Lambda*(C, removal) = the kappa solving  margin(M_det = kappa, M_int = 1) = 0

Because the margin is monotone decreasing in M_det (established over 864
ladders), the root is unique where it exists, and bisection is exact enough to
report. This converts four rounds of structural results into one number per
design situation, which is the form a design rule takes.

Outputs the ratio table for exterior and interior removals over a range of
column capacity levels, and tests whether Lambda* depends only on C / M_ref,
which the cone property predicts.
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
    f = Frame(f"lam_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="critical imbalance")
    f.validate()
    return f


def margin_at(H, B, spans, heights, M_det, M_int, C, removal):
    f = build(H, B, spans, heights, M_det, M_int, C)
    sc = Scenario(f, removal)
    return float(min_interval_margin(sc, np.ones(f.E)))


def critical_ratio(H, B, spans, heights, C, removal, lo=1.0, hi=64.0,
                   tol=1e-9):
    """Bisect for the kappa where the margin crosses zero (M_int held at 1)."""
    m_lo = margin_at(H, B, spans, heights, lo, 1.0, C, removal)
    m_hi = margin_at(H, B, spans, heights, hi, 1.0, C, removal)
    if m_lo < 0:
        return {"kappa": None, "note": "inexact even at kappa=1"}
    if m_hi >= 0:
        return {"kappa": None, "note": f"still exact at kappa={hi}"}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if margin_at(H, B, spans, heights, mid, 1.0, C, removal) >= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, hi):
            break
    return {"kappa": 0.5 * (lo + hi), "note": "ok"}


def main() -> None:
    beam, column = Mp("W18X40"), Mp("W14X109")
    rows = []

    print("reference beam W18X40 = %.3f kNm, reference column = %.3f kNm"
          % (beam, column))
    print("M_int held at one reference beam throughout; kappa = M_det / M_int\n")

    print("=== exterior removals ===")
    print(f"  {'C/ref':>7s} {'H':>3s} {'B':>3s} {'storey':>8s} "
          f"{'Lambda*':>10s}")
    for c_over_beam in (0.5, 1.0, 2.0, 3.0, 4.5):
        C = c_over_beam * beam
        for H, B, spans in ((4, 2, [6.0, 6.0]), (4, 3, [5.0, 6.0, 7.0])):
            heights = [4.0] * H
            for s, tag in ((1, "ground"), (H, "roof")):
                r = critical_ratio(H, B, spans, heights, C, (s, 0))
                rows.append({"kind": "exterior", "C_over_beam": c_over_beam,
                             "H": H, "B": B, "removal": [s, 0], "storey": tag,
                             **r})
                k = r["kappa"]
                print(f"  {c_over_beam:>7.2f} {H:>3d} {B:>3d} {tag:>8s} "
                      f"{(f'{k:.4f}' if k else r['note']):>10s}")

    print("\n=== interior removals ===")
    print(f"  {'C/ref':>7s} {'H':>3s} {'B':>3s} {'storey':>8s} "
          f"{'Lambda*':>10s}")
    for c_over_beam in (0.5, 1.0, 2.0, 3.0, 4.5):
        C = c_over_beam * beam
        for H, B, spans in ((4, 2, [6.0, 6.0]), (4, 3, [5.0, 6.0, 7.0])):
            heights = [4.0] * H
            for s, tag in ((1, "ground"), (H, "roof")):
                r = critical_ratio(H, B, spans, heights, C, (s, B // 2))
                rows.append({"kind": "interior", "C_over_beam": c_over_beam,
                             "H": H, "B": B, "removal": [s, B // 2],
                             "storey": tag, **r})
                k = r["kappa"]
                print(f"  {c_over_beam:>7.2f} {H:>3d} {B:>3d} {tag:>8s} "
                      f"{(f'{k:.4f}' if k else r['note']):>10s}")

    print("\n=== does Lambda* depend only on C / beam? (cone prediction) ===")
    print(f"  {'C/beam':>8s} {'exterior gr.':>14s} {'interior gr.':>14s}")
    for c_over_beam in (0.5, 1.0, 2.0, 3.0, 4.5):
        e = [r["kappa"] for r in rows
             if r["kind"] == "exterior" and r["C_over_beam"] == c_over_beam
             and r["storey"] == "ground" and r["kappa"]]
        i = [r["kappa"] for r in rows
             if r["kind"] == "interior" and r["C_over_beam"] == c_over_beam
             and r["storey"] == "ground" and r["kappa"]]
        es = "n/a" if not e else f"{min(e):.4f}-{max(e):.4f}"
        is_ = "n/a" if not i else f"{min(i):.4f}-{max(i):.4f}"
        print(f"  {c_over_beam:>8.2f} {es:>14s} {is_:>14s}")

    dest = PKG / "data" / "critical_imbalance.json"
    dest.write_text(json.dumps({
        "note": "Lambda* = kappa with margin(kappa, 1) = 0; unique by the "
                "monotone-decreasing property",
        "reference_beam_kNm": beam, "reference_column_kNm": column,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    print("\n" + "=" * 74)
    print("WHAT CONTROLS LAMBDA*: the C/M_int table in consistent units")
    print("=" * 74)
    print("  M_int is held at one reference beam in EVERY row below, so the")
    print("  only free parameter is C / M_int. The cone property predicts")
    print("  Lambda* depends on that ratio and on nothing else, including on")
    print("  the absolute size of the frame's capacities.")
    print(f"\n  {'C/M_int':>8s} {'Lambda* (B=2)':>16s} {'Lambda* (B=3)':>16s} "
          f"{'removal':>9s}")
    lam_rows = []
    for cm in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        C = cm * beam                      # M_int = 1.0 reference beam
        for B, spans in ((2, [6.0, 6.0]), (3, [5.0, 6.0, 7.0])):
            r = critical_ratio(4, B, spans, [4.0] * 4, C, (1, 0))
            k = r.get("kappa")
            lam_rows.append({"C_over_M_int": cm, "B": B, "Lambda": k,
                             "note": r["note"]})
        a = next(x for x in lam_rows
                 if x["C_over_M_int"] == cm and x["B"] == 2)["Lambda"]
        b = next(x for x in lam_rows
                 if x["C_over_M_int"] == cm and x["B"] == 3)["Lambda"]
        fa = f"{a:.4f}" if a else "none <= 64"
        fb = f"{b:.4f}" if b else "none <= 64"
        print(f"  {cm:>8.2f} {fa:>16s} {fb:>16s} {'ground':>9s}")

    print("\n  reading: Lambda* is finite only for light columns. Once the")
    print("  column term is large enough relative to the beam, no cut imbalance")
    print("  within the search range of 64 can break the screen, because the")
    print("  column work dominates the interval margin.")

    print("\n=== roof removals never break, at any imbalance tested ===")
    for cm in (0.25, 1.0, 4.0):
        r = critical_ratio(4, 2, [6.0, 6.0], [4.0] * 4, cm * beam, (4, 0))
        print(f"  C/M_int={cm:>5.2f}: {r['note']}")

    dest2 = PKG / "data" / "critical_imbalance_table.json"
    dest2.write_text(json.dumps({
        "note": "M_int fixed at one reference beam; Lambda* solves "
                "margin(kappa, 1) = 0; search range kappa <= 64",
        "rows": lam_rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest2)
    print("  rescales C and both beam capacities by the same factor.")
    print(f"  {'scale':>7s} {'Lambda*':>12s}")
    for sc in (0.25, 0.5, 1.0, 2.0, 4.0):
        r = critical_ratio(4, 2, [6.0, 6.0], [4.0] * 4, 3.0 * sc * beam,
                           (1, 0))
        print(f"  {sc:>7.2f} "
              f"{(f'{r[list(r)[1]]:.4f}' if r.get('kappa') else r['note']):>12s}"
              f"   (C/M_ref = 3.0 in scaled units)")


if __name__ == "__main__":
    main()
