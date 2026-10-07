"""Anchor and band structure of INTERIOR removals, exactly.

Exterior removals collapse to one nonzero band because the intact-side beam has
anchor exactly zero. Interior removals keep two bands, and the whole point of
this script is to find out what the two anchors actually are and how the bands
are laid out, rather than assuming the symmetric +-1/l pair.

Prints, for each interior removal, every surviving member's anchor, the sorted
band list with widths and sign conventions q, and the bandwise interval minima,
so the two-band closed form can be derived from measured structure.
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
    edges = np.r_[0.0, np.cumsum(spans)]
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
    f = Frame(f"int_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="interior removal structure")
    f.validate()
    return f


def band_report(sc: Scenario, x: np.ndarray):
    """Bands with width, sign convention, and the bandwise minimum interval."""
    H = sc.f.H
    out = []
    for lo, hi, t, q in sc.bands:
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
        out.append({
            "lo": round(lo, 8), "hi": round(hi, 8), "width": round(hi - lo, 8),
            "t": round(t, 8), "q": q, "B0": float(B0),
            "min_interval": float(np.min(m + exit_cost)),
            "entry": entry.tolist(), "interior": interior.tolist(),
            "exit": exit_cost.tolist(),
        })
    return out


def main() -> None:
    beam = Mp("W18X40")
    column = Mp("W14X109")

    print("=" * 78)
    print("INTERIOR REMOVAL: B = 2, H = 2, equal spans, balanced capacities")
    print("=" * 78)
    f = build(2, 2, [6.0, 6.0], [4.0, 4.0], beam, beam, column)
    sc = Scenario(f, (1, 1))
    x = np.ones(f.E)
    print(f"removal = {sc.removal},  D = {sc.D}")
    print("\nsurviving members and anchors:")
    for k, m in enumerate(sc.members):
        a = sc.anchor[k]
        print(f"   {k:>2d}  kind={m[0]} story={int(m[1])} grid={int(m[2])}  "
              f"anchor={'None' if a is None else round(a, 6)}")
    print("\nbands and bandwise minima (margin = min over bands of min_interval):")
    for bd in band_report(sc, x):
        print(f"   [{bd['lo']:+.6f}, {bd['hi']:+.6f}]  width={bd['width']:.6f} "
              f" t={bd['t']:+.6f}  q={bd['q']}  B0={bd['B0']:.2f}  "
              f"min_interval={bd['min_interval']:.3f}")
    print(f"\n   reported margin = {float(min_interval_margin(sc, x)):.3f}")

    # ---- does the two-band case also have a special ground-storey term? ----
    print("\n" + "=" * 78)
    print("PER-BAND INCREMENTS, interior removal, H = 4")
    print("=" * 78)
    f = build(4, 2, [6.0, 6.0], [4.0] * 4, beam, beam, column)
    for removal in ((1, 1), (2, 1), (4, 1)):
        sc = Scenario(f, removal)
        x = np.ones(f.E)
        print(f"\n--- removal {removal}, margin = "
              f"{float(min_interval_margin(sc, x)):.3f} ---")
        for i, bd in enumerate(band_report(sc, x)):
            if bd["width"] <= 0:
                print(f"   band {i}: width 0, skipped")
                continue
            print(f"   band {i}: t={bd['t']:+.6f} q={bd['q']} "
                  f"width={bd['width']:.6f} min={bd['min_interval']:.3f}")
            print(f"      entry    = {[round(v,2) for v in bd['entry']]}")
            print(f"      interior = {[round(v,2) for v in bd['interior']]}")
            print(f"      exit     = {[round(v,2) for v in bd['exit']]}")

    # ---- which band wins? does it depend on capacities? ----
    print("\n" + "=" * 78)
    print("WHICH BAND ATTAINS THE MINIMUM, as capacities vary")
    print("=" * 78)
    print(f"  {'left':>8s} {'right':>8s} {'col':>8s} "
          f"{'band(-)':>12s} {'band(+)':>12s} {'margin':>12s} {'winner':>8s}")
    for fl, fr, cf in [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0),
                       (2.0, 2.0, 1.0), (0.5, 1.0, 1.0), (1.0, 1.0, 0.5),
                       (3.0, 1.0, 2.0), (1.0, 3.0, 2.0)]:
        f = build(4, 2, [6.0, 6.0], [4.0] * 4, fl * beam, fr * beam,
                  cf * column)
        sc = Scenario(f, (1, 1))
        x = np.ones(f.E)
        rep = [b for b in band_report(sc, x) if b["width"] > 0]
        vals = [b["min_interval"] for b in rep]
        margin = float(min_interval_margin(sc, x))
        win = "neg" if vals[0] < vals[-1] else ("pos" if vals[-1] < vals[0]
                                                else "tie")
        print(f"  {fl:>8.2f} {fr:>8.2f} {cf:>8.2f} {vals[0]:>12.3f} "
              f"{vals[-1]:>12.3f} {margin:>12.3f} {win:>8s}")


if __name__ == "__main__":
    main()
