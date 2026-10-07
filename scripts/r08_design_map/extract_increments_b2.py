"""Extract the exact increments e_a, d_r, f_b for B = 2 and fit the closed form.

The previous attempt guessed the formula from the balanced node table only, and
got it wrong: the margin is M_L + C_u there, not 2M_L + 2C_u, and M_R clearly
enters because raising it changes the margin materially. This script prints the
actual entry, interior and exit increments band by band, for several capacity
combinations, so the formula can be read off instead of guessed.
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


def build(H, B, span, height, beam_left, beam_right, col):
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(B - 1, span), span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = beam_left if g == 0 else beam_right
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, col, col, 14.0 * col, heights[r - 1]])
    f = Frame(f"inc_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="increment extraction")
    f.validate()
    return f


def increments(sc: Scenario, x: np.ndarray):
    """Return per-band e_a, d_r, f_b and B0 exactly as the engine builds them."""
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
        best = float(np.min(m + exit_cost))
        out.append({"q": q, "t": t, "B0": float(B0),
                    "entry": entry.tolist(),
                    "interior": interior.tolist(),
                    "exit": exit_cost.tolist(),
                    "min_interval": best,
                    "witness": int(np.argmin(m + exit_cost)) + 1})
    return out


def main() -> None:
    Mp_ref = Mp("W18X40")
    Mp_col = Mp("W14X109")
    H = 4
    combos = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0), (5.0, 4.0),
              (0.5, 1.0), (1.0, 0.5)]
    print(f"M_L(ref) = {Mp_ref:.3f} kNm, C_u(ref) = {Mp_col:.3f} kNm, H = {H}\n")
    for fl, fr in combos:
        f = build(H, 2, 6.0, 4.0, fl * Mp_ref, fr * Mp_ref, Mp_col)
        sc = Scenario(f, (1, 0))
        x = np.ones(f.E)
        engine = float(min_interval_margin(sc, x))
        print(f"--- M_L={fl*Mp_ref:.2f}  M_R={fr*Mp_ref:.2f}  "
              f"margin={engine:.3f} ---")
        for band in increments(sc, x):
            print(f"   band q={band['q']} t={band['t']:+.5f}  "
                  f"B0={band['B0']:.2f}  min={band['min_interval']:.3f} "
                  f"at story {band['witness']}")
            print(f"     entry    = {[round(v, 2) for v in band['entry']]}")
            print(f"     interior = {[round(v, 2) for v in band['interior']]}")
            print(f"     exit     = {[round(v, 2) for v in band['exit']]}")
        print()

    # ---- candidate closed forms -------------------------------------------
    print("=== candidate formulas vs engine ===")
    cands = {
        "M_L + C_u": lambda ML, MR, C: ML + C,
        "M_L + M_R + C_u": lambda ML, MR, C: ML + MR + C,
        "min(M_L,M_R) + C_u": lambda ML, MR, C: min(ML, MR) + C,
        "M_L + C_u  (C_u from story below)": lambda ML, MR, C: ML + C,
    }
    for name, fn in cands.items():
        ok = 0
        tot = 0
        errs = []
        for fl, fr in combos:
            f = build(H, 2, 6.0, 4.0, fl * Mp_ref, fr * Mp_ref, Mp_col)
            sc = Scenario(f, (1, 0))
            x = np.ones(f.E)
            e = float(min_interval_margin(sc, x))
            p = fn(fl * Mp_ref, fr * Mp_ref, Mp_col)
            tot += 1
            if abs(e - p) <= 1e-6 * max(1.0, abs(e)):
                ok += 1
            else:
                errs.append(round(e - p, 3))
        print(f"  {name:<38s} {ok}/{tot}  residuals {errs[:5]}")


if __name__ == "__main__":
    main()
