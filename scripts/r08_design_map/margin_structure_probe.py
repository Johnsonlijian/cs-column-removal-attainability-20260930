"""Structure of the interval margin under uniform story capacity.

Measured result that needs an explanation: with B >= 2 and a uniform capacity
per story, every one of 46,368 removals tested is exact. Why?

If the margin of a contiguous-story interval were additive over its story
increments, then a margin would be a sum of non-negative pieces whenever the
capacity pattern is "well ordered", and non-negativity would follow rather than
have to be checked interval by interval.

This script tests that structure directly. For each removal it recomputes the
interval margins from the certificate's own story increments and compares their
minimum against the certificate's reported minimum margin. If the two agree for
every removal, the margin is a pure contiguous-run cost and the uniform case can
be argued from the increments rather than from a search.
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

GRAVITY, TRIB, BETA = 6.0, 1.0, 1.15


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
    f = Frame(f"struct_H{H}_B{B}", spans, heights, members, loads.tolist(),
              provenance="margin-structure probe")
    f.validate()
    return f


def increments(sc: Scenario, x: np.ndarray):
    """Recompute entry e_a, interior d_r and exit f_b for every threshold band.

    Uses the engine's own band potentials, so this is the certificate's own
    decomposition rather than a reimplementation of the theorem.
    """
    bands = []
    for lo, hi, t, q in sc.bands:
        V, B0, _ = sc._potentials(x, t, q)
        H = sc.f.H
        # V[r-1, a, b] is the transition potential at story r for states a -> b
        d = np.array([V[r, 1, 1] - V[r, 0, 0] for r in range(H)])
        e = np.empty(H)
        f = np.empty(H)
        for r in range(H):
            # entering the run at story r+1 (1-based r+1) from the local path
            e[r] = B0 if r == 0 else (V[r - 1, 0, 1] - V[r - 1, 0, 0])
            # leaving the run after story r+1
            f[r] = V[r, 1, 0] - V[r, 0, 0]
        bands.append({"width": hi - lo, "q": q, "d": d, "e": e, "f": f,
                      "V": V, "B0": B0})
    return bands


def min_margin_from_increments(sc: Scenario, x: np.ndarray):
    """Minimum interval margin rebuilt from entry/interior/exit increments."""
    best = np.inf
    witness = None
    for band in bands_iter(sc, x):
        H = sc.f.H
        e, d, f, width = band["e"], band["d"], band["f"], band["width"]
        for a in range(H):
            acc = e[a]
            for b in range(a, H):
                if b > a:
                    acc += d[b - 1]
                val = (acc + f[b]) * width
                # margins are accumulated over bands; track the per-band value
                if val < best:
                    best, witness = val, (a + 1, b + 1, band["q"])
    return best, witness


def bands_iter(sc: Scenario, x: np.ndarray):
    yield from increments(sc, x)


def main() -> None:
    pairs = [("W18X40", "W14X109"), ("W21X50", "W14X132"),
             ("W24X62", "W14X176")]
    agree = 0
    disagree = 0
    examples = []
    tested = 0
    for B in (1, 2, 3):
        for H in (2, 3, 4, 6):
            for beam, column in pairs:
                f = build(H, B, 6.0, 4.0, beam, column)
                for removal in f.scenarios():
                    sc = Scenario(f, removal)
                    x = np.ones(f.E)
                    reported = float(min_interval_margin(sc, x))
                    rebuilt, wit = min_margin_from_increments(sc, x)
                    tested += 1
                    if abs(reported - rebuilt) <= 1e-6 * max(1.0, abs(reported)):
                        agree += 1
                    else:
                        disagree += 1
                        if len(examples) < 6:
                            examples.append({
                                "H": H, "B": B, "removal": list(removal),
                                "reported": reported, "rebuilt": rebuilt,
                                "witness": wit,
                            })
    print(f"removals tested        : {tested}")
    print(f"increment model agrees : {agree}")
    print(f"disagrees              : {disagree}")
    for e in examples:
        print("  ", e)

    dest = PKG / "data" / "margin_structure_probe.json"
    dest.write_text(json.dumps({"tested": tested, "agree": agree,
                                "disagree": disagree,
                                "examples": examples}, indent=2),
                    encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
