"""Attack the uniform multi-bay result: is it real, or an artefact of symmetry?

The claim "no multi-bay frame with uniform story capacity is ever inexact"
produced 0 counterexamples in 51,520 removals. A result that clean in a
structural problem is suspicious. The leading suspicion is that the test cases
were too symmetric.

Why symmetry could be doing the work
------------------------------------
In the certificate, the two threshold bands and the interval margin are built
from the beam chord-rate anchors adjacent to the removed column line. Those
anchors are -delta/l_L and +delta/l_R. If the frame has equal spans, or if the
load is distributed so that the demand is symmetric about the cut, the margin
may reduce to a trivially non-negative quantity that has nothing to do with
multi-bay robustness.

This script therefore varies exactly the quantities that were fixed:

  span asymmetry      l_L / l_R away from 1
  load asymmetry      nodal load on the two sides of the cut
  capacity by bay     uniform per story is relaxed to per-bay
  capacity by story   vertical taper / non-uniform storey strength
  unequal storey height

and reports the first counterexamples with the case printed, so the claim's
true domain of validity can be stated instead of assumed.
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


def make(H, B, spans, heights, beam, column, loads, bay_factors=None,
         story_factors=None):
    Mp_b, Mp_c = Mp(beam), Mp(column)
    if bay_factors is None:
        bay_factors = [1.0] * B
    if story_factors is None:
        story_factors = [1.0] * H
    members = []
    for r in range(1, H + 1):
        sf = story_factors[r - 1]
        for g in range(B):
            mb = Mp_b * bay_factors[g] * sf
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            # column factor follows the bays it frames into
            cf = bay_factors[min(g, B - 1)] if g < B else bay_factors[-1]
            mc = Mp_c * cf * sf
            members.append(["c", r, g, mc, mc, 14.0 * mc, heights[r - 1]])
    f = Frame(f"attack_H{H}_B{B}", list(spans), list(heights), members,
              [list(row) for row in loads], provenance="symmetry attack")
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
    results = []

    def run(label, H, B, spans, heights, beam, column, loads,
            bay_factors=None, story_factors=None):
        f = make(H, B, spans, heights, beam, column, loads, bay_factors,
                 story_factors)
        worst, rem, n_bad, n = scan(f)
        results.append({"family": label, "H": H, "B": B, "spans": spans,
                        "heights": heights, "beam": beam, "column": column,
                        "bay_factors": bay_factors,
                        "story_factors": story_factors,
                        "worst_margin": worst, "worst_removal": rem,
                        "n_inexact": n_bad, "n_removals": n,
                        "exact_all": n_bad == 0})
        return n_bad, n

    beam, column = "W18X40", "W14X109"

    # ---- Family 1: span asymmetry, otherwise the verified configuration ----
    print("=== FAMILY 1: span asymmetry (l_L/l_R away from 1) ===")
    for ratio in (1.0, 1.5, 2.0, 3.0, 5.0):
        for B in (2, 3):
            spans = [6.0 * ratio] + [6.0] * (B - 1)
            share = np.r_[spans[0] / 2, (np.array(spans[:-1]) + np.array(spans[1:])) / 2, spans[-1] / 2]
            loads = np.outer(np.ones(4), share) * (GRAVITY * TRIB)
            n_bad, n = run(f"span_ratio_{ratio}", 4, B, spans, [4.0] * 4,
                           beam, column, loads)
            print(f"  ratio={ratio:<4} B={B}: {n_bad}/{n} inexact")

    # ---- Family 2: load asymmetry across the cut ----
    print("\n=== FAMILY 2: nodal load asymmetry across the removed line ===")
    for skew in (1.0, 1.5, 2.5, 4.0):
        for B in (2, 3):
            spans = [6.0] * B
            share = np.r_[6.0 / 2, np.full(B - 1, 6.0), 6.0 / 2]
            share = share * np.linspace(1.0, skew, B + 1)
            loads = np.outer(np.ones(4), share) * (GRAVITY * TRIB)
            n_bad, n = run(f"load_skew_{skew}", 4, B, spans, [4.0] * 4,
                           beam, column, loads)
            print(f"  skew={skew:<4} B={B}: {n_bad}/{n} inexact")

    # ---- Family 3: capacity varying by bay (uniform per story relaxed) ----
    print("\n=== FAMILY 3: capacity varying by bay ===")
    for ratio in (1.0, 1.5, 2.0, 3.0):
        for B in (2, 3):
            spans = [6.0] * B
            share = np.r_[3.0, np.full(B - 1, 6.0), 3.0]
            loads = np.outer(np.ones(4), share) * (GRAVITY * TRIB)
            factors = [ratio] + [1.0] * (B - 1)
            n_bad, n = run(f"bay_factor_{ratio}", 4, B, spans, [4.0] * 4,
                           beam, column, loads, bay_factors=factors)
            print(f"  bay factor={ratio:<4} B={B}: {n_bad}/{n} inexact")

    # ---- Family 4: capacity varying by story ----
    print("\n=== FAMILY 4: capacity varying by story (taper / soft storey) ===")
    for H in (4, 6):
        spans = [6.0] * 3
        share = np.r_[3.0, 6.0, 6.0, 3.0]
        loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
        for label, sf in (
            ("uniform", [1.0] * H),
            ("soft_ground", [0.5] + [1.0] * (H - 1)),
            ("stiff_ground", [2.0] + [1.0] * (H - 1)),
            ("linear_taper", list(np.linspace(1.6, 1.0, H))),
            ("alternating", [1.0 if i % 2 == 0 else 1.6 for i in range(H)]),
        ):
            n_bad, n = run(f"story_{label}", H, 3, spans, [4.0] * H,
                           beam, column, loads, story_factors=sf)
            print(f"  {label:<14} H={H}: {n_bad}/{n} inexact")

    # ---- Family 5: unequal storey height ----
    print("\n=== FAMILY 5: unequal storey height ===")
    for H in (4, 6):
        spans = [6.0] * 3
        share = np.r_[3.0, 6.0, 6.0, 3.0]
        loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
        for label, hh in (("uniform", [4.0] * H),
                          ("graded", list(np.linspace(5.0, 3.0, H))),
                          ("tall_ground", [6.0] + [4.0] * (H - 1))):
            n_bad, n = run(f"height_{label}", H, 3, spans, hh, beam, column,
                           loads)
            print(f"  {label:<14} H={H}: {n_bad}/{n} inexact")

    dest = PKG / "data" / "symmetry_attack.json"
    dest.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    print("\n=== OVERALL ===")
    multi = [r for r in results if r["B"] >= 2]
    bad = [r for r in multi if not r["exact_all"]]
    print(f"  multi-bay design points tested : {len(multi)}")
    print(f"  with at least one inexact removal: {len(bad)}")
    for r in bad:
        print(f"   COUNTEREXAMPLE family={r['family']} H={r['H']} B={r['B']} "
              f"spans={r['spans']} bay_factors={r['bay_factors']} "
              f"story_factors={r['story_factors']}")
        print(f"     worst_margin={r['worst_margin']:.4f} at "
              f"{r['worst_removal']} ({r['n_inexact']}/{r['n_removals']})")


if __name__ == "__main__":
    main()
