"""Independent verification of the uniform multi-bay result.

Claim under test
----------------
For a frame with B >= 2 bays and a capacity pattern that is uniform within each
story, the zero-sway screen is exact at every removal position.

The sharp-rule sweep tested this through the packaged certificate
(`min_interval_margin`) and found 0 counterexamples in 46,368 removals. Because
that test uses the engine's own certificate, it could in principle share a bug
with the object it certifies. This script therefore re-verifies the claim on a
subset using a DIFFERENT mechanism: a direct kinematic linear program for the
complete limit and for the zero-sway restriction, assembled independently by
`Scenario.kinematic`, which builds its own equilibrium and compatibility
matrices and does not call the chain solver at all.

A design is reported as verified when, for every one of its removals,

    |lambda_loc(LP) - lambda_full(LP)| <= tol * lambda_full(LP)

with lambda_loc from `kinematic(local=True)` and lambda_full from `kinematic()`.
A counterexample is any removal where the LP pair separates while the chain
certificate called it exact, or vice versa.
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
TOL = 1e-7


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
    f = Frame(f"verify_H{H}_B{B}", spans, heights, members, loads.tolist(),
              provenance="independent LP verification of the uniform result")
    f.validate()
    return f


def main() -> None:
    pairs = [("W18X40", "W14X109"), ("W21X50", "W14X132"),
             ("W24X62", "W14X176"), ("HN400X200", "HW350X350")]
    rows = []
    n_checked = 0
    n_agree = 0
    n_counter = 0
    for B in (1, 2, 3):
        for H in (2, 3, 4, 6):
            for beam, column in pairs:
                f = build(H, B, 6.0, 4.0, beam, column)
                for removal in f.scenarios():
                    sc = Scenario(f, removal)
                    x = np.ones(f.E)
                    margin = float(min_interval_margin(sc, x))
                    chain_full = float(sc.chain(x)["capacity"])
                    chain_loc = float(sc.chain(x, local=True)["capacity"])
                    lp_full = float(sc.kinematic(x)["capacity"])
                    lp_loc = float(sc.kinematic(x, local=True)["capacity"])
                    lp_gap = lp_loc - lp_full
                    chain_gap = chain_loc - chain_full
                    tol = TOL * max(1.0, abs(lp_full))
                    lp_exact = abs(lp_gap) <= tol
                    cert_exact = margin >= -1e-8
                    n_checked += 1
                    if lp_exact == cert_exact:
                        n_agree += 1
                    else:
                        n_counter += 1
                    rows.append({
                        "H": H, "B": B, "removal": list(removal),
                        "kappa": Mp(beam) / Mp(column),
                        "margin": margin, "cert_exact": cert_exact,
                        "lp_gap": lp_gap, "lp_exact": lp_exact,
                        "chain_gap": chain_gap,
                        "lp_vs_chain_full": abs(lp_full - chain_full),
                    })

    print(f"removals checked                                  : {n_checked}")
    print(f"chain certificate and independent LP agree        : {n_agree}")
    print(f"disagreements (potential counterexamples)         : {n_counter}")

    multi = [r for r in rows if r["B"] >= 2]
    print(f"\nB >= 2 removals                                   : {len(multi)}")
    print(f"  of which the independent LP calls inexact       : "
          f"{sum(1 for r in multi if not r['lp_exact'])}")
    print(f"  max |lp_loc - lp_full| / lp_full                : "
          f"{max((abs(r['lp_gap'])/max(r['lp_gap']+1e-30,1e-30) for r in multi), default=0.0):.3e}"
          if False else
          f"  max |lp_loc - lp_full|                          : "
          f"{max(abs(r['lp_gap']) for r in multi):.3e}")
    print(f"  max |lp_full - chain_full|                      : "
          f"{max(r['lp_vs_chain_full'] for r in multi):.3e}")

    single = [r for r in rows if r["B"] == 1]
    print(f"\nB = 1 removals                                    : {len(single)}")
    print(f"  independent LP calls inexact                    : "
          f"{sum(1 for r in single if not r['lp_exact'])} "
          f"({100.0*sum(1 for r in single if not r['lp_exact'])/max(len(single),1):.1f}%)")

    dest = PKG / "data" / "independent_lp_verification.json"
    dest.write_text(json.dumps({"checked": n_checked, "agree": n_agree,
                                "disagreements": n_counter,
                                "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
