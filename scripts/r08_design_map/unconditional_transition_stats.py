"""Unconditional transition statistics for the frozen ensemble.

The manuscript reports transition fractions conditional on frames that were
exact at every removal before the intervention ("baseline-clean"). The four-model
review panel objected that this can read as denominator shopping, and asked for
the unconditional tabulation as well. This script produces it.

Definitions
-----------
For each frame and each treatment, a frame is classified by the exactness of its
own removal set before and after the intervention:

    exact -> exact      (EE)   screen stays valid
    exact -> inexact    (EI)   intervention opens a gap  [the manuscript's headline]
    inexact -> exact    (IE)   intervention closes a gap
    inexact -> inexact  (II)   screen was already invalid and remains so

"Exact" here means the certificate margin is non-negative at every removal
position of that frame. The unconditional table is over all generated frames;
the conditional table restricts the denominator to the EE+EI row, which is the
manuscript's baseline-clean filter.
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

from engine import Scenario  # noqa: E402
from final_capacity_pattern_gate import (  # noqa: E402
    B_VALUES, DRAW_COUNT, GEOMETRIES, H_VALUES, REGIMES, SEED, TREATMENTS,
    controlled_frame, min_interval_margin, treatment_vectors,
)


def exact_mask(frame, x) -> np.ndarray:
    """Boolean array over removals: is the certificate non-negative?"""
    s, g = frame.H, frame.B
    out = np.zeros((s, g + 1), dtype=bool)
    for removal in frame.scenarios():
        sc = Scenario(frame, removal)
        out[removal[0] - 1, removal[1]] = min_interval_margin(sc, x) >= -1e-8
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    treatments = [t for t in TREATMENTS]
    counts = {t: {"EE": 0, "EI": 0, "IE": 0, "II": 0} for t in treatments}
    frames = 0

    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    for _draw in range(DRAW_COUNT):
                        seed = int(rng.integers(0, 2**32 - 1))
                        f = controlled_frame(H, B, seed, regime, mode)
                        base = exact_mask(f, np.ones(f.E))
                        base_all = bool(base.all())
                        vecs = treatment_vectors(f, rng)
                        frames += 1
                        for t, x in vecs.items():
                            after = bool(exact_mask(f, x).all())
                            if base_all and after:
                                counts[t]["EE"] += 1
                            elif base_all and not after:
                                counts[t]["EI"] += 1
                            elif not base_all and after:
                                counts[t]["IE"] += 1
                            else:
                                counts[t]["II"] += 1

    out = {"seed": SEED, "frames_generated": frames, "tables": {}}
    print(f"frames generated: {frames}\n")
    header = (f"{'treatment':<34s} {'EE':>5s} {'EI':>5s} {'IE':>5s} "
              f"{'II':>5s} | {'uncond EI%':>10s} {'cond EI%':>9s}")
    print(header)
    print("-" * len(header))
    for t in treatments:
        c = counts[t]
        tot = sum(c.values())
        uncond = 100.0 * c["EI"] / tot if tot else float("nan")
        denom = c["EE"] + c["EI"]
        cond = 100.0 * c["EI"] / denom if denom else float("nan")
        out["tables"][t] = {**c, "total": tot,
                            "unconditional_EI_pct": uncond,
                            "conditional_EI_pct": cond,
                            "baseline_clean": denom}
        print(f"{t:<34s} {c['EE']:>5d} {c['EI']:>5d} {c['IE']:>5d} "
              f"{c['II']:>5d} | {uncond:>10.2f} {cond:>9.2f}")

    dest = PKG / "data" / "unconditional_transition_table.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", dest)

    print("\nInterpretation for the manuscript:")
    for t in treatments:
        c = counts[t]
        tot = sum(c.values())
        print(f"  {t}: of all {tot} frames, {c['EE']} stay exact, {c['EI']} "
              f"lose exactness, {c['IE']} gain it, {c['II']} were already "
              f"inexact and remain so.")
    loss = counts["componentwise_strengthening"]["EI"]
    gain = counts["componentwise_strengthening"]["IE"]
    print(f"\n  Under memberwise strengthening the screen loses exactness in "
          f"{loss} frames and regains it in {gain}; the intervention is "
          f"therefore not a one-way degradation.")


if __name__ == "__main__":
    main()
