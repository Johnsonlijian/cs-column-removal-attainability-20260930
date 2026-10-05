"""Paired control for the model's positive story-height invariance.

Within the stated first-order chord-rate model, changing only positive story
heights changes recovered horizontal displacements u_r but not the dimensionless
limit values or interval margins. This script checks that consequence directly
for paired geometry labels using identical capacities, loads, treatments and
seeds.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
CODE = PACKAGE / "scripts" / "code"
sys.path.insert(0, str(CODE))
from engine import Scenario  # noqa: E402
from batch_sweep import BatchSweep  # noqa: E402
from final_capacity_pattern_gate import controlled_frame, treatment_vectors, H_VALUES, B_VALUES, REGIMES  # noqa: E402

SEED = 20261005
PAIRS = (("equal", "unequal_height"), ("unequal_bay", "combined"))

def margins(frame, x):
    out = []
    for removal in frame.scenarios():
        sc = Scenario(frame, removal)
        best = float("inf")
        for _lo, _hi, t, q in sc.bands:
            V, B0, _ = sc._potentials(x, t, q)
            entry = np.empty(frame.H)
            entry[0] = B0
            if frame.H > 1:
                entry[1:] = V[:-1, 0, 1] - V[:-1, 0, 0]
            interior = V[:, 1, 1] - V[:, 0, 0]
            exit_cost = V[:, 1, 0] - V[:, 0, 0]
            pref = np.empty(frame.H)
            pref[0] = entry[0]
            for i in range(1, frame.H):
                pref[i] = min(pref[i-1] + interior[i-1], entry[i])
            best = min(best, float(np.min(pref + exit_cost)))
        out.append(best)
    return np.asarray(out)


def main():
    rng = np.random.default_rng(SEED)
    max_full = max_local = max_margin = 0.0
    checked = 0
    records = []
    for H in H_VALUES:
        for B in B_VALUES:
            for regime in REGIMES:
                for _draw in range(12):
                    seed = int(rng.integers(0, 2**32 - 1))
                    for left, right in PAIRS:
                        f0 = controlled_frame(H, B, seed, regime, left)
                        f1 = controlled_frame(H, B, seed, regime, right)
                        vecs = treatment_vectors(f0, np.random.default_rng(seed ^ 0x5A17))
                        for treatment, x in vecs.items():
                            b0, b1 = BatchSweep(f0, x), BatchSweep(f1, x)
                            full0, full1 = b0.capacities(), b1.capacities()
                            loc0, loc1 = b0.capacities(local=True), b1.capacities(local=True)
                            mar0, mar1 = margins(f0, x), margins(f1, x)
                            df = float(np.max(np.abs(full0-full1)))
                            dl = float(np.max(np.abs(loc0-loc1)))
                            dm = float(np.max(np.abs(mar0-mar1)))
                            max_full = max(max_full, df); max_local = max(max_local, dl); max_margin = max(max_margin, dm)
                            checked += 1
                            if len(records) < 6:
                                records.append({"H":H,"B":B,"regime":regime,"seed":seed,"pair":[left,right],"treatment":treatment,"max_full_abs":df,"max_local_abs":dl,"max_margin_abs":dm})
    tol = 1.0e-10
    status = "PASS" if max(max_full, max_local, max_margin) <= tol else "FAIL"
    payload={"status":status,"tolerance":tol,"seed":SEED,"pairs":PAIRS,"checked_pair_treatments":checked,"max_full_abs_difference":max_full,"max_local_abs_difference":max_local,"max_margin_abs_difference":max_margin,"records":records,"interpretation":"Positive story-height changes leave dimensionless capacities, local/full values and interval margins invariant in the stated first-order chord-rate model; recovered horizontal displacements may change."}
    out=PACKAGE/"data"/"geometry_invariance_check.json"
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
if __name__ == "__main__": main()
