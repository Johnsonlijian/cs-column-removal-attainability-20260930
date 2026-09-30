"""Canonical monotone-strengthening counterexample and independent LP check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from engine import Frame, Scenario  # noqa: E402
from batch_sweep import BatchSweep  # noqa: E402


def frame():
    f = Frame(
        "canonical_H2_B1_equal_baseline",
        [1.0], [1.0, 1.0],
        [
            ["b", 1, 0, 1.0, 1.0, 8.0, 1.0],
            ["b", 2, 0, 1.0, 1.0, 8.0, 1.0],
            ["c", 1, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 1, 1, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 1, 5.0, 5.0, 50.0, 1.0],
        ],
        [[0.5, 0.5], [0.5, 0.5]],
        provenance="analytical counterexample; dimensionless canonical frame",
    )
    f.validate()
    return f


def run(k_values=(1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)):
    f = frame()
    rows = []
    for k in k_values:
        x = np.ones(f.E)
        for i, member in enumerate(f.members):
            # Leave one first-story exterior base column unchanged; increase
            # every beam and every other column by k.
            if not (member[0] == "c" and member[1] == 1 and member[2] == 0):
                x[i] = k
        batch = BatchSweep(f, x)
        complete = batch.capacities()
        local = batch.capacities(local=True)
        row = {"k": k, "n_fail": int(np.sum(local - complete > 1e-8)),
               "min_ratio": float(np.min(complete / local)),
               "max_gap": float(np.max(local - complete)),
               "batch_max_abs": float(np.max(np.abs(complete - batch.capacities())))}
        # Direct chain, kinematic and static checks for all four removals.
        checks = []
        for removal in f.scenarios():
            sc = Scenario(f, removal)
            chain = float(sc.chain(x)["capacity"])
            kin = float(sc.kinematic(x)["capacity"])
            stat = float(sc.static(x)["capacity"])
            loc = float(sc.chain(x, local=True)["capacity"])
            checks.append({"removal": list(removal), "chain": chain,
                           "kinematic": kin, "static": stat, "local": loc,
                           "static_duality_residual": float(sc.static(x)["duality_res"])})
        row["checks"] = checks
        rows.append(row)
    return {"model": "H=2, B=1; all baseline beam/column end moments (1, 5)",
            "mechanical_prediction": "for removal (1,1), local=4k and complete=min(4k,5)",
            "rows": rows}


if __name__ == "__main__":
    out = HERE.parent / "data" / "canonical_case.json"
    out.write_text(json.dumps(run(), indent=2), encoding="utf-8")
    print(out)


