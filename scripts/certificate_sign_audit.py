"""Population certificate sign audit on the frozen gate stream.

For every eligible removal of every baseline-clean frame, compare the sign of
the minimum interval margin with the sign of the local-complete gap. The random
stream matches ``final_capacity_pattern_gate.main``, including the archived
margin-sample draws, so the population is the frozen one.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from engine import Scenario  # noqa: E402
from batch_sweep import BatchSweep  # noqa: E402
from final_capacity_pattern_gate import (  # noqa: E402
    B_VALUES,
    DRAW_COUNT,
    GEOMETRIES,
    H_VALUES,
    REGIMES,
    SEED,
    TREATMENTS,
    controlled_frame,
    min_interval_margin,
    treatment_vectors,
)


def main() -> None:
    rng = np.random.default_rng(SEED)
    stats = {
        t: {"n": 0, "agree": 0, "margin_neg_gap_pos": 0, "margin_pos_gap_pos": 0,
            "margin_neg_gap_nonpos": 0, "max_abs_disagreement_gap": 0.0}
        for t in TREATMENTS if t != "homogeneous_k2"
    }
    t0 = time.perf_counter()
    clean_frames = 0
    archived_draws = 0
    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    for _draw in range(DRAW_COUNT):
                        seed = int(rng.integers(0, 2**32 - 1))
                        f = controlled_frame(H, B, seed, regime, mode)
                        base = BatchSweep(f, np.ones(f.E))
                        full0 = base.capacities()
                        local0 = base.capacities(local=True)
                        clean = np.abs(local0 - full0) <= np.maximum(1.0e-8, 1.0e-8 * np.abs(local0))
                        if not bool(np.all(clean)):
                            continue
                        clean_frames += 1
                        vecs = treatment_vectors(f, rng)
                        for treatment, x in vecs.items():
                            batch = BatchSweep(f, x)
                            full = batch.capacities()
                            local = batch.capacities(local=True)
                            if treatment != "homogeneous_k2" and archived_draws < 6000:
                                # Same overshoot rule as the frozen gate: the
                                # three-draw block is entered whenever the archive
                                # is still below 6,000, and it is allowed to finish.
                                for _ in range(min(3, H * (B + 1))):
                                    rng.integers(1, H + 1)
                                    rng.integers(0, B + 1)
                                    archived_draws += 1
                            if treatment == "homogeneous_k2":
                                continue
                            slot = stats[treatment]
                            for s in range(1, H + 1):
                                for g in range(B + 1):
                                    if not clean[s - 1, g]:
                                        continue
                                    sc = Scenario(f, (s, g))
                                    margin = min_interval_margin(sc, x)
                                    gap = float(local[s - 1, g] - full[s - 1, g])
                                    tol = 1.0e-8 * max(1.0, abs(float(local[s - 1, g])))
                                    gap_pos = gap > tol
                                    margin_neg = margin < -1.0e-8
                                    slot["n"] += 1
                                    if margin_neg == gap_pos:
                                        slot["agree"] += 1
                                    else:
                                        slot["max_abs_disagreement_gap"] = max(
                                            slot["max_abs_disagreement_gap"], abs(gap)
                                        )
                                    if margin_neg and gap_pos:
                                        slot["margin_neg_gap_pos"] += 1
                                    elif (not margin_neg) and gap_pos:
                                        slot["margin_pos_gap_pos"] += 1
                                    elif margin_neg and (not gap_pos):
                                        slot["margin_neg_gap_nonpos"] += 1
    elapsed = time.perf_counter() - t0
    payload = {
        "status": "PASS" if all(v["agree"] == v["n"] and v["n"] > 0 for v in stats.values()) else "FAIL",
        "baseline_clean_frames": clean_frames,
        "elapsed_seconds": round(elapsed, 2),
        "tolerance": "gap > 1e-8*max(1,|local|) versus margin < -1e-8",
        "treatments": stats,
    }
    out = ROOT / "data" / "certificate_sign_audit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: {"n": v["n"], "agree": v["agree"]} for k, v in stats.items()}, indent=2))
    print("status", payload["status"], "frames", clean_frames, f"{elapsed:.1f}s")


if __name__ == "__main__":
    main()
