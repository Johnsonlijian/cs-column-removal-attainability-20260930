from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from final_capacity_pattern_gate import (
    B_VALUES,
    DRAW_COUNT,
    GEOMETRIES,
    H_VALUES,
    REGIMES,
    SEED,
    controlled_frame,
    treatment_vectors,
)
from engine import Scenario
from batch_sweep import BatchSweep


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
REL_TOL = 2.0e-8


def compare(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def main() -> None:
    rng = np.random.default_rng(SEED)
    clean_frames = 0
    frames_seen = 0
    instances = 0
    max_full = 0.0
    max_local = 0.0
    max_product_log_sum = 0.0
    static_checks = []
    margin_count = 0
    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    for draw in range(DRAW_COUNT):
                        seed = int(rng.integers(0, 2**32 - 1))
                        f = controlled_frame(H, B, seed, regime, mode)
                        bs0 = BatchSweep(f, np.ones(f.E))
                        base_full = bs0.capacities()
                        base_local = bs0.capacities(local=True)
                        clean = np.abs(base_full - base_local) <= np.maximum(1.0e-8, 1.0e-8 * np.abs(base_local))
                        frames_seen += 1
                        if not bool(np.all(clean)):
                            continue
                        clean_frames += 1
                        vecs = treatment_vectors(f, rng)
                        for treatment, x in vecs.items():
                            if treatment == "geometric_mean_one_redistribution":
                                max_product_log_sum = max(max_product_log_sum, abs(float(np.sum(np.log(x)))))
                            bs = BatchSweep(f, x)
                            fast_full = bs.capacities()
                            fast_local = bs.capacities(local=True)
                            slow_full = np.zeros_like(fast_full)
                            slow_local = np.zeros_like(fast_local)
                            for s in range(1, H + 1):
                                for g in range(B + 1):
                                    sc = Scenario(f, (s, g))
                                    slow_full[s - 1, g] = sc.chain(x, local=False)["capacity"]
                                    slow_local[s - 1, g] = sc.chain(x, local=True)["capacity"]
                                    instances += 1
                            max_full = max(max_full, compare(fast_full, slow_full))
                            max_local = max(max_local, compare(fast_local, slow_local))
                            if len(static_checks) < 18 and treatment != "homogeneous_k2":
                                # Stratified independent LP checks; these are
                                # deliberately not the selected best witness.
                                s = 1 + (len(static_checks) % H)
                                g = len(static_checks) % (B + 1)
                                sc = Scenario(f, (s, g))
                                chain = sc.chain(x, local=False)["capacity"]
                                kin = sc.kinematic(x, local=False)["capacity"]
                                stat = sc.static(x, domain="strip")["capacity"]
                                static_checks.append({
                                    "frame": f.name,
                                    "treatment": treatment,
                                    "removal": [s, g],
                                    "chain": float(chain),
                                    "kinematic": float(kin),
                                    "static": float(stat),
                                    "chain_kinematic_residual": float(abs(chain - kin)),
                                    "chain_static_residual": float(abs(chain - stat)),
                                })
                        # Preserve the original gate's random stream: these
                        # draws generated the bounded margin sample and affect
                        # subsequent frame seeds.
                        for treatment in ("componentwise_strengthening", "geometric_mean_one_redistribution"):
                            if margin_count < 6000:
                                n = min(3, H * (B + 1))
                                for _ in range(n):
                                    rng.integers(1, H + 1)
                                    rng.integers(0, B + 1)
                                margin_count += n

    out = {
        "status": "PASS" if max_full <= REL_TOL and max_local <= REL_TOL and max_product_log_sum <= 1e-12 else "FAIL",
        "frames_seen": frames_seen,
        "baseline_clean_frames": clean_frames,
        "audited_instances": instances,
        "max_batch_vs_scenario_full_relative_error": max_full,
        "max_batch_vs_scenario_local_relative_error": max_local,
        "max_abs_log_product_error": max_product_log_sum,
        "lp_checks": static_checks,
        "lp_max_chain_kinematic_residual": max((x["chain_kinematic_residual"] for x in static_checks), default=0.0),
        "lp_max_chain_static_residual": max((x["chain_static_residual"] for x in static_checks), default=0.0),
        "note": "Separate per-scenario Scenario chain audit against the batched sweep; the LP checks use independent kinematic and static formulations.",
    }
    (DATA / "independent_oracle_audit.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    if out["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

