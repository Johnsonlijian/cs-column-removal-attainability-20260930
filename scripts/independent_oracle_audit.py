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


def _lp_removal(H: int, B: int, ordinal: int) -> tuple[int, int]:
    """Deterministic boundary/interior coverage for the independent LP lane."""
    if ordinal % 3 == 0:
        return 1, 0
    if ordinal % 3 == 1:
        return H, B
    return (1 if B == 1 else H), (0 if B == 1 else B // 2)


def _lp_record(sc: Scenario, x: np.ndarray, frame_name: str, treatment: str,
               removal: tuple[int, int], clean_frame: bool) -> dict:
    chain = sc.chain(x, local=False)["capacity"]
    kin = sc.kinematic(x, local=False)["capacity"]
    stat = sc.static(x, domain="strip")["capacity"]
    return {
        "frame": frame_name,
        "treatment": treatment,
        "removal": [int(removal[0]), int(removal[1])],
        "clean_frame": bool(clean_frame),
        "chain": float(chain),
        "kinematic": float(kin),
        "static": float(stat),
        "chain_kinematic_objective_gap": float(abs(chain - kin)),
        "chain_static_objective_gap": float(abs(chain - stat)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    clean_frames = 0
    frames_seen = 0
    instances = 0
    max_full = 0.0
    max_local = 0.0
    max_product_log_sum = 0.0
    static_checks = []
    lp_pair_quota = {(2, 1): 3, (2, 2): 3, (2, 4): 3,
                     (4, 1): 2, (4, 2): 2, (8, 3): 3, (12, 4): 2}
    lp_pair_count = {key: 0 for key in lp_pair_quota}
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
                        frame_clean = bool(np.all(clean))
                        pair = (H, B)
                        # Include non-clean baseline cases in the independent
                        # LP lane. This is separate from the conditional
                        # transition denominator.
                        if (not frame_clean and pair in lp_pair_quota
                                and lp_pair_count[pair] < lp_pair_quota[pair]
                                and len(static_checks) < 18):
                            removal = _lp_removal(H, B, lp_pair_count[pair])
                            static_checks.append(_lp_record(
                                Scenario(f, removal), np.ones(f.E), f.name,
                                "baseline", removal, False))
                            lp_pair_count[pair] += 1
                        if not frame_clean:
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
                            pair = (H, B)
                            if (len(static_checks) < 18
                                    and treatment != "homogeneous_k2"
                                    and pair in lp_pair_quota
                                    and lp_pair_count[pair] < lp_pair_quota[pair]):
                                removal = _lp_removal(H, B, lp_pair_count[pair])
                                static_checks.append(_lp_record(
                                    Scenario(f, removal), x, f.name,
                                    treatment, removal, True))
                                lp_pair_count[pair] += 1
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
        "lp_max_chain_kinematic_objective_gap": max((x["chain_kinematic_objective_gap"] for x in static_checks), default=0.0),
        "lp_max_chain_static_objective_gap": max((x["chain_static_objective_gap"] for x in static_checks), default=0.0),
        "lp_coverage": {
            "checks": len(static_checks),
            "clean_frame_checks": sum(bool(x["clean_frame"]) for x in static_checks),
            "nonclean_frame_checks": sum(not bool(x["clean_frame"]) for x in static_checks),
            "by_height_bay": {f"{H}x{B}": int(lp_pair_count[(H, B)]) for H, B in lp_pair_quota},
        },
        "note": "Separate per-scenario Scenario chain audit against the batched sweep; LP entries report objective-value differences between the chain and independent kinematic/static programs, not equilibrium residuals, inequality violations, or primal-dual gaps.",
    }
    (DATA / "independent_oracle_audit.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    if out["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

