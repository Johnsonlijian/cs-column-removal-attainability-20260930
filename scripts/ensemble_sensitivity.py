"""Ensemble sensitivity and convergence study for the capacity-pattern gate.

The manuscript reports conditional transition fractions from a single frozen
ensemble (master seed 20260925, 12 draws per group).  A reviewer will ask how
much those fractions depend on the seed and on the ensemble size.  This script
re-runs the *identical* generator and the *identical* treatments at

* several independent master seeds at the frozen draw count, and
* increasing draw counts at the frozen master seed,

and reports the spread of the three headline statistics.  It writes
``data/ensemble_sensitivity.json`` and never overwrites the frozen gate archive.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from final_capacity_pattern_gate import (  # noqa: E402
    DRAW_COUNT,
    SEED,
    main as run_gate,
)

ALT_SEEDS = (20260926, 20260927, 20260928, 20260929)
CONVERGENCE_DRAWS = (3, 6, 12, 24, 48)
STATISTICS = ("homogeneous_k2", "componentwise_strengthening",
              "geometric_mean_one_redistribution")


def extract(payload: dict) -> dict:
    agg = payload["aggregate"]
    clean = agg["homogeneous_k2"]["baseline_clean_frames"]
    out = {
        "frames": payload["frame_records"],
        "baseline_clean_frames": clean,
        "baseline_clean_fraction": clean / max(payload["frame_records"], 1),
        "margin_sample_n": payload["margin_sample"]["n"],
        "spearman_minus_margin_gap": payload["margin_sample"]["spearman_minus_margin_gap"],
    }
    for key in STATISTICS:
        out[f"{key}__frame_fraction"] = agg[key]["frame_fraction_with_new_gap"]
        out[f"{key}__instance_fraction"] = agg[key]["gap_instance_fraction"]
        out[f"{key}__max_relative_gap"] = agg[key]["max_relative_gap"]
    return out


def main() -> None:
    started = time.perf_counter()
    runs = []

    frozen = extract(run_gate(seed=SEED, draw_count=DRAW_COUNT,
                              out_path=DATA / "ensemble_sensitivity_seed20260925.json",
                              quiet=True))
    runs.append({"kind": "frozen", "seed": SEED, "draw_count": DRAW_COUNT, **frozen})

    for seed in ALT_SEEDS:
        payload = run_gate(seed=seed, draw_count=DRAW_COUNT,
                           out_path=DATA / f"ensemble_sensitivity_seed{seed}.json",
                           quiet=True)
        runs.append({"kind": "alternate_seed", "seed": seed,
                     "draw_count": DRAW_COUNT, **extract(payload)})

    for draws in CONVERGENCE_DRAWS:
        if draws == DRAW_COUNT:
            continue
        payload = run_gate(seed=SEED, draw_count=draws,
                           out_path=DATA / f"ensemble_sensitivity_draws{draws}.json",
                           quiet=True)
        runs.append({"kind": "draw_count", "seed": SEED, "draw_count": draws,
                     **extract(payload)})

    summary: dict[str, dict] = {}
    for key in STATISTICS:
        for metric in ("frame_fraction", "instance_fraction"):
            field = f"{key}__{metric}"
            seed_values = np.array([r[field] for r in runs if r["kind"] != "draw_count"])
            draw_values = np.array([r[field] for r in runs if r["kind"] == "draw_count"]
                                   + [r[field] for r in runs if r["kind"] == "frozen"])
            summary[field] = {
                "frozen": float(next(r[field] for r in runs if r["kind"] == "frozen")),
                "alternate_seed_min": float(seed_values.min()),
                "alternate_seed_max": float(seed_values.max()),
                "alternate_seed_mean": float(seed_values.mean()),
                "alternate_seed_std": float(seed_values.std(ddof=1)) if seed_values.size > 1 else 0.0,
                "alternate_seed_range": float(seed_values.max() - seed_values.min()),
                "draw_count_min": float(draw_values.min()),
                "draw_count_max": float(draw_values.max()),
            }

    payload = {
        "status": "PASS",
        "frozen_reference": {"seed": SEED, "draw_count": DRAW_COUNT},
        "alternate_seeds": list(ALT_SEEDS),
        "convergence_draw_counts": list(CONVERGENCE_DRAWS),
        "runs": runs,
        "summary": summary,
        "seconds": round(time.perf_counter() - started, 2),
        "boundary": (
            "These runs re-use the identical synthetic generator, treatments and "
            "baseline-clean precondition. The spread quantifies Monte Carlo "
            "sensitivity of the reported conditional fractions to the master seed "
            "and to the ensemble size; it says nothing about real building "
            "populations."
        ),
    }
    (DATA / "ensemble_sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"seconds={payload['seconds']}")
    for field, stats in summary.items():
        print(f"{field:<58} frozen={stats['frozen']:.4f} "
              f"seed_range=[{stats['alternate_seed_min']:.4f}, "
              f"{stats['alternate_seed_max']:.4f}] "
              f"draw_range=[{stats['draw_count_min']:.4f}, {stats['draw_count_max']:.4f}]")


if __name__ == "__main__":
    main()
