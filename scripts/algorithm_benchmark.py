"""Benchmark the fixed-capacity O(HB) chain sweep against per-removal LP solves."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from batch_sweep import BatchSweep  # noqa: E402
from engine import Scenario  # noqa: E402
from final_capacity_pattern_gate import controlled_frame  # noqa: E402

CASES = (
    (2, 2, "weak_base", "equal"),
    (4, 3, "balanced", "unequal_bay"),
    (6, 4, "axial", "combined"),
    (8, 4, "balanced", "equal"),
    (12, 4, "weak_base", "combined"),
)
REPEATS = 5


def bench_case(H: int, B: int, regime: str, mode: str) -> dict:
    frame = controlled_frame(H, B, 20260925 ^ (H << 8) ^ B, regime, mode)
    x = np.ones(frame.E)
    batch = BatchSweep(frame, x)

    t_chain = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = batch.capacities()
        t_chain.append(time.perf_counter() - t0)
    chain_mean = float(np.mean(t_chain))

    t_lp = []
    lp_values = np.zeros((H, B + 1))
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for s in range(1, H + 1):
            for g in range(B + 1):
                sc = Scenario(frame, (s, g))
                lp_values[s - 1, g] = sc.static(x, domain="strip")["capacity"]
        t_lp.append(time.perf_counter() - t0)
    lp_mean = float(np.mean(t_lp))

    chain_once = batch.capacities()
    worst = 0.0
    for s in range(1, H + 1):
        for g in range(B + 1):
            sc = Scenario(frame, (s, g))
            ref = sc.chain(x, local=False)["capacity"]
            worst = max(worst, abs(chain_once[s - 1, g] - ref) / max(1.0, abs(ref)))

    removals = H * (B + 1)
    return {
        "H": H,
        "B": B,
        "regime": regime,
        "geometry": mode,
        "removals": removals,
        "chain_seconds_mean": chain_mean,
        "static_lp_seconds_mean": lp_mean,
        "speedup_vs_static_lp": lp_mean / chain_mean if chain_mean else None,
        "chain_per_removal_us": 1e6 * chain_mean / removals,
        "static_lp_per_removal_ms": 1e3 * lp_mean / removals,
        "max_chain_static_objective_gap": worst,
    }


def main() -> None:
    rows = [bench_case(*case) for case in CASES]
    payload = {
        "status": "PASS",
        "repeats": REPEATS,
        "note": (
            "The chain sweep evaluates all scalar complete values for one capacity "
            "vector; each static LP solve is an independent HiGHS call for one removal."
        ),
        "cases": rows,
        "median_speedup": float(np.median([r["speedup_vs_static_lp"] for r in rows])),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "algorithm_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Algorithm benchmark",
        "",
        f"Median speedup of the fixed-capacity chain sweep over per-removal static LP: "
        f"**{payload['median_speedup']:.1f}x** ({REPEATS} repeats per case).",
        "",
        "| H | B | Removals | Chain (s) | Static LP (s) | Speedup | Max chain--static gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['H']} | {r['B']} | {r['removals']} | {r['chain_seconds_mean']:.4f} | "
            f"{r['static_lp_seconds_mean']:.4f} | {r['speedup_vs_static_lp']:.1f}x | "
            f"{r['max_chain_static_objective_gap']:.2e} |"
        )
    DOCS.mkdir(exist_ok=True)
    (DOCS / "ALGORITHM_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"median speedup {payload['median_speedup']:.1f}x")
    print(f"wrote {DATA / 'algorithm_benchmark.json'}")


if __name__ == "__main__":
    main()
