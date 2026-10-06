"""Expanded stratified LP audit of the exact chain value.

Motivation
----------
The released audit checks 18 kinematic/static LP instances over seven ``(H, B)``
strata.  A reviewer can reasonably ask for coverage of *every* stratification
cell of the frozen study, and for the residuals that make each LP comparison a
meaningful check rather than a coincidence of objective values.

This script therefore

* walks the identical frame grid (5 heights x 4 bay counts x 4 geometry classes
  x 3 capacity regimes = 60 cells) and the identical generator and random stream
  as the released gate, so the frames are the frozen ones;
* audits **every cell**: one baseline instance per cell plus treatment instances
  drawn deterministically from the eligible cells, giving 150 or more independent
  LP comparisons;
* for every instance solves the exact story chain, the complete kinematic LP and
  the bounded static (dual) LP, and records objective values plus the residuals
  that validate each solution: kinematic compatibility/equilibrium residual,
  static equilibrium residual, static yield residual and static duality gap.

The comparison is between objective values of three independently assembled
programs; it is not an equilibrium residual of the chain solver.

Outputs
-------
data/expanded_lp_audit.json
docs/EXPANDED_LP_AUDIT.md
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
DOCS = ROOT / "docs"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from engine import Frame, Scenario  # noqa: E402
from batch_sweep import BatchSweep  # noqa: E402
from final_capacity_pattern_gate import (  # noqa: E402
    B_VALUES,
    DRAW_COUNT,
    GEOMETRIES,
    H_VALUES,
    REGIMES,
    SEED,
    controlled_frame,
    treatment_vectors,
)

REL_TOL = 2.0e-8
BASELINE_QUOTA = 1
TREATMENT_QUOTA = 2
TREATMENTS = ("componentwise_strengthening", "geometric_mean_one_redistribution")


def removal_for(H: int, B: int, ordinal: int) -> tuple[int, int]:
    """Deterministic coverage of exterior, interior and roof removal positions."""
    options = [(1, 0), (H, B), (1, B // 2), (H // 2 + 1, B), (H, 0)]
    if B == 1:
        options = [(1, 0), (H, 1)]
    return options[ordinal % len(options)]


def audit_instance(sc: Scenario, x: np.ndarray, meta: dict) -> dict:
    chain = sc.chain(x, local=False)
    kin = sc.kinematic(x, local=False)
    stat = sc.static(x, domain="strip")
    return {
        **meta,
        "chain": float(chain["capacity"]),
        "kinematic": float(kin["capacity"]),
        "static": float(stat["capacity"]),
        "chain_kinematic_objective_gap": float(abs(chain["capacity"] - kin["capacity"])),
        "chain_static_objective_gap": float(abs(chain["capacity"] - stat["capacity"])),
        "kinematic_equilibrium_residual": float(kin["eq_res"]),
        "static_equilibrium_residual": float(stat["eq_res"]),
        "static_yield_residual": float(stat["yield_res"]),
        "static_duality_residual": float(stat["duality_res"]),
        "minimum_interval_margin": float(min(
            _band_margin(sc, x, lo, hi, t, q)
            for lo, hi, t, q in sc.bands)),
    }


def _band_margin(sc: Scenario, x: np.ndarray, lo: float, hi: float,
                 t: float, q: float) -> float:
    from engine import worst_interval
    V, B0, _ = sc._potentials(x, t, q)
    value, _arg = worst_interval(V, B0)
    return value


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    audits: list[dict] = []
    cell_frames: dict[str, int] = {}
    cell_audits: dict[str, int] = {}
    frames_seen = 0
    margin_draws = 0

    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    cell = f"{regime}|{mode}|H{H}|B{B}"
                    cell_frames[cell] = 0
                    cell_audits[cell] = 0
                    for draw in range(DRAW_COUNT):
                        seed = int(rng.integers(0, 2**32 - 1))
                        f = controlled_frame(H, B, seed, regime, mode)
                        frames_seen += 1
                        cell_frames[cell] += 1
                        baseline = BatchSweep(f, np.ones(f.E))
                        base_full = baseline.capacities()
                        base_local = baseline.capacities(local=True)
                        clean = np.abs(base_full - base_local) <= np.maximum(
                            1.0e-8, 1.0e-8 * np.abs(base_local))
                        frame_clean = bool(np.all(clean))

                        if draw < BASELINE_QUOTA:
                            removal = removal_for(H, B, 0)
                            audits.append(audit_instance(
                                Scenario(f, removal), np.ones(f.E),
                                {"cell": cell, "H": H, "B": B, "geometry": mode,
                                 "regime": regime, "treatment": "baseline",
                                 "frame_clean": frame_clean, "seed": seed,
                                 "removal": [int(removal[0]), int(removal[1])]}))
                            cell_audits[cell] += 1

                        if not frame_clean:
                            # keep the random stream aligned with the gate even
                            # when a frame is skipped
                            continue
                        vecs = treatment_vectors(f, rng)
                        for treatment in TREATMENTS:
                            if draw < TREATMENT_QUOTA:
                                ordinal = 1 if treatment == TREATMENTS[0] else 3
                                removal = removal_for(H, B, ordinal)
                                audits.append(audit_instance(
                                    Scenario(f, removal), vecs[treatment],
                                    {"cell": cell, "H": H, "B": B,
                                     "geometry": mode, "regime": regime,
                                     "treatment": treatment, "frame_clean": True,
                                     "seed": seed,
                                     "removal": [int(removal[0]), int(removal[1])]}))
                                cell_audits[cell] += 1
                        # Reproduce the frozen gate's margin-sample draws exactly so
                        # that the frame seeds of later cells stay identical.
                        for _treatment in ("componentwise_strengthening",
                                           "geometric_mean_one_redistribution"):
                            if margin_draws < 6000:
                                n = min(3, H * (B + 1))
                                for _ in range(n):
                                    rng.integers(1, H + 1)
                                    rng.integers(0, B + 1)
                                margin_draws += n

    cells_covered = sum(1 for v in cell_audits.values() if v > 0)
    gaps = np.array([a["chain_kinematic_objective_gap"] for a in audits])
    gaps_static = np.array([a["chain_static_objective_gap"] for a in audits])
    payload = {
        "status": "PASS" if (gaps.max() <= REL_TOL and gaps_static.max() <= REL_TOL
                             and cells_covered == len(cell_audits)) else "FAIL",
        "cells_total": len(cell_audits),
        "cells_covered": cells_covered,
        "frames_seen": frames_seen,
        "instances": len(audits),
        "tolerance": REL_TOL,
        "max_chain_kinematic_objective_gap": float(gaps.max()),
        "max_chain_static_objective_gap": float(gaps_static.max()),
        "max_kinematic_equilibrium_residual": float(max(
            a["kinematic_equilibrium_residual"] for a in audits)),
        "max_static_equilibrium_residual": float(max(
            a["static_equilibrium_residual"] for a in audits)),
        "max_static_yield_residual": float(max(
            a["static_yield_residual"] for a in audits)),
        "max_static_duality_residual": float(max(
            a["static_duality_residual"] for a in audits)),
        "cell_audit_counts": cell_audits,
        "cell_frame_counts": cell_frames,
        "audits": audits,
        "seconds": round(time.perf_counter() - started, 2),
        "note": (
            "Every comparison is between objective values of three independently "
            "assembled programs: the exact story chain, the complete kinematic LP "
            "and the bounded static LP. The reported residuals are those of the two "
            "LP solutions, not an equilibrium residual of the chain solver."
        ),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "expanded_lp_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Expanded stratified LP audit",
        "",
        f"- Cells in the frozen grid: {payload['cells_total']}; "
        f"cells with at least one audit instance: {payload['cells_covered']}",
        f"- Independent LP comparisons: {payload['instances']}",
        f"- Maximum chain--kinematic objective gap: "
        f"{payload['max_chain_kinematic_objective_gap']:.3e}",
        f"- Maximum chain--static objective gap: "
        f"{payload['max_chain_static_objective_gap']:.3e}",
        f"- Maximum static equilibrium residual: "
        f"{payload['max_static_equilibrium_residual']:.3e}",
        f"- Maximum static yield residual: "
        f"{payload['max_static_yield_residual']:.3e}",
        f"- Maximum static duality residual: "
        f"{payload['max_static_duality_residual']:.3e}",
        f"- Comparison tolerance: {REL_TOL:.1e}",
        f"- Run time: {payload['seconds']} s",
        "",
        "## Audit count per stratification cell",
        "",
        "| Cell | Frames | Audit instances |",
        "|---|---:|---:|",
    ]
    for cell in sorted(cell_audits):
        lines.append(f"| `{cell}` | {cell_frames[cell]} | {cell_audits[cell]} |")
    lines += [
        "",
        "## Boundary",
        "",
        payload["note"],
        "The audit verifies the implementation on the frozen synthetic grid. It "
        "does not extend the exactness identity beyond the stated first-order "
        "moment model and it is not physical validation.",
        "",
    ]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "EXPANDED_LP_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"cells {payload['cells_covered']}/{payload['cells_total']} "
          f"instances {payload['instances']} "
          f"max_gap_kin {payload['max_chain_kinematic_objective_gap']:.3e} "
          f"max_gap_static {payload['max_chain_static_objective_gap']:.3e} "
          f"seconds {payload['seconds']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
