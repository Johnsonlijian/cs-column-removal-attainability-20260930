"""Moment-strip versus declared diamond yield-domain probe.

The diamond domain ``|N|/Np + |M|/Mp <= x`` already exists in the bundled
static program. This script asks only whether the moment-domain exactness
verdict still points the same way once that declared interaction is active.
It is not an AISC curve and not an extension of the theorem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from engine import Scenario  # noqa: E402
from canonical_case import frame as canonical_frame  # noqa: E402
from final_capacity_pattern_gate import (  # noqa: E402
    B_VALUES,
    GEOMETRIES,
    H_VALUES,
    REGIMES,
    controlled_frame,
)


def compare(sc: Scenario, x: np.ndarray) -> dict:
    local = float(sc.chain(x, local=True)["capacity"])
    full = float(sc.chain(x, local=False)["capacity"])
    strip = float(sc.static(x, domain="strip")["capacity"])
    diamond = float(sc.static(x, domain="diamond")["capacity"])
    tol = 1.0e-8 * max(1.0, abs(local))
    return {
        "local_moment": local,
        "complete_moment": full,
        "static_strip": strip,
        "static_diamond": diamond,
        "moment_gap": local - full,
        "diamond_below_local": diamond < local - tol,
        "moment_inexact": local - full > tol,
        "relative_diamond_drop": (strip - diamond) / max(abs(strip), 1e-12),
    }


def canonical_rows() -> list[dict]:
    f = canonical_frame()
    rows = []
    for k in (1.0, 1.25, 2.0, 5.0):
        x = np.ones(f.E)
        for i, member in enumerate(f.members):
            if not (member[0] == "c" and member[1] == 1 and member[2] == 0):
                x[i] = k
        sc = Scenario(f, (1, 1))
        row = compare(sc, x)
        row["k"] = k
        row["removal"] = [1, 1]
        rows.append(row)
    return rows


def ensemble_rows() -> list[dict]:
    rows = []
    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    f = controlled_frame(H, B, 20260925, regime, mode)
                    s, g = 1, min(1, B)
                    sc = Scenario(f, (s, g))
                    row = compare(sc, np.ones(f.E))
                    row.update({"H": H, "B": B, "geometry": mode, "regime": regime, "removal": [s, g]})
                    rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    inexact = [r for r in rows if r["moment_inexact"]]
    exact = [r for r in rows if not r["moment_inexact"]]
    verdict_hold = sum(
        1 for r in rows if r["moment_inexact"] == r["diamond_below_local"] or (
            r["moment_inexact"] and r["diamond_below_local"]
        )
    )
    # Verdict of interest: if the moment screen is inexact, diamond is also
    # below the local moment value. If the moment screen is exact, diamond may
    # still drop below the moment value; that is a domain change, not a
    # certificate failure.
    inexact_still_low = sum(1 for r in inexact if r["diamond_below_local"])
    drops = [r["relative_diamond_drop"] for r in rows]
    return {
        "n": n,
        "moment_inexact": len(inexact),
        "moment_exact": len(exact),
        "inexact_and_diamond_below_local": inexact_still_low,
        "max_relative_diamond_drop": max(drops) if drops else 0.0,
        "median_relative_diamond_drop": float(np.median(drops)) if drops else 0.0,
    }


def main() -> None:
    canonical = canonical_rows()
    ensemble = ensemble_rows()
    payload = {
        "domain_statement": (
            "diamond means |N|/Np + |M|/Mp <= x as declared in engine.Scenario.yield_matrix; "
            "not an AISC interaction formula"
        ),
        "canonical": canonical,
        "canonical_summary": summarize(canonical),
        "ensemble_one_removal_per_cell": ensemble,
        "ensemble_summary": summarize(ensemble),
    }
    out = ROOT / "data" / "diamond_domain_probe.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"canonical": payload["canonical_summary"], "ensemble": payload["ensemble_summary"]}, indent=2))


if __name__ == "__main__":
    main()
