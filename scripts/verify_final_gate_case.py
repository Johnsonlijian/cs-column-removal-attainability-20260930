from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
sys.path.insert(0, str(PACKAGE / "scripts" / "code"))

from engine import Scenario  # noqa: E402
from final_capacity_pattern_gate import controlled_frame, min_interval_margin  # noqa: E402


def main() -> None:
    data_dir = ROOT.parent / "data"
    p = json.loads((data_dir / "final_capacity_pattern_gate.json").read_text(encoding="utf-8"))
    best = None
    for key, group in p["groups"].items():
        case = group["treatments"]["componentwise_strengthening"]["best"]
        if case is not None and (best is None or case["relative_gap"] > best["relative_gap"]):
            best = {**case, "group": key}
    if best is None:
        raise SystemExit("no componentwise-strengthening gap case found")
    match = re.fullmatch(r"final_(?P<regime>[^_]+)_(?P<geometry>.+)_H(?P<H>\d+)_B(?P<B>\d+)_seed(?P<seed>\d+)", best["frame"])
    if match is None:
        raise SystemExit(f"cannot parse frame name: {best['frame']}")
    regime = match.group("regime")
    geometry = match.group("geometry")
    H = int(match.group("H"))
    B = int(match.group("B"))
    seed = int(match.group("seed"))
    f = controlled_frame(H, B, seed, regime, geometry)
    x = np.asarray(best["multipliers"], dtype=float)
    removal = tuple(best["removal"])
    sc = Scenario(f, removal)
    local = sc.chain(x, local=True)
    full = sc.chain(x, local=False)
    kin = sc.kinematic(x, local=False)
    stat = sc.static(x, domain="strip")
    base = Scenario(f, removal)
    base_local = base.chain(np.ones(f.E), local=True)["capacity"]
    base_full = base.chain(np.ones(f.E), local=False)["capacity"]
    result = {
        "frame": best["frame"],
        "group": best["group"],
        "removal": list(removal),
        "multipliers_min": float(np.min(x)),
        "multipliers_max": float(np.max(x)),
        "baseline_local": float(base_local),
        "baseline_full": float(base_full),
        "local_capacity": float(local["capacity"]),
        "full_chain_capacity": float(full["capacity"]),
        "kinematic_capacity": float(kin["capacity"]),
        "static_strip_capacity": float(stat["capacity"]),
        "chain_kinematic_residual": float(abs(full["capacity"] - kin["capacity"])),
        "chain_static_residual": float(abs(full["capacity"] - stat["capacity"])),
        "static_equilibrium_residual": float(stat["eq_res"]),
        "static_yield_residual": float(stat["yield_res"]),
        "static_duality_residual": float(stat["duality_res"]),
        "minimum_interval_margin": float(min_interval_margin(sc, x)),
        "multipliers": best["multipliers"],
        "interpretation": "synthetic frozen-gate witness; all component factors >=1; not a universal retrofit claim",
    }
    out = data_dir / "final_gate_case_verification.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
