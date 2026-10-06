"""Write capacity-path statistics from the frozen gate pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))

from final_capacity_pattern_gate import main  # noqa: E402


def verify() -> bool:
    frozen = json.loads((DATA / "final_capacity_pattern_gate.json").read_text(encoding="utf-8"))
    path = json.loads((DATA / "capacity_path_analysis.json").read_text(encoding="utf-8"))
    ok = True
    for treatment in ("componentwise_strengthening", "geometric_mean_one_redistribution"):
        agg = frozen["aggregate"][treatment]
        got = path["treatments"][treatment]
        ok &= got["frames_with_new_gap"] == agg["frames_with_new_gap"]
        ok &= got["gap_instances"] == agg["gap_instances"]
        print(f"{treatment}: frames {got['frames_with_new_gap']} gap_instances {got['gap_instances']}")
    return ok


if __name__ == "__main__":
    main(write_path_analysis=True)
    print("alignment", "PASS" if verify() else "FAIL")
