"""Why is a single-bay frame never exact? Inspect the actual mechanisms.

The 0/96 split is a measured result. Before it can be stated as a theorem, the
mechanism has to be understood, because the obvious argument is wrong.

Obvious (incorrect) argument: "with B=1 the removed column is always exterior, so
the surviving vertical chain is one line; setting every story drift to zero makes
every column chord rate zero, so the local screen already contains the global
mechanism."

Check: with all u_r = 0, and only one vertical line surviving, the vertical
amplitudes are uniform, so the surviving column has zero chord rate too. The
local screen should then be minimal. If the certificate reports a gap, either
this reasoning is wrong or the removal is not what I think it is.

This script prints, for one single-bay frame and each removal, the exact story
drift vector w, the local and complete values, and which member ends hinge, so
the mechanism can be read off rather than assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "scripts" / "code"))
sys.path.insert(0, str(HERE))

from engine import Scenario, two_story  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402
from real_section_design_map import frame_from_sections  # noqa: E402


def report(frame, label: str) -> None:
    print(f"\n=== {label}: H={frame.H}, B={frame.B} ===")
    print(f"  members: {[(m[0], int(m[1]), int(m[2])) for m in frame.members]}")
    for removal in frame.scenarios():
        sc = Scenario(frame, removal)
        x = np.ones(frame.E)
        full = sc.chain(x)
        loc = sc.chain(x, local=True)
        margin = float(min_interval_margin(sc, x))
        gap = loc["capacity"] - full["capacity"]
        print(f"\n  removal (s,g)={removal}")
        print(f"    local={loc['capacity']:.6f}  complete={full['capacity']:.6f}"
              f"  gap={gap:+.6f}  margin={margin:+.6f}")
        print(f"    local   story drifts w = "
              f"{np.round(loc['w'] * sc.D, 6).tolist()}")
        print(f"    complete story drifts w = "
              f"{np.round(full['w'] * sc.D, 6).tolist()}")
        print(f"    surviving members: "
              f"{[(m[0], int(m[1]), int(m[2])) for m in sc.members]}")
        # which member ends carry hinge work in each mechanism
        print(f"    local   coeff>0 on members "
              f"{[i for i, c in enumerate(loc['coeff']) if c > 1e-9]}")
        print(f"    complete coeff>0 on members "
              f"{[i for i, c in enumerate(full['coeff']) if c > 1e-9]}")


def main() -> None:
    f1, _ = frame_from_sections(2, 1, 6.0, 4.0, "W18X40", "W14X109")
    report(f1, "catalogue single-bay (B=1)")

    f2 = two_story()
    report(f2, "analytical two-story one-bay")

    f3, _ = frame_from_sections(2, 2, 6.0, 4.0, "W18X40", "W14X109")
    print(f"\n=== comparison: catalogue two-bay (B=2), H=2 ===")
    for removal in f3.scenarios():
        sc = Scenario(f3, removal)
        x = np.ones(f3.E)
        margin = float(min_interval_margin(sc, x))
        full = sc.chain(x)
        loc = sc.chain(x, local=True)
        print(f"  removal {removal}: margin={margin:+.3f} "
              f"local={loc['capacity']:.4f} complete={full['capacity']:.4f} "
              f"same={abs(margin) < 1e-8 or margin > 0}")


if __name__ == "__main__":
    main()
