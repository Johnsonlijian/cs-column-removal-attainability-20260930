"""The design rule: a column-to-beam strength threshold for screen validity.

Established:
  * the margin is homogeneous of degree one in the capacities, so the exactness
    set is a cone and the boundary depends on capacity RATIOS only;
  * the margin decreases monotonically in the detached-side beam capacity and
    increases monotonically in the intact-side beam capacity;
  * therefore, for fixed everything else, the boundary is a threshold on the
    ratio of the column capacity to the detached-side beam capacity.

Define

    theta(geometry) = critical C / M_det

so that the screen is exact when C / M_det >= theta and inexact below it.
This script measures theta by bisection for several geometries and removal
types, and then tests the rule as a predictor rather than reporting the root.

The bisection is written with explicit keyword arguments: an earlier version
passed positional arguments in the wrong order and silently probed a different
quantity, so the signatures below are keyword-only where it matters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "scripts"))
sys.path.insert(0, str(PKG / "scripts" / "code"))
sys.path.insert(0, str(HERE))

from engine import Frame, Scenario  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402
from real_section_design_map import Mp  # noqa: E402

GRAVITY, TRIB = 6.0, 1.0


def build(H, B, spans, heights, M_det, M_int, C):
    loads = np.zeros((H, B + 1))
    for j in range(B + 1):
        left = spans[j - 1] / 2 if j > 0 else 0.0
        right = spans[j] / 2 if j < B else 0.0
        loads[:, j] = (left + right) * GRAVITY * TRIB
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = M_det if g == 0 else M_int
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, C, C, 14.0 * C, heights[r - 1]])
    f = Frame(f"thr_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="column threshold")
    f.validate()
    return f


def margin(H, B, spans, heights, M_det, M_int, C, removal):
    """Signed minimum interval margin. All capacity arguments by keyword use."""
    f = build(H, B, spans, heights, M_det, M_int, C)
    sc = Scenario(f, removal)
    return float(min_interval_margin(sc, np.ones(f.E)))


def theta(H, B, spans, heights, removal, *, M_int=1.0, M_det=1.0,
          lo=1e-6, hi=10.0, tol=1e-12):
    """Critical C / M_det. Exact for C/M_det >= theta.

    The margin increases with C, so the screen fails at small C and holds at
    large C; the brackets are set accordingly.
    """
    m_lo = margin(H, B, spans, heights, M_det=M_det, M_int=M_int,
                  C=lo * M_det, removal=removal)
    m_hi = margin(H, B, spans, heights, M_det=M_det, M_int=M_int,
                  C=hi * M_det, removal=removal)
    if m_lo >= 0:
        return 0.0, "exact even at minimal column"
    if m_hi < 0:
        return None, f"still inexact at C/M_det={hi}"
    a, b = lo, hi
    for _ in range(400):
        mid = 0.5 * (a + b)
        m = margin(H, B, spans, heights, M_det=M_det, M_int=M_int,
                   C=mid * M_det, removal=removal)
        if m >= 0:
            b = mid
        else:
            a = mid
        if b - a <= tol * max(1.0, b):
            break
    return 0.5 * (a + b), "ok"


def main() -> None:
    rows = []
    geoms = [
        ("H4 B2 span6 h4", 4, 2, [6.0, 6.0], [4.0] * 4),
        ("H4 B3 span5-7 h4", 4, 3, [5.0, 6.0, 7.0], [4.0] * 4),
        ("H4 B4 span6 h4", 4, 4, [6.0] * 4, [4.0] * 4),
        ("H4 B2 span6 h3", 4, 2, [6.0, 6.0], [3.0] * 4),
        ("H4 B2 span6 h5", 4, 2, [6.0, 6.0], [5.0] * 4),
        ("H6 B2 span6 h4", 6, 2, [6.0, 6.0], [4.0] * 6),
        ("H2 B2 span6 h4", 2, 2, [6.0, 6.0], [4.0] * 2),
        ("H4 B2 span9 h4", 4, 2, [9.0, 9.0], [4.0] * 4),
    ]

    print("=== critical column-to-detached-beam ratio theta = C / M_det ===")
    print("  screen exact for C/M_det >= theta")
    print(f"  {'geometry':<20s} {'removal':>10s} {'theta':>12s} {'note':>26s}")
    for name, H, B, spans, heights in geoms:
        for s, g, tag in ((1, 0, "ground ext"), (1, B // 2, "ground int"),
                          (H, 0, "roof ext")):
            t, note = theta(H, B, spans, heights, (s, g))
            rows.append({"geometry": name, "H": H, "B": B, "spans": spans,
                         "heights": heights, "removal": [s, g], "tag": tag,
                         "theta": t, "note": note})
            ts = f"{t:.6f}" if t is not None else "none"
            print(f"  {name:<20s} {tag:>10s} {ts:>12s} {note:>26s}")

    print("\n=== theta is scale-free: scaling every capacity must not move it ===")
    base = next(r for r in rows
                if r["geometry"] == "H4 B2 span6 h4" and r["tag"] == "ground int")
    th = base["theta"]
    print(f"  interior ground removal: theta = {th:.6f}")
    print("  theta is a ratio of two capacities, so scaling every capacity by")
    print("  alpha must leave the margin at alpha * theta equal to zero")
    print(f"  {'alpha':>7s} {'margin at C/M_det = theta':>28s}")
    for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
        m = margin(4, 2, [6.0, 6.0], [4.0] * 4,
                   M_det=alpha, M_int=alpha, C=th * alpha,
                   removal=(1, 1))
        print(f"  {alpha:>7.2f} {m:>28.3e}")

    print("\n=== the rule as a predictor, interior ground removal ===")
    print(f"  {'C/M_det':>9s} {'margin':>13s} {'verdict':>9s} {'predicted':>10s}")
    for ratio in (0.3, 0.6, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5, 3.0):
        m = margin(4, 2, [6.0, 6.0], [4.0] * 4,
                   M_det=1.0, M_int=1.0, C=ratio * th, removal=(1, 1))
        verdict = "exact" if m >= -1e-9 else "inexact"
        pred = "exact" if ratio >= 1.0 else "inexact"
        flag = "ok" if verdict == pred else "MISMATCH"
        print(f"  {ratio:>9.2f} {m:>13.4f} {verdict:>9s} {pred:>10s}  {flag}")

    print("\n=== does theta depend on the span-to-height ratio? ===")
    print(f"  {'span':>6s} {'height':>7s} {'h/L':>6s} {'theta':>10s} "
          f"{'theta/(h/L)':>13s}")
    for span, height in ((6.0, 4.0), (9.0, 4.0), (6.0, 3.0), (6.0, 5.0),
                         (12.0, 4.0)):
        t, note = theta(4, 2, [span, span], [height] * 4, (1, 1))
        t = t if t is not None else float("nan")
        print(f"  {span:>6.1f} {height:>7.1f} {height/span:>6.3f} {t:>10.6f} "
              f"{t/(height/span):>13.6f}")

    dest = PKG / "data" / "column_threshold.json"
    dest.write_text(json.dumps({
        "rule": "screen exact iff C / M_det >= theta(geometry, removal)",
        "theta_base": th, "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
