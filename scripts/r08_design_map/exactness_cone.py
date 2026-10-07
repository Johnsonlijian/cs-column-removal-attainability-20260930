"""Is the exactness set a cone in capacity space?

If the margin is positively homogeneous of degree one in the capacities, then
Delta_min(alpha * M) = alpha * Delta_min(M) for every alpha > 0, so the
exactness set is a cone with apex at the origin and the boundary can be stated
in ratios alone. That would be the cleanest possible statement of the design
map, and it is testable directly and exactly.

The engine works with absolute values, so this is a property of the certificate
and not an assumption. Homogeneity of the per-joint min-cut weight is obvious
(the weight is a sum of capacities), but the minimum over bands and intervals
could in principle break it, and the demands D enter through lambda rather than
through the margin, so the margin formula should not see D at all. This script
checks both claims numerically over a wide sweep.
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


def build(H, B, spans, heights, beam_factors, col_factor, load_scale=1.0):
    beam = Mp("W18X40")
    column = Mp("W14X109") * col_factor
    edges = np.r_[0.0, np.cumsum(spans)]
    loads = np.zeros((H, B + 1))
    for j in range(B + 1):
        left = spans[j - 1] / 2 if j > 0 else 0.0
        right = spans[j] / 2 if j < B else 0.0
        loads[:, j] = (left + right) * GRAVITY * TRIB * load_scale
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mbf = beam_factors[g] if g < len(beam_factors) else beam_factors[-1]
            mb = beam * mbf
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, column, column, 14.0 * column,
                            heights[r - 1]])
    f = Frame(f"cone_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="homogeneity check")
    f.validate()
    return f


def margins_for(H, B, spans, heights, beam_factors, col_factor, removal,
                load_scale=1.0):
    f = build(H, B, spans, heights, beam_factors, col_factor, load_scale)
    sc = Scenario(f, removal)
    return float(min_interval_margin(sc, np.ones(f.E))), \
        float(sc.chain(np.ones(f.E))["capacity"]), \
        float(sc.chain(np.ones(f.E), local=True)["capacity"])


def main() -> None:
    rows = []
    hom_ok = hom_tot = 0
    worst_hom = 0.0
    lam_ok = lam_tot = 0
    worst_lam = 0.0

    configs = [
        (2, 2, [6.0, 6.0], [4.0, 4.0], [1.0, 1.0]),
        (2, 3, [6.0, 6.0, 6.0], [4.0, 4.0], [1.0, 1.0, 1.0]),
        (4, 2, [6.0, 6.0], [4.0] * 4, [2.0, 1.0]),
        (4, 3, [4.0, 6.0, 8.0], [5.0, 4.0, 4.0, 3.0], [1.0, 2.0, 0.5]),
        (6, 2, [6.0, 6.0], [3.0, 4.0, 4.0, 4.0, 4.0, 5.0], [1.5, 1.0]),
        (3, 4, [5.0, 6.0, 6.0, 7.0], [4.0] * 3, [0.5, 1.0, 2.0, 1.0]),
    ]
    scales = (0.25, 0.5, 1.0, 2.0, 5.0, 8.0)

    print("=== homogeneity of the margin in the capacities ===")
    print(f"  {'config':<28s} {'removal':>9s} {'alpha':>7s} "
          f"{'margin':>12s} {'margin/alpha':>13s}")
    for ci, (H, B, spans, heights, bf) in enumerate(configs):
        for removal in ((1, 0), (1, B // 2), (H, 0)):
            seq = []
            for a in scales:
                m, _l, _c = margins_for(H, B, spans, heights,
                                        [v * a for v in bf], a, removal)
                seq.append((a, m))
            # the reference must be the alpha = 1.0 run, not the first one in
            # the scale ladder (which is 0.25)
            base = next(m for a, m in seq if abs(a - 1.0) < 1e-12)
            for a, m in seq:
                pred = a * base
                err = abs(m - pred)
                worst_hom = max(worst_hom, err)
                hom_tot += 1
                hom_ok += int(err <= 1e-6 * max(1.0, abs(pred)))
            rows.append({"config": ci, "H": H, "B": B, "removal": list(removal),
                         "margin_sequence": [{"alpha": a, "margin": m}
                                             for a, m in seq]})
            if ci < 2:
                for a, m in seq:
                    print(f"  {f'H={H} B={B}':<28s} {str(removal):>9s} "
                          f"{a:>7.2f} {m:>12.3f} {m/a:>13.3f}")

    print(f"\n  margin homogeneity: {hom_ok}/{hom_tot} checks pass "
          f"(worst absolute error {worst_hom:.3e})")

    print("\n=== the certificate values scale too, so the exactness cone is exact ===")
    for ci, (H, B, spans, heights, bf) in enumerate(configs):
        for removal in ((1, 0),):
            v = []
            for a in (1.0, 4.0):
                m, lf, ll = margins_for(H, B, spans, heights,
                                        [x * a for x in bf], a, removal)
                v.append((m, lf, ll))
            m1, lf1, ll1 = v[0]
            m4, lf4, ll4 = v[1]
            for got, want in ((m4, 4 * m1), (lf4, 4 * lf1), (ll4, 4 * ll1)):
                lam_tot += 1
                err = abs(got - want)
                worst_lam = max(worst_lam, err)
                lam_ok += int(err <= 1e-6 * max(1.0, abs(want)))
    print(f"  margin, local and complete all scale linearly: {lam_ok}/{lam_tot} "
          f"(worst error {worst_lam:.3e})")

    print("\n=== the margin does not see the demand magnitude ===")
    print(f"  {'load scale':>11s} {'margin':>12s} {'lambda_full':>13s}")
    for ls in (0.5, 1.0, 2.0, 4.0):
        m, lf, ll = margins_for(4, 2, [6.0, 6.0], [4.0] * 4, [2.0, 1.0],
                                1.0, (1, 0), load_scale=ls)
        print(f"  {ls:>11.2f} {m:>12.3f} {lf:>13.4f}")

    dest = PKG / "data" / "exactness_cone.json"
    dest.write_text(json.dumps({
        "homogeneity_checks": hom_tot, "homogeneity_pass": hom_ok,
        "worst_error": worst_hom,
        "linearity_checks": lam_tot, "linearity_pass": lam_ok,
        "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
