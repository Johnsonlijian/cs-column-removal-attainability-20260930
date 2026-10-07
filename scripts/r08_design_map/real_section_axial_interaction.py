"""Stage 3: does the design-map conclusion survive axial-moment interaction?

The external review panel's central objection was that the catalogue-section
design map uses the full plastic moment of every column. Real columns carry
gravity axial force, and the manuscript's own diamond-domain probe shows that
activating |N|/Np + |M|/Mp <= x can cut the usable moment capacity by tens of
percent. If the 0/96 versus 192/192 split is an artefact of unchallenged M_p,
it must not be reported.

This script therefore recomputes the same catalogue-section design points with
an axial-force-reduced effective column plastic moment, and reports:

  * the interaction DCR  P/(A Fy)  and the reduction factor r = 1 - P/(A Fy);
  * the effective moment ratio  kappa_eff = Mp_beam / (r * Mp_col);
  * whether each design is exact at every removal, before and after reduction;
  * a strip-versus-interaction confusion matrix.

The certificate itself is unchanged and imported, so the comparison isolates
the effect of the capacity reduction.
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
from real_section_design_map import SECTIONS, Mp, depth_m  # noqa: E402

FY = 345.0          # MPa
GRAVITY = 6.0       # kN/m2 factored floor load
TRIB = 1.0          # m tributary width for the 2-D model
BETA = 1.15
PHI = 0.9           # resistance factor applied to the plastic moment


def area_mm2(section: str) -> float:
    """Section area in mm^2 (stored in units of 100 mm^2)."""
    return SECTIONS[section][2] * 100.0


def axial_dcr(section: str, P_kN: float) -> float:
    """P / (A Fy) with A in mm^2, Fy in MPa, P in kN."""
    A = area_mm2(section)
    return P_kN * 1.0e3 / (A * FY)


def build(H, B, span, height, beam, column, reduce_axial: bool,
          load_shape="ribbon"):
    """Frame whose column capacity is optionally reduced for axial force."""
    Mp_b = Mp(beam)
    w = GRAVITY * TRIB
    D_b = BETA * w * span ** 2 / 8.0
    V_beam = w * span / 2.0

    spans = [span] * B
    heights = [height] * H
    lines_share = np.r_[span / 2, np.full(max(B - 1, 0), span), span / 2] / span
    loads = np.outer(np.ones(H), lines_share) * (w * span)

    rows = []
    max_dcr = 0.0
    for r in range(1, H + 1):
        n_above = H - r + 1
        # Interior column on the removed line carries two beam reactions per
        # floor above; exterior columns carry one. Use the tributary-area
        # average as the design axial force for the section-level check.
        P = 2.0 * V_beam * n_above * np.mean(lines_share) * (B + 1) / 2.0
        dcr = axial_dcr(column, P)
        max_dcr = max(max_dcr, dcr)
        # Effective plastic moment from the linear (diamond) interaction:
        #   |N|/Np + |M|/Mp <= 1  with Np = A Fy  =>
        #   M_eff = Mp * (1 - P/(A Fy)).
        r_eff = max(1.0 - dcr, 0.05)
        Mp_c = Mp(column) * (r_eff if reduce_axial else 1.0)
        for g in range(B):
            rows.append(["b", r, g, Mp_b, Mp_b, 8.0 * Mp_b, spans[g]])
        for g in range(B + 1):
            rows.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])

    f = Frame(f"ax_{beam}_{column}_H{H}_B{B}", spans, heights, rows,
              loads.tolist(),
              provenance=f"axial-reduced={reduce_axial}")
    f.validate()
    return f, {"kappa_eff": Mp_b / (Mp(column) * max(1.0 - max_dcr, 0.05)),
               "max_axial_dcr": float(max_dcr),
               "reduction": float(max(1.0 - max_dcr, 0.05))}


def scan(f):
    worst = np.inf
    n_bad = 0
    for removal in f.scenarios():
        sc = Scenario(f, removal)
        x = np.ones(f.E)
        m = float(min_interval_margin(sc, x))
        worst = min(worst, m)
        if m < -1.0e-8:
            n_bad += 1
    return {"worst_margin": float(worst), "n_inexact": n_bad,
            "n_removals": len(f.scenarios()), "exact_all": n_bad == 0}


def main() -> None:
    pairs = [("W18X40", "W14X109"), ("W21X50", "W14X132"),
             ("W24X62", "W14X176"), ("HN400X200", "HW350X350"),
             ("HN500X200", "HW350X350")]
    rows = []
    for H in (2, 4, 8):
        for B in (1, 2, 3):
            for beam, column in pairs:
                base, meta_b = build(H, B, 6.0, 4.0, beam, column, False)
                red, meta_r = build(H, B, 6.0, 4.0, beam, column, True)
                sb, sr = scan(base), scan(red)
                rows.append({
                    "H": H, "B": B, "beam": beam, "column": column,
                    "kappa": Mp(beam) / Mp(column),
                    "kappa_eff": meta_r["kappa_eff"],
                    "max_axial_dcr": meta_r["max_axial_dcr"],
                    "reduction": meta_r["reduction"],
                    "strip_exact": sb["exact_all"],
                    "strip_inexact": sb["n_inexact"],
                    "inter_exact": sr["exact_all"],
                    "inter_inexact": sr["n_inexact"],
                    "n_removals": sb["n_removals"],
                    "strip_worst_margin": sb["worst_margin"],
                    "inter_worst_margin": sr["worst_margin"],
                })

    out = {"note": "strip vs axial-moment interaction on catalogue sections",
           "FY_MPa": FY, "PHI": PHI, "rows": rows}
    dest = PKG / "data" / "real_section_axial_interaction.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)

    print("\n=== axial demand range on these designs ===")
    d = np.array([r["max_axial_dcr"] for r in rows])
    print(f"  P/(A Fy) max per design: min {d.min():.3f}  median "
          f"{np.median(d):.3f}  max {d.max():.3f}")
    print(f"  implied column moment reduction r = 1 - P/(A Fy): "
          f"{1 - d.max():.3f} .. {1 - d.min():.3f}")

    print("\n=== confusion matrix: strip exactness vs interaction exactness ===")
    tb = sum(1 for r in rows if r["strip_exact"])
    ti = sum(1 for r in rows if r["inter_exact"])
    both = sum(1 for r in rows if r["strip_exact"] and r["inter_exact"])
    only_strip = sum(1 for r in rows if r["strip_exact"] and not r["inter_exact"])
    only_inter = sum(1 for r in rows if not r["strip_exact"] and r["inter_exact"])
    neither = sum(1 for r in rows if not r["strip_exact"] and not r["inter_exact"])
    print(f"  n = {len(rows)}")
    print(f"  exact under BOTH                          : {both}")
    print(f"  exact under strip only (lost by interaction): {only_strip}")
    print(f"  exact under interaction only              : {only_inter}")
    print(f"  inexact under both                        : {neither}")
    print(f"  strip exact total {tb}, interaction exact total {ti}")

    print("\n=== exactness by bay count ===")
    for label, key in (("strip (full Mp)", "strip_exact"),
                       ("interaction-reduced", "inter_exact")):
        for B in (1, 2, 3):
            sub = [r for r in rows if r["B"] == B]
            ok = sum(1 for r in sub if r[key])
            print(f"  {label:<22s} B={B}: {ok:2d}/{len(sub):2d} exact at every removal")

    print("\n=== designs that LOSE exactness under interaction ===")
    lost = [r for r in rows if r["strip_exact"] and not r["inter_exact"]]
    if not lost:
        print("  none")
    for r in lost:
        print(f"  H={r['H']} B={r['B']} {r['beam']}/{r['column']} "
              f"dcr={r['max_axial_dcr']:.3f} r={r['reduction']:.3f} "
              f"kappa {r['kappa']:.3f} -> {r['kappa_eff']:.3f} "
              f"inexact {r['inter_inexact']}/{r['n_removals']}")


if __name__ == "__main__":
    main()
