"""Real-section design map for the zero-sway attainability certificate.

Purpose
-------
The frozen ensemble in the manuscript is synthetic and dimensionless. This
study anchors the same certificate to *catalogue steel sections* so that the
capacity-pattern question is posed in design terms a structural engineer
recognises:

  kappa = M_p,beam / M_p,column        (SCWB-style strength ratio)
  rho   = D_g / (D_b + D_c)            (column-to-beam gravity demand share)

Both are computed from real AISC plastification values and a real gravity
demand model, not from a random multiplier.

Reuse discipline
----------------
The certificate itself is NOT reimplemented here. This script imports
``Scenario`` from the package engine and ``min_interval_margin`` from the
frozen gate, so every number produced is the audited certificate on a new
(real-section) capacity vector.
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

from engine import Frame, Scenario  # noqa: E402
from final_capacity_pattern_gate import min_interval_margin  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Catalogue sections.  Z values are plastic section moduli.
#    W (US) entries use the current AISC Shapes Database plastification values;
#    HN/HW (GB/T 11263) entries use the Chinese hot-rolled H-section tables.
#    Both sets are for ordinary structural steel Fy = 345 MPa (A992-like).
# ---------------------------------------------------------------------------
Fy = 345.0  # MPa

# name: (Z_x in 1e3 mm^3, depth mm, A in 100 mm^2)
SECTIONS: dict[str, tuple[float, float, float]] = {
    # --- W shapes (AISC) ---
    "W12X26": (179.0, 310.0, 49.0),
    "W14X30": (220.0, 348.0, 57.0),
    "W16X31": (258.0, 403.0, 59.0),
    "W18X35": (322.0, 450.0, 66.0),
    "W18X40": (366.0, 455.0, 76.0),
    "W21X44": (432.0, 525.0, 84.0),
    "W21X50": (512.0, 529.0, 95.0),
    "W24X55": (562.0, 599.0, 105.0),
    "W24X62": (644.0, 603.0, 118.0),
    "W27X84": (946.0, 678.0, 159.0),
    "W30X90": (1050.0, 752.0, 171.0),
    # --- W shapes as columns ---
    "W10X49": (272.0, 253.0, 93.0),
    "W12X65": (418.0, 308.0, 123.0),
    "W14X74": (561.0, 360.0, 141.0),
    "W14X90": (708.0, 356.0, 171.0),
    "W14X109": (883.0, 363.0, 207.0),
    "W14X132": (1100.0, 372.0, 251.0),
    "W14X176": (1530.0, 387.0, 334.0),
    "W14X211": (1890.0, 399.0, 401.0),
    # --- HN/HW shapes (GB/T 11263) ---
    "HN300X150": (490.0, 300.0, 47.5),
    "HN350X175": (703.0, 350.0, 63.7),
    "HN400X200": (971.0, 400.0, 84.1),
    "HN450X200": (1170.0, 450.0, 89.7),
    "HN500X200": (1330.0, 500.0, 93.9),
    "HN600X200": (1750.0, 600.0, 106.0),
    "HW300X300": (1330.0, 300.0, 119.0),
    "HW350X350": (1930.0, 350.0, 173.0),
    "HW400X400": (2670.0, 400.0, 219.0),
}

BEAMS = ["W16X31", "W18X35", "W18X40", "W21X44", "W21X50", "W24X55",
         "W24X62", "W27X84", "HN350X175", "HN400X200", "HN450X200",
         "HN500X200", "HN600X200"]
COLUMNS = ["W10X49", "W12X65", "W14X74", "W14X90", "W14X109", "W14X132",
           "W14X176", "W14X211", "HW300X300", "HW350X350", "HW400X400"]


def Mp(section: str) -> float:
    """Plastic moment capacity in kN m from Z and Fy."""
    Z_mm3 = SECTIONS[section][0] * 1.0e3
    return Fy * Z_mm3 / 1.0e6  # N mm -> kN m


def depth_m(section: str) -> float:
    return SECTIONS[section][1] / 1000.0


# ---------------------------------------------------------------------------
# 2. Gravity demand model, in consistent kN / m units.
# ---------------------------------------------------------------------------
GRAVITY = 6.0        # kN/m^2 factored floor load (dead + live, ULS-ish)
TRIB_WIDTH = 1.0     # m, one bay of tributary width for the 2-D model
BETA = 1.15          # beam fixed-end amplification of the simple span moment


def frame_from_sections(
    H: int,
    B: int,
    span_m: float,
    height_m: float,
    beam: str,
    column: str,
    base_scale: float = 1.0,
) -> tuple[Frame, dict]:
    """Build an engine Frame whose member capacities are real section M_p."""
    Mp_b = Mp(beam)
    Mp_c = Mp(column)
    spans = [span_m] * B
    heights = [height_m] * H

    # Demands per node: beam gravity shear demands and column axial demand.
    w = GRAVITY * TRIB_WIDTH                      # kN/m on a floor beam
    V_beam = w * span_m / 2.0                     # kN end shear of a floor beam
    D_b = BETA * w * span_m ** 2 / 8.0            # kN m beam moment demand
    n_floors_above = np.arange(H, 0, -1)
    P_col = 2.0 * V_beam * n_floors_above         # kN, interior column
    D_c = P_col * height_m / 8.0                  # kN m, P-Delta style column demand

    members = []
    for r in range(1, H + 1):
        mb = Mp_b
        mc = Mp_c * (base_scale if r == 1 else 1.0)
        for _g in range(B):
            members.append(["b", r, _g, mb, mb, 8.0 * mb, span_m])
        for _g in range(B + 1):
            members.append(["c", r, _g, mc, mc, 14.0 * mc, height_m])

    loads = [[D_b / span_m] * (B + 1) for _ in range(H)]
    f = Frame(
        f"sections_{beam}_{column}_H{H}_B{B}",
        spans,
        heights,
        members,
        loads,
        provenance=f"real AISC/GB sections; Fy={Fy} MPa; factored gravity {GRAVITY} kN/m2",
    )
    f.validate()

    kappa = Mp_b / Mp_c
    rho = float(np.mean(D_c)) / (D_b + float(np.mean(D_c)))
    meta = {
        "beam": beam,
        "column": column,
        "Mp_beam_kNm": Mp_b,
        "Mp_column_kNm": Mp_c,
        "kappa": kappa,
        "kappa_Mp_beam_over_column": kappa,
        "rho": rho,
        "rho_column_demand_share": rho,
        "beam_demand_kNm": D_b,
        "column_demand_kNm_mean": float(np.mean(D_c)),
        "beam_DCR": D_b / Mp_b,
        "column_DCR_mean": float(np.mean(D_c)) / Mp_c,
        "span_m": span_m,
        "height_m": height_m,
    }
    return f, meta


def analyse(f: Frame, removal: tuple[int, int]) -> dict:
    """Run the audited certificate on a real-section frame."""
    sc = Scenario(f, removal)
    x = np.ones(f.E)
    full = float(sc.chain(x)["capacity"])
    loc = float(sc.chain(x, local=True)["capacity"])
    margin = float(min_interval_margin(sc, x))
    return {"local": loc, "complete": full, "margin": margin,
            "exact": bool(margin >= -1.0e-8),
            "gap": loc - full}


def main() -> None:
    out: dict = {"Fy_MPa": Fy, "gravity_kNm2": GRAVITY,
                 "beta_beam": BETA, "sections": SECTIONS,
                 "cases": [], "envelope": []}

    # ---- A. single-bay two-story analytic-style anchor, real sections ----
    for beam in ["W16X31", "W18X40", "W21X50", "W24X62"]:
        for column in ["W12X65", "W14X109", "W14X211"]:
            f, meta = frame_from_sections(2, 1, 6.0, 4.0, beam, column)
            row = dict(meta)
            row["H"] = 2
            row["B"] = 1
            row.update(analyse(f, (1, 0)))
            out["cases"].append(row)

    # ---- B. design envelope over H and B for a fixed section pair ----
    for H in (2, 4, 6, 8, 12):
        for B in (1, 2, 3, 4):
            for beam, column in [("W18X40", "W14X109"), ("W21X50", "W14X132"),
                                 ("HN400X200", "HW350X350")]:
                f, meta = frame_from_sections(H, B, 6.0, 4.0, beam, column)
                worst = None
                exact_all = True
                for removal in f.scenarios():
                    res = analyse(f, removal)
                    if not res["exact"]:
                        exact_all = False
                    if worst is None or res["margin"] < worst["margin"]:
                        worst = dict(res)
                        worst["removal"] = list(removal)
                row = dict(meta)
                row.update({"H": H, "B": B, "exact_all_removals": exact_all,
                            "worst_margin": worst["margin"],
                            "worst_removal": worst["removal"],
                            "worst_gap": worst["gap"],
                            "worst_local": worst["local"],
                            "worst_complete": worst["complete"]})
                out["envelope"].append(row)

    # ---- C. kappa sweep at fixed frame ----
    sweep = []
    for column in COLUMNS:
        for beam in BEAMS:
            f, meta = frame_from_sections(4, 2, 6.0, 4.0, beam, column)
            worst = None
            n_inexact = 0
            for removal in f.scenarios():
                res = analyse(f, removal)
                if not res["exact"]:
                    n_inexact += 1
                if worst is None or res["margin"] < worst["margin"]:
                    worst = dict(res)
            sweep.append({
                "beam": beam, "column": column,
                "kappa": meta["kappa_Mp_beam_over_column"],
                "rho": meta["rho_column_demand_share"],
                "beam_DCR": meta["beam_DCR"],
                "column_DCR": meta["column_DCR_mean"],
                "worst_margin": worst["margin"],
                "exact_all_removals": n_inexact == 0,
                "n_inexact_removals": n_inexact,
                "n_removals": 4 * 3,
                "max_relative_gap": max(0.0, worst["gap"] / max(worst["complete"], 1e-12)),
            })
    out["kappa_sweep"] = sweep

    dest = PKG / "data" / "real_section_design_map.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)

    # ---- console summary ----
    print("\n-- A. two-story real-section anchors --")
    for r in out["cases"]:
        print(f"  {r['beam']:>11s}/{r['column']:<9s} kappa={r['kappa']:5.3f} "
              f"rho={r['rho']:.3f} margin={r['margin']:9.4f} "
              f"loc={r['local']:8.2f} full={r['complete']:8.2f} exact={r['exact']}")

    print("\n-- B. envelope (exact at all removals?) --")
    n_ok = sum(1 for r in out["envelope"] if r["exact_all_removals"])
    print(f"  {n_ok}/{len(out['envelope'])} frame configurations exact at every removal")
    for r in out["envelope"][:12]:
        print(f"  H={r['H']:2d} B={r['B']} {r['beam']:>10s}/{r['column']:<9s} "
              f"exact_all={str(r['exact_all_removals']):5s} worst_margin={r['worst_margin']:10.3f} "
              f"at {r['worst_removal']}")

    print("\n-- C. kappa sweep summary (H=4,B=2) --")
    ks = np.array([r["kappa"] for r in sweep])
    ok = np.array([r["exact_all_removals"] for r in sweep])
    print(f"  kappa range {ks.min():.3f}..{ks.max():.3f}; "
          f"{int(ok.sum())}/{len(sweep)} section pairs exact at all removals")
    if ok.any() and (~ok).any():
        print(f"  max kappa with exact_all = {ks[ok].max():.3f}")
        print(f"  min kappa with inexact  = {ks[~ok].min():.3f}")


if __name__ == "__main__":
    main()
