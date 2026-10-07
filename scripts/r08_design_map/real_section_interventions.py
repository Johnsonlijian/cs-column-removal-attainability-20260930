"""Real-section design map, stage 2: strengthening interventions on catalogue shapes.

Stage 1 (real_section_design_map.py) showed that uniform-capacity multi-bay
frames built from catalogue sections are exact at every removal, while
single-bay frames are not. This stage asks the design question directly:

    which realistic member-upgrade decisions break the zero-sway screen?

Interventions, all applied to a real catalogue baseline and all keeping the
gravity demand fixed (the load does not change when a designer upsizes a member):

  all         k -> k+1 for every member        (uniform upgrade)
  beam_only   beam k -> k+1, columns unchanged (strong-beam/weak-column drift)
  col_only    column k -> k+1, beams unchanged
  base_only   ground-story columns k -> k+1 only
  upper_only  every column above the ground story k -> k+1

The certificate is the audited one (engine.Scenario + min_interval_margin);
nothing about the theorem is reimplemented here.
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
from real_section_design_map import (  # noqa: E402
    BEAMS, COLUMNS, GRAVITY, TRIB_WIDTH, BETA, Mp, frame_from_sections,
)

# Ranked upgrade ladders (light -> heavy), real catalogue shapes.
BEAM_LADDER = ["W16X31", "W18X35", "W18X40", "W21X44", "W21X50", "W24X55",
               "W24X62", "W27X84", "W30X90"]
COL_LADDER = ["W10X49", "W12X65", "W14X74", "W14X90", "W14X109", "W14X132",
              "W14X176", "W14X211"]
GB_BEAM_LADDER = ["HN300X150", "HN350X175", "HN400X200", "HN450X200",
                  "HN500X200", "HN600X200"]
GB_COL_LADDER = ["HW300X300", "HW350X350", "HW400X400"]


def next_up(ladder: list[str], name: str) -> str | None:
    if name not in ladder:
        return None
    i = ladder.index(name)
    return ladder[i + 1] if i + 1 < len(ladder) else None


def build_with_overrides(H, B, span, height, beam, column, overrides):
    """Frame with per-member section substitution; capacities recomputed.

    overrides: list of (kind, story1, grid, new_section) applied on top.
    Demand is unchanged, which is the point of a strength intervention.
    """
    f, meta = frame_from_sections(H, B, span, height, beam, column)
    for kind, r, g, newsec in overrides:
        for i, m in enumerate(f.members):
            if m[0] == kind and m[1] == r and m[2] == g:
                newMp = Mp(newsec)
                f.members[i][3] = newMp
                f.members[i][4] = newMp
                f.members[i][5] = (8.0 if kind == "b" else 14.0) * newMp
    f.validate()
    return f, meta


def interventions(H, B, beam, column, upgrade):
    """Return {label: overrides} for one design point."""
    nb = next_up(BEAM_LADDER if "W" in beam else GB_BEAM_LADDER, beam)
    nc = next_up(COL_LADDER if "W" in column else GB_COL_LADDER, column)
    out = {}
    if nb and nc:
        out["all"] = [
            *[("b", r, g, nb) for r in range(1, H + 1) for g in range(B)],
            *[("c", r, g, nc) for r in range(1, H + 1) for g in range(B + 1)],
        ]
    if nb:
        out["beam_only"] = [("b", r, g, nb) for r in range(1, H + 1) for g in range(B)]
    if nc:
        out["col_only"] = [("c", r, g, nc) for r in range(1, H + 1) for g in range(B + 1)]
        out["base_only"] = [("c", 1, g, nc) for g in range(B + 1)]
        if H > 1:
            out["upper_only"] = [
                ("c", r, g, nc) for r in range(2, H + 1) for g in range(B + 1)
            ]
    return out


def design_point(H, B, span, height, beam, column):
    """Analyse a baseline frame and each intervention over all removals."""
    f0, meta = frame_from_sections(H, B, span, height, beam, column)
    rows = []

    def scan(frame, label):
        worst_margin = np.inf
        worst_removal = None
        n_inexact = 0
        max_rel_gap = 0.0
        for removal in frame.scenarios():
            sc = Scenario(frame, removal)
            x = np.ones(frame.E)
            full = float(sc.chain(x)["capacity"])
            loc = float(sc.chain(x, local=True)["capacity"])
            margin = float(min_interval_margin(sc, x))
            if margin < worst_margin:
                worst_margin = margin
                worst_removal = list(removal)
            if margin < -1.0e-8:
                n_inexact += 1
                max_rel_gap = max(max_rel_gap, (loc - full) / max(full, 1e-12))
        return {
            "treatment": label,
            "worst_margin": float(worst_margin),
            "worst_removal": worst_removal,
            "n_inexact_removals": int(n_inexact),
            "n_removals": len(frame.scenarios()),
            "exact_all_removals": n_inexact == 0,
            "max_relative_gap": float(max_rel_gap),
        }

    base = scan(f0, "baseline")
    rows.append(base)
    base_exact = base["exact_all_removals"]

    for label, ov in interventions(H, B, beam, column, None).items():
        f1, _ = build_with_overrides(H, B, span, height, beam, column, ov)
        r = scan(f1, label)
        r["opens_gap_from_clean_baseline"] = bool(base_exact and not r["exact_all_removals"])
        rows.append(r)

    for r in rows:
        r.update({
            "H": H, "B": B, "span_m": span, "height_m": height,
            "baseline_beam": beam, "baseline_column": column,
            "kappa_baseline": meta["kappa"],
            "rho": meta["rho"],
            "baseline_exact": base_exact,
        })
    return rows


def main() -> None:
    rows = []
    # Sweep that reaches strong-beam/weak-column territory.
    for H in (2, 4, 8):
        for B in (1, 2, 3):
            for beam, column in [
                ("W18X40", "W14X132"), ("W18X40", "W14X109"),
                ("W21X50", "W14X132"), ("W21X50", "W14X109"),
                ("W24X62", "W14X132"), ("W24X62", "W14X176"),
                ("HN400X200", "HW350X350"), ("HN500X200", "HW350X350"),
            ]:
                rows.extend(design_point(H, B, 6.0, 4.0, beam, column))

    out = {"model": "catalogue-section strengthening interventions",
           "Fy_MPa": 345.0, "gravity_kNm2": GRAVITY, "rows": rows}
    dest = PKG / "data" / "real_section_interventions.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)

    tr = {}
    for r in rows:
        tr.setdefault(r["treatment"], []).append(r)

    print("\n=== per-treatment summary (all design points) ===")
    print(f"{'treatment':<12s} {'n':>4s} {'exact_all':>9s} {'opens_gap':>9s} "
          f"{'worst_margin_min':>17s}")
    for label, rs in tr.items():
        n = len(rs)
        ex = sum(1 for r in rs if r["exact_all_removals"])
        og = sum(1 for r in rs if r.get("opens_gap_from_clean_baseline"))
        wm = min(r["worst_margin"] for r in rs)
        print(f"{label:<12s} {n:>4d} {ex:>9d} {og:>9d} {wm:>17.3f}")

    print("\n=== clean baselines where an intervention opens a gap ===")
    hits = [r for r in rows
            if r.get("opens_gap_from_clean_baseline")
            and r["treatment"] != "baseline"]
    if not hits:
        print("  none")
    for r in hits[:20]:
        print(f"  H={r['H']:2d} B={r['B']} {r['baseline_beam']:>10s}/"
              f"{r['baseline_column']:<9s} kappa={r['kappa_baseline']:5.3f} "
              f"{r['treatment']:<11s} worst_margin={r['worst_margin']:10.3f} "
              f"inexact={r['n_inexact_removals']}/{r['n_removals']} "
              f"maxgap={r['max_relative_gap']:.4f}")

    print("\n=== which baselines are already inexact? ===")
    bad = [r for r in rows if r["treatment"] == "baseline" and not r["exact_all_removals"]]
    print(f"  {len(bad)} of {len(tr['baseline'])} baselines inexact at some removal")
    seen = set()
    for r in bad:
        key = (r["H"], r["B"], r["baseline_beam"], r["baseline_column"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  H={r['H']:2d} B={r['B']} {r['baseline_beam']:>10s}/"
              f"{r['baseline_column']:<9s} kappa={r['kappa_baseline']:5.3f} "
              f"inexact={r['n_inexact_removals']}/{r['n_removals']} "
              f"worst_margin={r['worst_margin']:.3f}")


if __name__ == "__main__":
    main()
