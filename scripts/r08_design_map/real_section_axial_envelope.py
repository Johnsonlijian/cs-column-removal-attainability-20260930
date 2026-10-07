"""Stage 4: where does axial-moment interaction break the screen conclusion?

Stage 3 showed that the catalogue-section design map survives the axial
interaction *at its own gravity demand*, because the columns there are lightly
loaded (P/(A Fy) <= 0.072). That is not a general answer: it only says the
interaction is negligible at that load level. A referee is entitled to ask
where the screen's validity actually ends.

This script therefore imposes the axial demand directly, as a design column
demand ratio

    dcr = P / (A Fy)   in {0, 0.05, 0.1, ..., 0.6},

converts it to the linear (diamond) interaction capacity reduction

    M_eff = M_p (1 - dcr),

and recomputes the certificate over the same catalogue-section design points
and bay counts. The output is the engineering envelope: the axial demand up to
which the reported screen verdict can be trusted, and the demand beyond which
it changes.

The certificate is imported unchanged, so the sweep isolates the capacity
reduction.
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

GRAVITY = 6.0
TRIB = 1.0
BETA = 1.15
DCRS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def build(H, B, span, height, beam, column, dcr):
    """Catalogue-section frame with a prescribed column axial demand ratio."""
    Mp_b = Mp(beam)
    Mp_c = Mp(column) * max(1.0 - dcr, 0.05)
    w = GRAVITY * TRIB
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(max(B - 1, 0), span), span / 2] / span
    loads = np.outer(np.ones(H), share) * (w * span)

    members = []
    for r in range(1, H + 1):
        for g in range(B):
            members.append(["b", r, g, Mp_b, Mp_b, 8.0 * Mp_b, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, Mp_c, Mp_c, 14.0 * Mp_c, heights[r - 1]])
    f = Frame(f"dcr_{dcr}_{beam}_{column}_H{H}_B{B}", spans, heights, members,
              loads.tolist(), provenance=f"prescribed dcr={dcr}")
    f.validate()
    return f


def scan(f):
    worst = np.inf
    n_bad = 0
    loc_max = 0.0
    for removal in f.scenarios():
        sc = Scenario(f, removal)
        x = np.ones(f.E)
        m = float(min_interval_margin(sc, x))
        worst = min(worst, m)
        if m < -1.0e-8:
            n_bad += 1
            loc = float(sc.chain(x, local=True)["capacity"])
            full = float(sc.chain(x)["capacity"])
            loc_max = max(loc_max, (loc - full) / max(full, 1e-12))
    return {"worst_margin": float(worst), "n_inexact": n_bad,
            "n_removals": len(f.scenarios()), "exact_all": n_bad == 0,
            "max_relative_gap": float(loc_max)}


def main() -> None:
    pairs = [("W18X40", "W14X109"), ("W21X50", "W14X132"),
             ("W24X62", "W14X176"), ("HN400X200", "HW350X350"),
             ("HN500X200", "HW350X350")]
    rows = []
    for dcr in DCRS:
        for B in (1, 2, 3):
            for H in (2, 4, 8):
                for beam, column in pairs:
                    f = build(H, B, 6.0, 4.0, beam, column, dcr)
                    s = scan(f)
                    s.update({"dcr": dcr, "H": H, "B": B, "beam": beam,
                              "column": column,
                              "kappa_eff": Mp(beam) / (Mp(column) * max(1 - dcr, 0.05))})
                    rows.append(s)

    out = {"note": "certificate versus prescribed column axial demand ratio",
           "dcrs": DCRS, "rows": rows}
    dest = PKG / "data" / "axial_demand_envelope.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)

    print("\n=== exact-at-every-removal count by bay count and axial demand ===")
    print(f"{'dcr':>5s} | " + " | ".join(f"B={B} ({sum(1 for r in rows if r['B']==B) // len(DCRS)} designs)" for B in (1, 2, 3)))
    for dcr in DCRS:
        parts = []
        for B in (1, 2, 3):
            sub = [r for r in rows if r["dcr"] == dcr and r["B"] == B]
            ok = sum(1 for r in sub if r["exact_all"])
            parts.append(f"{ok:2d}/{len(sub):2d}")
        print(f"{dcr:5.2f} | " + " | ".join(f"{p:>12s}" for p in parts))

    print("\n=== first axial demand at which multi-bay exactness degrades ===")
    for B in (2, 3):
        broken = None
        for dcr in DCRS:
            sub = [r for r in rows if r["dcr"] == dcr and r["B"] == B]
            ok = sum(1 for r in sub if r["exact_all"])
            if ok < len(sub):
                broken = (dcr, ok, len(sub))
                break
        print(f"  B={B}: " + (f"degrades at dcr={broken[0]:.2f} "
                              f"({broken[1]}/{broken[2]} still exact)"
                              if broken else "no degradation up to dcr=0.60"))

    print("\n=== worst margin vs dcr (median over designs) ===")
    for dcr in DCRS:
        sub = [r for r in rows if r["dcr"] == dcr]
        med = np.median([r["worst_margin"] for r in sub])
        mn = min(r["worst_margin"] for r in sub)
        print(f"  dcr={dcr:.2f}  median worst margin {med:10.3f}   min {mn:10.3f}")

    print("\n=== largest relative local-complete gap at high axial demand ===")
    for dcr in (0.0, 0.3, 0.6):
        sub = [r for r in rows if r["dcr"] == dcr and not r["exact_all"]]
        if sub:
            g = max(r["max_relative_gap"] for r in sub)
            print(f"  dcr={dcr:.2f}: max relative gap over inexact designs = {g:.4f}")


if __name__ == "__main__":
    main()
