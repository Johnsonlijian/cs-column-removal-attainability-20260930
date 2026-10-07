"""Structure of the interval certificate across every removal type.

Everything established so far about the capacity-pattern boundary rests on the
anchor structure, and that structure was read off one removal type only: the
exterior ground-storey removal, which turned out to have a single nonzero
threshold band rather than the two the general one-column argument suggests.
That is a dangerously narrow base for a general claim.

This script maps the structure over the whole removal taxonomy:

    exterior vs interior column line
    ground vs upper vs roof storey
    B = 2, 3, 4 bays
    equal vs unequal spans

and records, for each case:

    the number of distinct beam chord-rate anchors
    the number of nonzero-width threshold bands
    the band widths
    which members carry the anchors

The output is the organising picture the paper currently lacks: which removal
types share a certified structure and which do not.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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


def build(H, B, spans, heights, beam, column):
    loads = np.zeros((H, B + 1))
    # nodal tributary load for unequal spans
    edges = np.r_[0.0, np.cumsum(spans)]
    for j in range(B + 1):
        left = spans[j - 1] / 2 if j > 0 else 0.0
        right = spans[j] / 2 if j < B else 0.0
        loads[:, j] = (left + right) * GRAVITY * TRIB
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            members.append(["b", r, g, beam, beam, 8.0 * beam, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, column, column, 14.0 * column,
                            heights[r - 1]])
    f = Frame(f"struct_H{H}_B{B}", list(spans), list(heights), members,
              [list(x) for x in loads], provenance="removal taxonomy")
    f.validate()
    return f


def case_structure(sc: Scenario, x: np.ndarray):
    """Anchors, bands and which members own them."""
    anchors = [(k, sc.anchor[k]) for k in range(len(sc.members))
               if sc.anchor[k] is not None]
    anchor_values = sorted({round(a, 12) for _, a in anchors})
    bands = [(lo, hi) for lo, hi, _t, _q in sc.bands if hi - lo > 1e-12]
    margin = float(min_interval_margin(sc, x))
    return {
        "n_distinct_anchors": len(anchor_values),
        "anchor_values": [round(v, 6) for v in anchor_values],
        "n_nonzero_bands": len(bands),
        "band_widths": [round(hi - lo, 6) for lo, hi in bands],
        "margin": margin,
        "exact": margin >= -1e-8,
        "anchors": [{"member": k, "kind": sc.members[k][0],
                     "story": int(sc.members[k][1]),
                     "grid": int(sc.members[k][2]),
                     "anchor": round(a, 6)} for k, a in anchors],
    }


def main() -> None:
    beam = Mp("W18X40")
    column = Mp("W14X109")
    rows = []
    summary = Counter()

    for B in (2, 3, 4):
        for H in (2, 4, 6):
            for span_mode in ("equal", "unequal"):
                if span_mode == "equal":
                    spans = [6.0] * B
                else:
                    spans = list(np.linspace(4.0, 8.0, B))
                heights = [4.0] * H
                f = build(H, B, spans, heights, beam, column)
                x = np.ones(f.E)
                for removal in f.scenarios():
                    s, g = removal
                    sc = Scenario(f, removal)
                    st = case_structure(sc, x)
                    position = ("exterior" if g in (0, B) else "interior")
                    storey = ("ground" if s == 1
                              else ("roof" if s == H else "upper"))
                    rows.append({
                        "B": B, "H": H, "span_mode": span_mode,
                        "s": s, "g": g, "position": position,
                        "storey": storey, **st})
                    summary[(position, storey, st["n_nonzero_bands"])] += 1

    print("=== nonzero-band count by removal position and storey ===")
    print(f"  {'position':<10s} {'storey':<8s} {'bands':>6s} {'count':>7s}")
    for (pos, sty, nb), cnt in sorted(summary.items()):
        print(f"  {pos:<10s} {sty:<8s} {nb:>6d} {cnt:>7d}")

    print("\n=== exactness rate by position and storey ===")
    print(f"  {'position':<10s} {'storey':<8s} {'n':>6s} {'exact':>7s} "
          f"{'rate':>8s}")
    for pos in ("exterior", "interior"):
        for sty in ("ground", "upper", "roof"):
            sub = [r for r in rows if r["position"] == pos and r["storey"] == sty]
            if not sub:
                continue
            ne = sum(1 for r in sub if r["exact"])
            print(f"  {pos:<10s} {sty:<8s} {len(sub):>6d} {ne:>7d} "
                  f"{100.0*ne/len(sub):>7.1f}%")

    print("\n=== distinct anchors by removal type (examples) ===")
    seen = set()
    for r in rows:
        key = (r["position"], r["storey"], r["n_distinct_anchors"],
               r["n_nonzero_bands"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['position']:<9s} {r['storey']:<7s} "
              f"anchors={r['n_distinct_anchors']} "
              f"bands={r['n_nonzero_bands']} "
              f"widths={r['band_widths']}  e.g. B={r['B']} H={r['H']} "
              f"removal=({r['s']},{r['g']}) margin={r['margin']:.2f}")

    print("\n=== do unequal spans change the anchor count? ===")
    for B in (2, 3):
        for pm in ("equal", "unequal"):
            sub = [r for r in rows if r["B"] == B and r["span_mode"] == pm
                   and r["position"] == "exterior" and r["storey"] == "ground"]
            if sub:
                c = Counter(r["n_distinct_anchors"] for r in sub)
                print(f"  B={B} {pm:<8s}: anchor counts {dict(c)}")

    dest = PKG / "data" / "removal_taxonomy.json"
    dest.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
