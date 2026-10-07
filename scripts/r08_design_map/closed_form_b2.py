"""Derive the closed form of the B=2 interval margins.

The interval margin is a minimum over threshold bands and story intervals of

    Delta_q[a,b] = e_a + sum_{r=a}^{b-1} d_r + f_b

and, for each band q with width w_q, the contribution is w_q * Delta_q[a,b].

The chain is a two-state chain, so its marginal cost can be written in terms of
the per-story transition potentials V_r(a,b) and the base term B0:

    entering a run at story a from the local path   e_a
    staying on the nonlocal side over stories r..r+1  d_r = V_r(1,1)-V_r(0,0)
    leaving the run after story b                    f_b = V_b(1,0)-V_b(0,0)

and V_r(a,b) is a sum of per-node minimum-cut weights

    V_r(a,b) = sum over nodes on story r of  min( d(y,yp), e(y,yp) )

with y = a ^ q, yp = b ^ q and

    d(y,yp) = A0 + C_below*y + C_above*yp
    e(y,yp) = A1 + C_below*(1-y) + C_above*(1-yp)

where A0 / A1 split the beam-end capacities at that node by whether their chord
anchor lies above or below the band threshold t, and C_below / C_above are the
column-end capacities below and above the node.

This script extracts those node-level quantities exactly for a B=2 frame so the
closed form can be read off, and checks the reconstruction against the engine's
own margin.
"""
from __future__ import annotations

import itertools
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


def build(H, B, span, height, beam_left, beam_right, col, extra_bays=None):
    """B bays; beam capacities may differ between the two cut-adjacent bays."""
    spans = [span] * B
    heights = [height] * H
    share = np.r_[span / 2, np.full(B - 1, span), span / 2]
    loads = np.outer(np.ones(H), share) * (GRAVITY * TRIB)
    members = []
    for r in range(1, H + 1):
        for g in range(B):
            mb = beam_left if g == 0 else beam_right
            members.append(["b", r, g, mb, mb, 8.0 * mb, spans[g]])
        for g in range(B + 1):
            members.append(["c", r, g, col, col, 14.0 * col, heights[r - 1]])
    f = Frame(f"cf_H{H}_B{B}", spans, heights, members,
              [list(x) for x in loads], provenance="closed-form derivation")
    f.validate()
    return f


def node_table(sc: Scenario, x: np.ndarray):
    """Per band, the node-level A0/A1/Cm/Cp and the resulting potentials."""
    out = []
    for lo, hi, t, q in sc.bands:
        if hi - lo <= 0:
            continue
        mcap = sc.m0 * np.asarray(x)[sc.ids, None]
        nodes = []
        for node, ends in sc.inc.items():
            j, r = node
            A0 = A1 = Cm = Cp = 0.0
            for k, e in ends:
                kind, er, *_ = sc.members[k]
                M = mcap[k, e]
                if kind == "b":
                    if sc.anchor[k] > t:
                        A0 += M
                    else:
                        A1 += M
                elif er == r:
                    Cm += M
                else:
                    Cp += M
            if A0 + A1 + Cm + Cp == 0:
                continue
            blocks = {}
            for a, b in itertools.product((0, 1), repeat=2):
                y, yp = a ^ q, b ^ q
                d = A0 + Cm * y + Cp * yp
                e2 = A1 + Cm * (1 - y) + Cp * (1 - yp)
                blocks[(a, b)] = (min(d, e2), int(e2 < d))
            nodes.append({"node": [int(j), int(r)], "A0": A0, "A1": A1,
                          "C_below": Cm, "C_above": Cp,
                          "blocks": {f"{a}{b}": blocks[(a, b)] for a, b in
                                     itertools.product((0, 1), repeat=2)}})
        out.append({"lo": lo, "hi": hi, "t": t, "q": q, "nodes": nodes})
    return out


def closed_form_margin(sc: Scenario, x: np.ndarray):
    """Rebuild the margin from the node table only (no chain solver).

    The engine's ``min_interval_margin`` (scripts/final_capacity_pattern_gate.py)
    returns the raw minimum interval value over all bands and intervals,

        min over q, a<=b of  e_a + sum_{r=a}^{b-1} d_r + f_b,

    with no band-width weighting. So the reconstruction must NOT multiply by the
    band width; doing so was the earlier error (it produced a constant factor of
    1/width = the span length).
    """
    V_all = []
    for band in node_table(sc, x):
        H = sc.f.H
        V = np.zeros((H, 2, 2))
        for nd in band["nodes"]:
            r = nd["node"][1]
            for a, b in itertools.product((0, 1), repeat=2):
                V[r - 1, a, b] += nd["blocks"][f"{a}{b}"][0]
        B0 = sum(sc.m0[k, e] * x[sc.ids[k]] for k, e in sc.base)
        w = band["hi"] - band["lo"]
        best = np.inf
        witness = None
        for a in range(H):
            for b in range(a, H):
                e_a = B0 if a == 0 else (V[a - 1, 0, 1] - V[a - 1, 0, 0])
                mid = sum(V[r, 1, 1] - V[r, 0, 0] for r in range(a, b))
                f_b = V[b, 1, 0] - V[b, 0, 0]
                val = e_a + mid + f_b
                if val < best:
                    best, witness = val, (a + 1, b + 1)
        V_all.append({"q": band["q"], "width": w, "t": band["t"],
                      "min_interval": best, "witness": witness})
    total = min(v["min_interval"] for v in V_all)
    return total, V_all


def main() -> None:
    Mp_ref = Mp("W18X40")
    cases = [
        ("balanced 1.0/1.0", 1.0, 1.0),
        ("imbalance 1.25/1.0", 1.25, 1.0),
        ("imbalance 1.5/1.0", 1.5, 1.0),
        ("imbalance 2.0/1.0", 2.0, 1.0),
        ("imbalance 5.0/4.0", 5.0, 4.0),
        ("imbalance 1.0/2.0", 1.0, 2.0),
    ]
    rows = []
    print(f"{'case':<22s} {'engine margin':>14s} {'closed form':>13s} "
          f"{'agree':>6s}")
    for label, fl, fr in cases:
        f = build(4, 2, 6.0, 4.0, fl * Mp_ref, fr * Mp_ref, Mp("W14X109"))
        sc = Scenario(f, (1, 0))
        x = np.ones(f.E)
        engine = float(min_interval_margin(sc, x))
        cf, detail = closed_form_margin(sc, x)
        agree = abs(engine - cf) <= 1e-6 * max(1.0, abs(engine))
        rows.append({"case": label, "beam_left": fl, "beam_right": fr,
                     "engine_margin": engine, "closed_form": cf,
                     "agree": bool(agree),
                     "bands": [{"q": d["q"], "width": d["width"],
                                "t": d["t"],
                                "min_interval": d["min_interval"],
                                "witness": d["witness"]} for d in detail]})
        print(f"{label:<22s} {engine:>14.4f} {cf:>13.4f} {str(agree):>6s}")

    print("\n=== node table for the balanced case, band by band ===")
    f = build(4, 2, 6.0, 4.0, Mp_ref, Mp_ref, Mp("W14X109"))
    sc = Scenario(f, (1, 0))
    x = np.ones(f.E)
    for band in node_table(sc, x):
        print(f"\nband t={band['t']:+.6f}  q={band['q']}  "
              f"width={band['hi'] - band['lo']:.6f}")
        for nd in band["nodes"]:
            g0 = [nd["blocks"][k][0] for k in ("00", "01", "10", "11")]
            print(f"  node (g={nd['node'][0]}, r={nd['node'][1]}): "
                  f"A0={nd['A0']:10.2f} A1={nd['A1']:10.2f} "
                  f"Cb={nd['C_below']:9.2f} Ca={nd['C_above']:9.2f} | "
                  f"V00={g0[0]:9.2f} V01={g0[1]:9.2f} "
                  f"V10={g0[2]:9.2f} V11={g0[3]:9.2f}")

    dest = PKG / "data" / "closed_form_b2.json"
    dest.write_text(json.dumps({"Mp_ref_kNm": Mp_ref, "rows": rows},
                               indent=2), encoding="utf-8")
    print("\nwrote", dest)
    n_ok = sum(1 for r in rows if r["agree"])
    print(f"\nclosed form reproduces the engine margin on {n_ok}/{len(rows)} cases")


if __name__ == "__main__":
    main()
