#!/usr/bin/env python3
"""Bundled all-removal batched exact chain sweep for the planar frame model.

The prefix/suffix splicing construction reuses the bundled ``Frame`` and
``Scenario`` conventions. For a fixed scale vector and threshold band, the
unmarked floor potential is shared across removals; marking one beam line and
removing one column change only a constant number of node incidences. The
result is one unmarked prefix plus one marked suffix per beam line and a
constant-size splice for every removal.

Scope: capacity values for all H(B+1) removals. Full velocity and force
witnesses for every removal are outside this batched routine.
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from engine import Frame, Scenario  # noqa: E402


def _joint(A0, A1, Cm, Cp, q):
    """Bundled per-node 2x2 potential shared with Scenario._potentials."""
    V = np.zeros((2, 2))
    for a, b in itertools.product((0, 1), repeat=2):
        y, yp = a ^ q, b ^ q
        z0 = A0 + Cm * y + Cp * yp
        z1 = A1 + Cm * (1 - y) + Cp * (1 - yp)
        V[a, b] += min(z0, z1)
    return V


class BatchSweep:
    """All-removal exact chain capacities for one frame at one scale vector x."""

    def __init__(self, frame: Frame, x):
        frame.validate()
        self.f = frame
        x = np.asarray(x, float)
        if x.shape != (frame.E,) or np.any(x <= 0):
            raise ValueError('positive member scales required')
        self.x = x
        H, B = frame.H, frame.B
        # member end capacities, ORIGINAL member indexing (same convention as
        # engine.Scenario: mcap = m0 * x[ids]); member = [kind, story, grid, Mi, Mj, Np, w]
        self.mcap = np.array([[m[3], m[4]] for m in frame.members], float) * x[:, None]

        # member incidence by node, over ALL members (no removal)
        inc = {(j, r): [] for r in range(1, H + 1) for j in range(B + 1)}
        base_ends = []
        for k, (kind, er, jc, *_rest) in enumerate(frame.members):
            ni, nj = ((jc, er), (jc + 1, er)) if kind == 'b' else ((jc, er - 1), (jc, er))
            for e, node in enumerate((ni, nj)):
                if node in inc:
                    inc[node].append((k, e))
                else:
                    base_ends.append((k, e))
        self.inc = inc
        self.base_ends = base_ends
        self.B0_full = float(sum(self.mcap[k, e] for k, e in base_ends))

        # beam spans, for widths and for the marked-beam potential
        spans = np.asarray(frame.spans, float)
        self.spans = spans
        self.H, self.B = H, B

        # node bookkeeping per node: beams (k,e) and columns below/above (k,e)
        self.node_beams = {n: [(k, e) for k, e in inc[n] if frame.members[k][0] == 'b'] for n in inc}
        self.node_Cm = {n: sum(self.mcap[k, e] for k, e in inc[n]
                               if frame.members[k][0] == 'c' and frame.members[k][1] == n[1]) for n in inc}
        self.node_Cp = {n: sum(self.mcap[k, e] for k, e in inc[n]
                               if frame.members[k][0] == 'c' and frame.members[k][1] != n[1]) for n in inc}

        # threshold signs actually used: t<0 (q=1) and t>0 (q=0)
        #   t<0 : unmarked beams have anchor 0 > t  -> A0 ; marked beam goes to A1
        #   t>0 : unmarked beams have anchor 0 < t  -> A1 ; marked beam goes to A0
        self.V0 = {}
        self.Vmark = {}
        for q in (1, 0):
            self.V0[q] = self._unmarked(q)
            self.Vmark[q] = {j: self._marked(q, j) for j in range(B)}

        self._prefix_cache = {}
        self._suffix_cache = {}

    # ---------- potentials ----------
    def _unmarked(self, q):
        V = np.zeros((self.H, 2, 2))
        for (j, r), _ends in self.inc.items():
            tot = sum(self.mcap[k, e] for k, e in self.node_beams[(j, r)])
            if q == 1:                      # t<0 -> all unmarked beams in A0
                A0, A1 = tot, 0.
            else:                           # t>0 -> all unmarked beams in A1
                A0, A1 = 0., tot
            V[r - 1] += _joint(A0, A1, self.node_Cm[(j, r)], self.node_Cp[(j, r)], q)
        return V

    def _is_marked(self, k, jmark, r):
        """True if member k is the beam on grid line jmark at story r."""
        m = self.f.members[k]
        return m[0] == 'b' and m[1] == r and m[2] == jmark

    def _marked(self, q, jmark):
        """Potentials with beam line jmark carrying a nonzero anchor.

        NOTE: jmark is a GRID index, not a member index; the beam on that grid
        line has a different member index at every story.
        """
        V = self.V0[q].copy()
        for r in range(1, self.H + 1):
            for j in (jmark, jmark + 1):
                node = (j, r)
                ends = self.node_beams[node]
                moved = sum(self.mcap[k, e] for k, e in ends if self._is_marked(k, jmark, r))
                if moved == 0.:
                    continue
                tot = sum(self.mcap[k, e] for k, e in ends if not self._is_marked(k, jmark, r))
                if q == 1:                  # marked beam leaves A0 for A1
                    A0, A1 = tot, moved
                else:                       # marked beam leaves A1 for A0
                    A0, A1 = moved, tot
                Cm, Cp = self.node_Cm[node], self.node_Cp[node]
                allbeam = tot + moved
                oldA0, oldA1 = (allbeam, 0.) if q == 1 else (0., allbeam)
                V[r - 1] += _joint(A0, A1, Cm, Cp, q) - _joint(oldA0, oldA1, Cm, Cp, q)
        return V

    # ---------- prefix / suffix ----------
    def prefix(self, q):
        if q not in self._prefix_cache:
            V = self.V0[q]
            P = np.zeros((self.H + 1, 2))
            P[0] = [0., self.B0_full]
            for r in range(self.H):
                P[r + 1] = np.min(P[r][:, None] + V[r], axis=0)
            self._prefix_cache[q] = P
        return self._prefix_cache[q]

    def suffix(self, q, j):
        key = (q, j)
        if key not in self._suffix_cache:
            V = self.Vmark[q][j]
            S = np.empty((self.H + 1, 2))
            S[self.H] = [0., np.inf]
            for r in reversed(range(self.H)):
                S[r] = np.min(V[r] + S[r + 1][None, :], axis=1)
            self._suffix_cache[key] = S
        return self._suffix_cache[key]

    # ---------- corrected floor potentials for one removal ----------
    def _corrected(self, q, j, s, g):
        """Return (B0, dict{floor1based: V2x2}) with the removed column's two
        incidences deleted, and floors >= s using the marked potentials."""
        Vm = self.Vmark[q][j]
        out = {}

        def floor_v(r):                      # floor r uses marked iff r >= s
            return (Vm if r >= s else self.V0[q])[r - 1].copy()

        B0 = self.B0_full
        if s == 1:
            k_rem = self._removed_id(s, g)
            B0 = B0 - self.mcap[k_rem, 0]    # base end at node (g,0)
            node = (g, 1)
            base = self.inc[node]
            tot_beam = sum(self.mcap[k, e] for k, e in base if self.f.members[k][0] == 'b')
            Cp = sum(self.mcap[k, e] for k, e in base
                     if self.f.members[k][0] == 'c' and self.f.members[k][1] != 1)
            Cm = sum(self.mcap[k, e] for k, e in base
                     if self.f.members[k][0] == 'c' and self.f.members[k][1] == 1) - self.mcap[k_rem, 1]
            A0, A1 = self._split(q, j, tot_beam, node, k_rem)
            out[1] = self._sub(Vm[0], self._jtot(q, tot_beam, node, A0, A1), A0, A1, Cm, Cp, q)
        else:
            k_rem = self._removed_id(s, g)
            for rf in (s - 1, s):
                node = (g, rf)
                base = self.inc[node]
                tot_beam = sum(self.mcap[k, e] for k, e in base if self.f.members[k][0] == 'b')
                Cm = sum(self.mcap[k, e] for k, e in base
                         if self.f.members[k][0] == 'c' and self.f.members[k][1] == rf)
                Cp = sum(self.mcap[k, e] for k, e in base
                         if self.f.members[k][0] == 'c' and self.f.members[k][1] != rf)
                if rf == s:
                    Cm -= self.mcap[k_rem, 1]
                else:
                    Cp -= self.mcap[k_rem, 0]
                # CRITICAL: only floors r >= s carry the marked (nonzero-anchor)
                # beam. Floor s-1 is UNMARKED, so its A0/A1 split must be the
                # unmarked one; using the marked split here silently transposes
                # the correction and turns the splice into a relaxation.
                if rf >= s:
                    A0, A1 = self._split(q, j, tot_beam, node, k_rem)
                else:
                    A0, A1 = (tot_beam, 0.) if q == 1 else (0., tot_beam)
                out[rf] = self._sub(floor_v(rf), self._jtot(q, tot_beam, node, A0, A1),
                                    A0, A1, Cm, Cp, q)
        return B0, out, floor_v

    def _removed_id(self, s, g):
        for k, m in enumerate(self.f.members):
            if m[0] == 'c' and m[1] == s and m[2] == g:
                return k
        raise KeyError('removed column not found')

    def _split(self, q, j, tot_beam, node, k_rem):
        ends = self.node_beams[node]
        r = node[1]
        moved = sum(self.mcap[k, e] for k, e in ends if self._is_marked(k, j, r))
        rest = tot_beam - moved
        if q == 1:
            return rest, moved
        return moved, rest

    def _jtot(self, q, tot_beam, node, A0, A1):
        Cm, Cp = self.node_Cm[node], self.node_Cp[node]
        return _joint(A0, A1, Cm, Cp, q)

    def _sub(self, Vfloor, Vold_node, A0, A1, Cm, Cp, q):
        return Vfloor - Vold_node + _joint(A0, A1, Cm, Cp, q)

    # ---------- public ----------
    def capacities(self, local=False):
        """lambda for every removal, shape (H, B+1); index [s-1, g]."""
        f = self.f
        out = np.zeros((self.H, self.B + 1))
        for s in range(1, self.H + 1):
            for g in range(self.B + 1):
                D = sum(f.loads[r - 1][g] for r in range(s, self.H + 1))
                total = 0.
                for j, width, q in self._sides(g):
                    if local:
                        B0, corr, floor_v = self._corrected(q, j, s, g)
                        e = 0.
                        for r in range(1, self.H + 1):
                            Vr = corr.get(r, floor_v(r))
                            e += Vr[0, 0]
                        e += B0 * 0.
                    else:
                        e = self._splice(s, g, j, q)
                    total += width * e
                out[s - 1, g] = total / D
        return out

    def _sides(self, g):
        if g > 0:
            yield g - 1, 1.0 / self.spans[g - 1], 1      # negative band, q=1
        if g < self.B:
            yield g, 1.0 / self.spans[g], 0              # positive band, q=0

    def _splice(self, s, g, j, q):
        B0, corr, floor_v = self._corrected(q, j, s, g)
        S = self.suffix(q, j)
        if s == 1:
            V1 = corr[1]
            return float(np.min(B0 * np.array([0., 1.])[:, None] + V1 + S[1][None, :]))
        P = self.prefix(q)
        Vs1, Vs = corr[s - 1], corr[s]
        acc = P[s - 2][:, None] + Vs1            # (a,b)
        acc = acc[:, :, None] + Vs[None, :, :]   # (a,b,c)
        acc = acc + S[s][None, None, :]          # floors s+1..H
        return float(acc.min())


def verify(frame: Frame, x, tol=1e-9):
    """Compare the batch sweep with the bundled per-scenario chain for every removal."""
    bs = BatchSweep(frame, x)
    fast = bs.capacities()
    slow = np.zeros_like(fast)
    slow_local = np.zeros_like(fast)
    fast_local = bs.capacities(local=True)
    worst = 0.0
    for s in range(1, frame.H + 1):
        for g in range(frame.B + 1):
            sc = Scenario(frame, (s, g))
            slow[s - 1, g] = sc.chain(x)['capacity']
            slow_local[s - 1, g] = sc.chain(x, local=True)['capacity']
    worst = float(np.max(np.abs(fast - slow) / np.maximum(1.0, np.abs(slow))))
    worst_local = float(np.max(np.abs(fast_local - slow_local) / np.maximum(1.0, np.abs(slow_local))))
    return fast, slow, worst, worst_local

