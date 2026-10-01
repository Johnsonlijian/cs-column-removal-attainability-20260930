# Exact attainability certificate for column-removal screens

This repository contains the public reproducibility layer for the manuscript
“Local optimality is not global attainability after column removal: Exact
certificates and capacity-pattern transitions in planar moment frames”.

The code implements the bundled first-order planar frame engine, the batched
story-chain sweep, the corrected frozen capacity-pattern gate, the separate
per-scenario audit, and the figure generators. It contains derived JSON
outputs and figures only. Third-party SAC/Elkady–Lignos raw files and the
active submission manuscript are not included.

## Quick start

Use Python 3.11 or 3.12 with NumPy, SciPy and Matplotlib from this repository
root:

```powershell
python scripts/canonical_case.py
python scripts/final_capacity_pattern_gate.py
python scripts/independent_oracle_audit.py
python scripts/verify_final_gate_case.py
python scripts/make_mechanism_figure.py
python scripts/make_transition_figure.py
python scripts/make_public_control_figure.py
```

In the frozen treatments, each member factor multiplies both endpoint moment capacities; the random factor vector is memberwise.

The exact model and frozen outputs are dimensionless. See
`data/final_capacity_pattern_gate.json` and
`data/independent_oracle_audit.json` for the stored results.
