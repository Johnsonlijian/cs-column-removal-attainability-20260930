# When is the zero-sway column removal screen exact?

This repository contains the public reproducibility layer for the manuscript
"When is the zero-sway column removal screen exact? An interval-margin
certificate for first-order frame attainability".

Zero-sway (story-localized) mechanisms are the standard fast screen for
column-removal robustness, and they are used as if they bounded the complete
first-order collapse limit. They are only an upper bound. The code here
implements the bundled first-order planar frame engine, the batched story-chain
sweep, the interval-margin attainability certificate, the frozen
capacity-pattern gate, the separate per-scenario audit, the catalogue-section
design map, and the figure generators. It contains derived JSON outputs and
figures only. Third-party SAC/Elkady–Lignos raw files and the active submission
manuscript are not included.

## Headline results reproduced here

- **Interval certificate.** The zero-sway screen equals the complete
  first-order limit if and only if every contiguous-story interval margin is
  nonnegative. Margins and their witness intervals cost O(H) per removal.
- **Catalogue-section design map** (`data/real_section_design_map.json`,
  `data/real_section_robustness.json`). Across 288 designs built from AISC and
  GB/T 11263 sections at Fy = 345 MPa, single-bay frames are never exact at
  every removal (0/96) while multi-bay frames with uniform story capacity always
  are (192/192), independently of gravity demand shape, story skeleton and
  section pair.
- **Interventions** (`data/real_section_interventions.json`). Five realistic
  strengthening strategies over 72 design points opened no gap in a frame that
  was exact beforehand.
- **Axial-demand envelope** (`data/axial_demand_envelope.json`). Imposing
  `dcr = P/(A Fy)` from 0 to 0.60 and the linear interaction
  `M_eff = M_p (1 - dcr)` leaves the split unchanged; multi-bay worst margins
  stay positive in all 45 designs at every level.
- **Conditional and unconditional transitions**
  (`data/unconditional_transition_table.json`). Over all 2,880 generated
  frames the unconditional transition rates are 14.83% (memberwise) and 40.03%
  (log-balanced redistribution), against 21.04% and 56.83% conditional on
  baseline-clean frames. The effect is not one-directional: memberwise
  strengthening restores exactness in 7 frames.

## Quick start

Use Python 3.11 or 3.12 with NumPy, SciPy and Matplotlib from this repository
root. Pinned versions are in `requirements.txt`.

```powershell
python scripts/canonical_case.py
python scripts/final_capacity_pattern_gate.py
python scripts/independent_oracle_audit.py
python scripts/verify_final_gate_case.py
python scripts/geometry_invariance_check.py
python scripts/make_mechanism_figure.py
python scripts/make_transition_figure.py
python scripts/make_public_control_figure.py
python scripts/expanded_lp_audit.py
python scripts/ensemble_sensitivity.py
python scripts/capacity_path_analysis.py
python scripts/algorithm_benchmark.py
python scripts/diamond_domain_probe.py
python scripts/certificate_sign_audit.py
python scripts/nonlinear_pushdown.py
python scripts/make_pushdown_figure.py
python scripts/make_capacity_path_figure.py
python scripts/make_diamond_figure.py
```

### R08 real-section design map

```powershell
python scripts/r08_design_map/real_section_design_map.py
python scripts/r08_design_map/real_section_interventions.py
python scripts/r08_design_map/real_section_robustness.py
python scripts/r08_design_map/real_section_axial_interaction.py
python scripts/r08_design_map/real_section_axial_envelope.py
python scripts/r08_design_map/unconditional_transition_stats.py
```

These import the packaged certificate engine rather than reimplementing it, so
every number they produce is the audited certificate applied to a new capacity
vector.

### Figures

```powershell
python scripts/r08_figures/remake_wide_figures.py
python scripts/r08_figures/make_r8_figures.py
python scripts/r08_figures/make_axial_envelope_figure.py
python scripts/r08_figures/make_diamond_figure_r8.py
python scripts/r08_figures/normalize_figure_width.py
python scripts/r08_figures/check_figure_typography.py
```

`normalize_figure_width.py` must be re-run after regenerating any figure: the
manuscript includes every body figure at `[width=0.99\linewidth]` = 142.5 mm,
and the normaliser rescales each vector master to that exact width.
`check_figure_typography.py` reports the printed point size of every label and
whether any ink reaches the page edge.

Note on `algorithm_benchmark.py`: its reported speedup is a wall-clock ratio
and is host- and load-dependent (roughly 95x-110x across runs on one machine).
The objective-value gaps it records are deterministic.

In the frozen treatments, each member factor multiplies both endpoint moment capacities; the random factor vector is memberwise.

For a fixed capacity vector, the prefix/suffix sweep evaluates the complete chain value at all H(B+1) removal positions in O(HB) arithmetic after the shared band potentials and chain caches are built. The separate local-screen path is intentionally audited independently.

The exact model and frozen outputs are dimensionless. The geometry check verifies invariance of dimensionless capacities and margins under positive story-height changes with spans, capacities, loads and topology fixed. See
`data/final_capacity_pattern_gate.json` and
`data/independent_oracle_audit.json` for the stored results.

Additional stored checks, all regenerated by the commands above:

- `data/expanded_lp_audit.json`: 912 kinematic/static LP instances over all 240 cells.
- `data/ensemble_sensitivity.json`: further seeds and sample sizes.
- `data/capacity_path_analysis.json`: removal-position gap fractions aligned with the frozen gate.
- `data/algorithm_benchmark.json`: chain sweep versus per-removal static LP.
- `data/diamond_domain_probe.json`: declared diamond yield-domain boundary.
- `data/certificate_sign_audit.json`: sign agreement on every eligible removal.
- `data/nonlinear_pushdown.json`: independent second-order plastic-hinge check.

The active submission manuscript, cover letter and third-party raw model files are not in this repository.
