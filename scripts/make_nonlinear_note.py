"""Render the nonlinear pushdown cross-check results as a reviewable note.

Reads ``data/nonlinear_pushdown.json`` and writes
``docs/NONLINEAR_PUSHDOWN_CROSS_CHECK.md`` so that the numeric claims used in the
manuscript can be re-derived and re-checked without re-running the solver.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "data" / "nonlinear_pushdown.json"
OUT = ROOT / "docs" / "NONLINEAR_PUSHDOWN_CROSS_CHECK.md"


def main() -> None:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    lines = [
        "# Nonlinear cross-check of the exactness certificate",
        "",
        f"- Solver: {payload['solver']}",
        f"- Run time: {payload['seconds']} s; steps per case: {payload['num_steps']}; "
        f"load range: 0 to {payload['lambda_max']}",
        f"- Hinge stiffness factor: {payload['hinge_stiffness_factor']} "
        f"(reference {5.0e3} kN m/rad-equivalent model units); "
        f"member inertia scale: {payload['i_scale']}",
        f"- Environment: {payload['platform']}; numpy {payload['numpy']}",
        "",
        "## What the lanes mean",
        "",
        "| Lane | Idealization | Reading |",
        "|---|---|---|",
        "| `first_order_perfectly_plastic` | small displacements, no geometric stiffness | "
        "same plastic moment capacities as the certificate; convergence to the "
        "certificate value is an implementation cross-check of the identity |",
        "| `second_order_perfectly_plastic` | consistent P-Delta added | "
        "isolates the second-order effect on the same mechanism |",
        "| `second_order_degrading` | P-Delta plus a degrading hinge backbone | "
        "upper bound on how far a practical pushdown can move below the idealized limit |",
        "",
        "## Results",
        "",
    ]
    for case in payload["cases"]:
        cert = case["certificate"]
        lines += [
            f"### `{case['case']}`",
            "",
            f"- Certificate: local = {cert['local']:.6f}, complete = {cert['complete']:.6f}, "
            f"minimum interval margin = {cert['minimum_interval_margin']:.6f}, "
            f"exact = {cert['exact_by_certificate']}",
            f"- Removal (s, g) = ({cert['removal'][0]}, {cert['removal'][1]}); "
            f"gravity factor = {case['gravity_factor']}",
            "",
            "| Lane | lambda limit | lambda / complete | yielded | steps |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in case["runs"]:
            ratio = row.get("ratio_to_complete")
            ratio_text = "-" if ratio is None else f"{ratio:.6f}"
            lines.append(
                f"| `{row['model_class']}` | {row['lambda_limit']:.6f} | {ratio_text} | "
                f"{row['num_yielded_ends']} | {row['converged_steps']} |"
            )
        lines.append("")
    lines += [
        "## Hinge-stiffness sensitivity (canonical k = 2)",
        "",
        "| Hinge stiffness factor | lambda limit |",
        "|---:|---:|",
    ]
    for row in payload["hinge_stiffness_sensitivity"]:
        lines.append(f"| {row['hinge_stiffness_factor']} | {row['lambda_limit']:.6f} |")
    lines += [
        "",
        "## Boundaries and residual bias",
        "",
        payload["boundary"],
        "",
        "Two numeric cautions are part of the record:",
        "",
        "1. In the canonical exactness case (`k = 1`) the load-controlled limit stops "
        "about 6% below the analytical rigid-plastic value because the tangent operator "
        "loses positive definiteness before the ideal mechanism is fully developed. The "
        "two separation cases (`k = 2`, `k = 5`) reproduce the analytical value "
        "`min(4k, 5) = 5` exactly, which is the verification gate used for the solver.",
        "2. The first-order lane therefore supports the *verdict* of the certificate on "
        "these cases (exact for the baseline witness, not exact after memberwise "
        "strengthening) but not the exact *magnitude* of the predicted gap. Only the "
        "verdict is quoted in the manuscript.",
        "",
        "The general-purpose nonlinear program lane in `scripts/nonlinear_cross_check.py` "
        "is retained as an auxiliary artifact. It requires a Python 3.12 interpreter with "
        "an OpenSeesPy build whose zero-length rotational hinge assembles in this "
        "topology; the committed JSON was produced under that configuration and the "
        "script is not part of the mandatory reproduction chain.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
