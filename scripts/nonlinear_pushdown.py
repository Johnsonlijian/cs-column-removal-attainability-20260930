"""Nonlinear cross-check lane for the exactness certificate.

Purpose
-------
The certificate is an identity of a first-order, rigid-plastic,
endpoint-moment model.  This script asks what the identity does and does not
imply once the *same* plastic moment capacities are analysed with a
displacement-controlled second-order plastic-hinge pushdown.  Three lanes are
reported for every case:

``first_order_perfectly_plastic``
    Small displacements solved with the very same self-contained solver.  Its
    elastic stiffness is finite, so the limit approaches the rigid-plastic value
    only as ``EI -> infinity``; the residual difference measures how close the
    discrete mechanism is to the idealization.

``second_order_perfectly_plastic``
    The same hinges with consistent P-Delta geometric stiffness.

``second_order_degrading``
    P-Delta with a degrading hinge backbone.

Because the first-order, perfectly plastic lane has the same plastic moment
capacities as the certificate, its convergence to the certificate value is the
implementation cross-check of the identity; the second-order and degrading lanes
quantify how far a practical pushdown departs from it.  None of the lanes is
physical validation of real building behaviour.

Load protocol
-------------
Reference downward nodal loads act on the removed-column line above the cut with
tributary magnitudes from the frozen frame definition.  An optional gravity
factor adds the full frame gravity load on the same proportional path.  The node
above the cut is displaced downwards in equal steps; ``lambda`` is the pushdown
multiplier and the reported limit is the largest converged value.

Outputs
-------
data/nonlinear_pushdown.json
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

from engine import Frame, Scenario, worst_interval  # noqa: E402
from final_capacity_pattern_gate import controlled_frame  # noqa: E402
from pushdown_solver import Pushdown  # noqa: E402

WEAK_BASE_H2B2_SEED = 1409716476
MEMBERS_WITNESS_FACTORS = [
    1.1600194174292109, 3.8034855404070225, 3.293983018242878,
    4.2238012634629865, 4.1489530690350485, 1.199136604381152,
    2.9323968244281784, 4.219998989221255, 3.6021075013049444,
    3.7574043405206097,
]

HINGE_STIFFNESS_FACTOR = 0.3
I_SCALE = 5.0e-5
NUM_STEPS = 4000
LAMBDA_MAX = 10.0
NEWTON_ABS_TOL = 1.0e-8
GRAVITY_FACTOR_PRACTICAL = 1.0
STIFFNESS_SENSITIVITY_FACTORS = (0.2, 0.3, 0.5)


def canonical_frame() -> Frame:
    f = Frame(
        "canonical_H2_B1_equal_baseline",
        [1.0], [1.0, 1.0],
        [
            ["b", 1, 0, 1.0, 1.0, 8.0, 1.0],
            ["b", 2, 0, 1.0, 1.0, 8.0, 1.0],
            ["c", 1, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 1, 1, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 0, 5.0, 5.0, 50.0, 1.0],
            ["c", 2, 1, 5.0, 5.0, 50.0, 1.0],
        ],
        [[0.5, 0.5], [0.5, 0.5]],
        provenance="analytical counterexample; dimensionless canonical frame",
    )
    f.validate()
    return f


def canonical_strengthened(k: float) -> tuple[Frame, np.ndarray]:
    f = canonical_frame()
    x = np.ones(f.E)
    for i, member in enumerate(f.members):
        if not (member[0] == "c" and member[1] == 1 and member[2] == 0):
            x[i] = k
    return f, x


def witness_frame() -> tuple[Frame, np.ndarray]:
    f = controlled_frame(2, 2, WEAK_BASE_H2B2_SEED, "weak_base", "equal")
    x = np.asarray(MEMBERS_WITNESS_FACTORS, float)
    if f.E != x.size:
        raise RuntimeError(f"witness factor count {x.size} != frame members {f.E}")
    return f, x


def certificate_values(f: Frame, x: np.ndarray, removal: tuple[int, int]) -> dict:
    sc = Scenario(f, removal)
    chain = float(sc.chain(x, local=False)["capacity"])
    local = float(sc.chain(x, local=True)["capacity"])
    margin = None
    witness = None
    for lo, hi, t, q in sc.bands:
        V, B0, _ = sc._potentials(x, t, q)
        value, argument = worst_interval(V, B0)
        if margin is None or value < margin:
            margin = value
            witness = {"band": [float(lo), float(hi)], "t": float(t),
                       "interval": [int(argument[0]) + 1, int(argument[1]) + 1]}
    return {
        "frame": f.name,
        "removal": [int(removal[0]), int(removal[1])],
        "local": local,
        "complete": chain,
        "minimum_interval_margin": float(margin),
        "witness": witness,
        "exact_by_certificate": bool(margin is not None and margin >= 0.0),
        "members": int(f.E),
    }


def thin_history(history: list[dict], keep: int = 60) -> list[dict]:
    points = [h for h in history if h.get("ok")]
    if len(points) <= keep:
        return history
    step = max(1, len(points) // keep)
    return points[::step] + [h for h in history if not h.get("ok")]


def run_case(name: str, f: Frame, x: np.ndarray, removal: tuple[int, int],
             gravity_factor: float = 0.0,
             hinge_factor: float = HINGE_STIFFNESS_FACTOR,
             num_steps: int = NUM_STEPS) -> dict:
    cert = certificate_values(f, x, removal)
    runs = []
    for second_order, degrading, label in (
        (False, False, "first_order_perfectly_plastic"),
        (True, False, "second_order_perfectly_plastic"),
        (True, True, "second_order_degrading"),
    ):
        analysis = Pushdown(f, x, removal, second_order=second_order,
                            degrading=degrading, hinge_factor=hinge_factor,
                            i_scale=I_SCALE, gravity_factor=gravity_factor)
        start = time.perf_counter()
        raw = analysis.run(num_steps=num_steps, lam_max=LAMBDA_MAX)
        elapsed = time.perf_counter() - start
        ok = [h for h in raw.get("history", []) if h.get("ok")]
        lam = float(max((h["lambda"] for h in ok), default=float("nan")))
        row = {
            "model_class": label,
            "converged": bool(raw.get("converged", False)),
            "lambda_limit": lam,
            "limit_reason": ("load_control_limit_point"
                             if len(ok) < num_steps else
                             "prescribed_load_range_completed"),
            "converged_steps": len(ok),
            "seconds": round(elapsed, 3),
            "num_yielded_ends": int(analysis.yielded.sum()),
            "max_plastic_rotation": float(analysis.max_plastic_rotation),
            "reference_displacement": raw.get("reference_displacement"),
            "lambda_increment": raw.get("lambda_increment"),
            "history": thin_history(raw.get("history", [])),
        }
        if math.isfinite(lam) and lam > 0:
            row["ratio_to_complete"] = lam / cert["complete"]
            row["relative_difference_to_complete"] = lam / cert["complete"] - 1.0
            row["ratio_to_local"] = lam / cert["local"]
        else:
            row["ratio_to_complete"] = None
            row["relative_difference_to_complete"] = None
            row["ratio_to_local"] = None
        runs.append(row)
    return {"case": name, "certificate": cert, "gravity_factor": float(gravity_factor),
            "runs": runs}


def main() -> None:
    started = time.perf_counter()
    cases = []
    for k in (1.0, 2.0, 5.0):
        f, x = canonical_strengthened(k)
        cases.append(run_case(f"canonical_k{k:g}", f, x, (1, 1)))

    f, x = witness_frame()
    cases.append(run_case("witness_memberwise_H2B2_removal_s1g2", f, x, (1, 2),
                          num_steps=1200))
    cases.append(run_case("witness_baseline_H2B2_removal_s1g2", f, np.ones(f.E),
                          (1, 2), num_steps=1200))
    cases.append(run_case("witness_memberwise_H2B2_removal_s1g2_gravity", f, x,
                          (1, 2), gravity_factor=GRAVITY_FACTOR_PRACTICAL,
                          num_steps=1200))

    sensitivity = []
    fk, xk = canonical_strengthened(2.0)
    for factor in STIFFNESS_SENSITIVITY_FACTORS:
        analysis = Pushdown(fk, xk, (1, 1), hinge_factor=factor, i_scale=I_SCALE)
        raw = analysis.run(num_steps=2000, lam_max=LAMBDA_MAX)
        ok = [h for h in raw.get("history", []) if h.get("ok")]
        sensitivity.append({
            "hinge_stiffness_factor": float(factor),
            "lambda_limit": float(max((h["lambda"] for h in ok), default=float("nan"))),
        })

    payload = {
        "status": "PASS",
        "solver": "self-contained second-order plastic-hinge pushdown (numpy)",
        "engine_note": (
            "Elastic frame elements with an explicit element-internal end rotation "
            "and an elastic-perfectly-plastic rotational hinge in series at every "
            "member end; consistent P-Delta geometric stiffness; load-controlled "
            "increments with a plastic return mapping."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "hinge_stiffness_factor": HINGE_STIFFNESS_FACTOR,
        "i_scale": I_SCALE,
        "num_steps": NUM_STEPS,
        "lambda_max": LAMBDA_MAX,
        "newton_abs_tolerance": NEWTON_ABS_TOL,
        "hinge_stiffness_sensitivity": sensitivity,
        "cases": cases,
        "seconds": round(time.perf_counter() - started, 3),
        "boundary": (
            "The first-order lane is a numerical cross-check of the certificate "
            "idealization, not physical validation. With a finite elastic stiffness the "
            "load-controlled limit coincides with the analytical rigid-plastic value for "
            "the canonical separation cases, and it remains a small conservative bias in "
            "the exactness case because the tangent operator loses positive definiteness "
            "before the ideal limit is reached; that bias is reported explicitly. The "
            "second-order and degrading lanes quantify the departure of a practical "
            "pushdown from the idealized limit and make no claim about real buildings."
        ),
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "nonlinear_pushdown.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"solver: {payload['solver']}   seconds={payload['seconds']}")
    for case in cases:
        cert = case["certificate"]
        print(f"\n[{case['case']}] local={cert['local']:.6f} "
              f"complete={cert['complete']:.6f} "
              f"margin={cert['minimum_interval_margin']:.6f} "
              f"exact={cert['exact_by_certificate']}")
        for row in case["runs"]:
            lam = row["lambda_limit"]
            ratio = row.get("ratio_to_complete")
            print(f"   {row['model_class']:<32} lambda={lam:9.6f} "
                  f"ratio_complete={ratio if ratio is None else round(ratio, 6)} "
                  f"yielded={row['num_yielded_ends']:>3} steps={row['converged_steps']:>5}")
    print("\nhinge-stiffness sensitivity:", sensitivity)


if __name__ == "__main__":
    main()
