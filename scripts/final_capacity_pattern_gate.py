from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
CODE = PACKAGE / "scripts" / "code"
sys.path.insert(0, str(CODE))

from engine import Frame, Scenario, make_frame  # noqa: E402
from batch_sweep import BatchSweep  # noqa: E402


SEED = 20260925
H_VALUES = (2, 4, 6, 8, 12)
B_VALUES = (1, 2, 3, 4)
GEOMETRIES = ("equal", "unequal_bay", "unequal_height", "combined")
REGIMES = ("balanced", "weak_base", "axial")
DRAW_COUNT = 12
TREATMENTS = ("homogeneous_k2", "componentwise_strengthening", "geometric_mean_one_redistribution")


def geometry(H: int, B: int, mode: str) -> tuple[list[float], list[float]]:
    spans = np.ones(B)
    heights = np.ones(H)
    if mode in ("unequal_bay", "combined"):
        spans = np.linspace(0.65, 1.35, B)
        spans /= np.mean(spans)
    if mode in ("unequal_height", "combined"):
        heights = np.linspace(0.75, 1.25, H)
        heights /= np.mean(heights)
    return spans.tolist(), heights.tolist()


def controlled_frame(H: int, B: int, seed: int, regime: str, mode: str) -> Frame:
    f = make_frame(H, B, seed=seed, regime=regime)
    f.spans, f.heights = geometry(H, B, mode)
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    story_factor = rng.uniform(0.90, 1.10, H)
    widths = np.r_[f.spans[0] / 2.0, (np.asarray(f.spans[:-1]) + np.asarray(f.spans[1:])) / 2.0, f.spans[-1] / 2.0]
    f.loads = (story_factor[:, None] * widths[None, :]).tolist()
    f.name = f"final_{regime}_{mode}_H{H}_B{B}_seed{seed}"
    f.provenance = "deterministic capacity-pattern gate; synthetic first-order frame"
    f.validate()
    return f


def eps(a: float) -> float:
    return 1.0e-8 * max(1.0, abs(a))


def min_interval_margin(sc: Scenario, x: np.ndarray) -> float:
    H = sc.f.H
    best = float("inf")
    for _lo, _hi, t, q in sc.bands:
        V, B0, _ = sc._potentials(x, t, q)
        entry = np.empty(H)
        entry[0] = B0
        if H > 1:
            entry[1:] = V[:-1, 0, 1] - V[:-1, 0, 0]
        interior = V[:, 1, 1] - V[:, 0, 0]
        exit_cost = V[:, 1, 0] - V[:, 0, 0]
        m = np.empty(H)
        m[0] = entry[0]
        for b in range(1, H):
            m[b] = min(m[b - 1] + interior[b - 1], entry[b])
        best = min(best, float(np.min(m + exit_cost)))
    return best


def treatment_vectors(f: Frame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # Exact bounded product-one construction. Reciprocal log pairs (and one
    # neutral factor when E is odd) guarantee sum(log x)=0; post-hoc clipping
    # would invalidate the stated geometric-mean constraint.
    half = f.E // 2
    logs = np.zeros(f.E, dtype=float)
    if half:
        pair = rng.uniform(np.log(0.2), np.log(5.0), half)
        logs[:half] = pair
        logs[half:2 * half] = -pair
    rng.shuffle(logs)
    redis = np.exp(logs)
    return {
        "homogeneous_k2": np.full(f.E, 2.0),
        "componentwise_strengthening": rng.uniform(1.0, 5.0, f.E),
        "geometric_mean_one_redistribution": redis,
    }


def summarize_case(f: Frame, x: np.ndarray, baseline_clean: np.ndarray, treatment: str):
    batch = BatchSweep(f, x)
    full = batch.capacities()
    local = batch.capacities(local=True)
    gap = local - full
    eligible = baseline_clean
    opened = gap > np.maximum(1.0e-8, 1.0e-8 * np.abs(local))
    eligible_gap = opened & eligible
    best = None
    if np.any(eligible_gap):
        idx = np.argwhere(eligible_gap)
        rank = max(idx, key=lambda ij: float(gap[tuple(ij)] / max(abs(full[tuple(ij)]), 1e-12)))
        s, g = (int(rank[0]) + 1, int(rank[1]))
        best = {
            "removal": [s, g],
            "local": float(local[s - 1, g]),
            "complete": float(full[s - 1, g]),
            "gap": float(gap[s - 1, g]),
            "relative_gap": float(gap[s - 1, g] / max(abs(full[s - 1, g]), 1e-12)),
            "multipliers": x.tolist(),
            "frame": f.name,
            "treatment": treatment,
        }
    return {
        "n_gap_instances": int(np.sum(eligible_gap)),
        "n_eligible_instances": int(np.sum(eligible)),
        "any_new_gap": bool(np.any(eligible_gap)),
        "max_relative_gap": float(np.max(np.where(eligible, gap / np.maximum(np.abs(full), 1e-12), 0.0))),
        "best": best,
        "full": full,
        "local": local,
    }


def _position_class(H: int, B: int, s: int, g: int) -> str:
    story = "ground" if s == 1 else ("roof" if s == H else "interior_story")
    bay = "exterior" if g in (0, B) else "interior_bay"
    return f"{story}|{bay}"


def _record_capacity_path(
    H: int,
    B: int,
    clean: np.ndarray,
    treatment: str,
    result: dict,
    counts: dict,
    eligible: dict,
    gap_instances: dict,
    eligible_instances: dict,
    frames_with_gap: dict,
) -> None:
    if treatment == "homogeneous_k2":
        return
    full = result["full"]
    local = result["local"]
    opened = (local - full) > np.maximum(1.0e-8, 1.0e-8 * np.abs(local))
    opened &= clean
    for s in range(1, H + 1):
        for g in range(B + 1):
            if not clean[s - 1, g]:
                continue
            cls = _position_class(H, B, s, g)
            eligible[treatment][cls] += 1
            eligible_instances[treatment] += 1
            if opened[s - 1, g]:
                counts[treatment][cls] += 1
                gap_instances[treatment] += 1
    if result["any_new_gap"]:
        frames_with_gap[treatment] += 1


def _summarize_capacity_paths(
    counts: dict,
    eligible: dict,
    gap_instances: dict,
    eligible_instances: dict,
    frames_with_gap: dict,
    baseline_clean_total: int,
) -> dict:
    summary = {}
    for treatment in counts:
        by_class = {}
        for cls in sorted(eligible[treatment]):
            denom = eligible[treatment][cls]
            num = counts[treatment][cls]
            by_class[cls] = {
                "eligible_instances": int(denom),
                "gap_instances": int(num),
                "gap_fraction": float(num / denom) if denom else 0.0,
            }
        summary[treatment] = {
            "baseline_clean_frames": baseline_clean_total,
            "frames_with_new_gap": frames_with_gap[treatment],
            "frame_fraction": float(frames_with_gap[treatment] / baseline_clean_total),
            "eligible_instances": eligible_instances[treatment],
            "gap_instances": gap_instances[treatment],
            "gap_instance_fraction": float(
                gap_instances[treatment] / eligible_instances[treatment]
            ),
            "by_position_class": by_class,
        }
    return summary


def collect_capacity_paths(seed: int = SEED, draw_count: int = DRAW_COUNT) -> dict:
    """Deprecated standalone pass; use ``main(..., write_path_analysis=True)``."""
    payload = main(seed=seed, draw_count=draw_count, write_path_analysis=True, quiet=True)
    return json.loads((ROOT.parent / "data" / "capacity_path_analysis.json").read_text(encoding="utf-8"))["treatments"]


def main(seed: int = SEED, draw_count: int = DRAW_COUNT,
         out_path: Path | None = None, quiet: bool = False,
         write_path_analysis: bool = True) -> dict:
    """Run the frozen gate.

    ``seed`` and ``draw_count`` are explicit parameters so that the ensemble
    sensitivity and convergence study can re-run the identical generator at other
    sample sizes.  The defaults are the frozen ``SEED`` and ``DRAW_COUNT``, so the
    released ``final_capacity_pattern_gate.json`` is unchanged.
    """
    rng = np.random.default_rng(seed)
    group = defaultdict(lambda: {"frames": 0, "baseline_clean_frames": 0, "treatments": defaultdict(lambda: {"frames_with_new_gap": 0, "gap_instances": 0, "eligible_instances": 0, "max_relative_gap": 0.0, "best": None})})
    margin_records: list[dict] = []
    frame_records = 0
    baseline_gap_instances = 0
    path_counts = {t: defaultdict(int) for t in TREATMENTS if t != "homogeneous_k2"}
    path_eligible = {t: defaultdict(int) for t in path_counts}
    path_gap_instances = {t: 0 for t in path_counts}
    path_eligible_instances = {t: 0 for t in path_counts}
    path_frames_with_gap = {t: 0 for t in path_counts}
    baseline_clean_total = 0

    for H in H_VALUES:
        for B in B_VALUES:
            for mode in GEOMETRIES:
                for regime in REGIMES:
                    key = f"{regime}|{mode}|H{H}|B{B}"
                    for draw in range(draw_count):
                        seed = int(rng.integers(0, 2**32 - 1))
                        f = controlled_frame(H, B, seed, regime, mode)
                        x0 = np.ones(f.E)
                        base_batch = BatchSweep(f, x0)
                        base_full = base_batch.capacities()
                        base_local = base_batch.capacities(local=True)
                        baseline_gap = base_local - base_full
                        clean = np.abs(baseline_gap) <= np.maximum(1.0e-8, 1.0e-8 * np.abs(base_local))
                        baseline_clean = bool(np.all(clean))
                        frame_records += 1
                        baseline_gap_instances += int(np.sum(~clean))
                        g = group[key]
                        g["frames"] += 1
                        g["baseline_clean_frames"] += int(baseline_clean)
                        if not baseline_clean:
                            continue
                        baseline_clean_total += 1
                        vecs = treatment_vectors(f, rng)
                        for treatment, x in vecs.items():
                            result = summarize_case(f, x, clean, treatment)
                            if write_path_analysis:
                                _record_capacity_path(
                                    H, B, clean, treatment, result,
                                    path_counts, path_eligible, path_gap_instances,
                                    path_eligible_instances, path_frames_with_gap,
                                )
                            slot = g["treatments"][treatment]
                            slot["frames_with_new_gap"] += int(result["any_new_gap"])
                            slot["gap_instances"] += result["n_gap_instances"]
                            slot["eligible_instances"] += result["n_eligible_instances"]
                            slot["max_relative_gap"] = max(slot["max_relative_gap"], result["max_relative_gap"])
                            if result["best"] is not None and (slot["best"] is None or result["best"]["relative_gap"] > slot["best"]["relative_gap"]):
                                slot["best"] = result["best"]
                            # Keep a bounded independent margin sample for the
                            # two nontrivial capacity domains.
                            if treatment != "homogeneous_k2" and len(margin_records) < 6000:
                                for _ in range(min(3, H * (B + 1))):
                                    s = int(rng.integers(1, H + 1))
                                    bay = int(rng.integers(0, B + 1))
                                    sc = Scenario(f, (s, bay))
                                    local_v = sc.chain(x, local=True)["capacity"]
                                    full_v = sc.chain(x, local=False)["capacity"]
                                    margin_records.append({
                                        "treatment": treatment,
                                        "H": H,
                                        "B": B,
                                        "regime": regime,
                                        "geometry": mode,
                                        "margin": min_interval_margin(sc, x),
                                        "gap": float(local_v - full_v),
                                    })

    aggregate = {}
    for treatment in TREATMENTS:
        frames_clean = sum(int(v["baseline_clean_frames"]) for v in group.values())
        frames_new = sum(int(v["treatments"][treatment]["frames_with_new_gap"]) for v in group.values())
        gap_instances = sum(int(v["treatments"][treatment]["gap_instances"]) for v in group.values())
        eligible_instances = sum(int(v["treatments"][treatment]["eligible_instances"]) for v in group.values())
        aggregate[treatment] = {
            "baseline_clean_frames": frames_clean,
            "frames_with_new_gap": frames_new,
            "frame_fraction_with_new_gap": frames_new / max(frames_clean, 1),
            "gap_instances": gap_instances,
            "eligible_instances": eligible_instances,
            "gap_instance_fraction": gap_instances / max(eligible_instances, 1),
            "max_relative_gap": max(float(v["treatments"][treatment]["max_relative_gap"]) for v in group.values()),
        }

    margins = np.asarray([r["margin"] for r in margin_records], dtype=float)
    gaps = np.asarray([r["gap"] for r in margin_records], dtype=float)
    rho = float(spearmanr(-margins, gaps).statistic) if len(margins) > 2 else float("nan")
    payload = {
        "status": "frozen_corrected_gate",
        "seed": SEED,
        "draw_count_per_group": DRAW_COUNT,
        "H_values": H_VALUES,
        "B_values": B_VALUES,
        "geometries": GEOMETRIES,
        "regimes": REGIMES,
        "frame_records": frame_records,
        "baseline_gap_instances": baseline_gap_instances,
        "aggregate": aggregate,
        "groups": {k: {"frames": v["frames"], "baseline_clean_frames": v["baseline_clean_frames"], "treatments": dict(v["treatments"])} for k, v in group.items()},
        "margin_sample": {
            "n": len(margin_records),
            "spearman_minus_margin_gap": rho,
            "sign_confusion": int(np.sum((margins < 0) != (gaps > 1.0e-8 * np.maximum(1.0, np.abs(gaps + 1.0e-12))))) if len(margins) else None,
            "records": margin_records,
        },
    }
    out = out_path if out_path is not None else ROOT.parent / "data" / "final_capacity_pattern_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if write_path_analysis and baseline_clean_total:
        path_summary = _summarize_capacity_paths(
            path_counts, path_eligible, path_gap_instances,
            path_eligible_instances, path_frames_with_gap, baseline_clean_total,
        )
        path_payload = {
            "status": "PASS",
            "baseline_clean_frames": baseline_clean_total,
            "treatments": path_summary,
        }
        path_out = ROOT.parent / "data" / "capacity_path_analysis.json"
        path_out.write_text(json.dumps(path_payload, indent=2), encoding="utf-8")
        docs = ROOT.parent / "docs" / "CAPACITY_PATH_ANALYSIS.md"
        lines = [
            "# Capacity-path analysis",
            "",
            f"Baseline-clean frames: **{baseline_clean_total}**.",
            "",
        ]
        for treatment, s in path_summary.items():
            lines += [
                f"## {treatment}",
                "",
                f"- Frame fraction with a new gap: **{100 * s['frame_fraction']:.2f}%**",
                f"- Instance fraction with a new gap: **{100 * s['gap_instance_fraction']:.2f}%**",
                "",
                "| Position class | Eligible | Gap instances | Gap fraction |",
                "|---|---:|---:|---:|",
            ]
            for cls, row in s["by_position_class"].items():
                lines.append(
                    f"| {cls.replace('|', ' / ')} | {row['eligible_instances']} | "
                    f"{row['gap_instances']} | {100 * row['gap_fraction']:.2f}% |"
                )
            lines.append("")
        docs.write_text("\n".join(lines), encoding="utf-8")
    if not quiet:
        print(f"wrote {out}")
        print(json.dumps({"frame_records": frame_records, "baseline_gap_instances": baseline_gap_instances, "aggregate": aggregate, "margin_n": len(margin_records), "rho": rho}, indent=2))
    return payload


if __name__ == "__main__":
    main()

