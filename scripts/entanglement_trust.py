"""Do tensor-network diagnostics predict where the energy surrogate fails?

Fits the Experiment 5 surrogates at their largest label budget, records the
per-geometry held-out error, and asks whether an entanglement descriptor of the
converged MPS ranks those errors -- and whether it beats the free signals a
practitioner already has.

    uv run python scripts/entanglement_trust.py
    uv run python scripts/entanglement_trust.py --chi 64   # the cheap tier

See ``tn_quantum_chemistry.trust`` for what is and is not claimed: this measures
whether the signal carries information, not whether obtaining it is affordable.
On this system it is not -- DMRG costs more than FCI here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tn_quantum_chemistry.schema import (
    RecordStore,
    Status,
    hardware_id,
    software_provenance,
)
from tn_quantum_chemistry.surrogate import (
    TRAIN_STEPS,
    Split,
    blocked_spacing_splits,
    fit_mlp,
    load_surface,
    random_splits,
)
from tn_quantum_chemistry.trust import (
    BASELINE_SIGNALS,
    DESCRIPTORS,
    entanglement_descriptors,
    evaluate_signal,
    nearest_training_distance,
    partial_rank_correlation,
    pooled_rank_correlation,
)

BLOCKS = ([1.0, 1.2], [1.8, 2.0], [2.8, 3.0])
N_RANDOM_SPLITS = 5
TEST_FRACTION = 0.2
MODEL_SEEDS = (0, 1, 2)
SPLIT_SEED = 7


def load_descriptors(store: Path, chi: int) -> dict[tuple[float, float], dict[str, float]]:
    """Entanglement descriptors keyed by ``(R, delta)`` for one DMRG tier."""
    method = f"DMRG(chi={chi})"
    out: dict[tuple[float, float], dict[str, float]] = {}
    for record in RecordStore(store):
        if record.method != method or record.status != Status.CONVERGED:
            continue
        if not record.sector_profile:
            continue
        key = (
            round(float(record.geometry_parameters["R"]), 6),
            round(float(record.geometry_parameters["delta"]), 6),
        )
        out[key] = entanglement_descriptors(record.sector_profile)
    return out


def held_out_errors(data, split: Split, steps: int) -> dict[str, np.ndarray]:
    """Per-geometry |error| on the split's test fold, and ensemble disagreement.

    The training pool is used whole -- the largest budget on the learning curve,
    where the surrogate is at its best and any remaining failure is a property
    of the geometry rather than of label starvation.
    """
    train_idx, test_idx = split.train, split.test
    x_train, x_test = data.x[train_idx], data.x[test_idx]
    y_train, y_test = data.y_target[train_idx], data.y_target[test_idx]
    base_train, base_test = data.y_baseline[train_idx], data.y_baseline[test_idx]

    out: dict[str, np.ndarray] = {}
    for name, target, restore in (
        ("mlp_direct", y_train, np.zeros_like(base_test)),
        ("mlp_delta", y_train - base_train, base_test),
    ):
        predictions = np.stack([
            fit_mlp(x_train, target, seed=seed, steps=steps).predict(x_test) + restore
            for seed in MODEL_SEEDS
        ])
        out[f"{name}_error"] = np.abs(predictions.mean(axis=0) - y_test)
        # Disagreement across seeds: the standard free uncertainty estimate,
        # and the signal an entanglement descriptor has to beat to be useful.
        out[f"{name}_ensemble_std"] = predictions.std(axis=0)
    out["dist_to_train"] = nearest_training_distance(x_test, x_train)
    out["spacing_R"] = x_test[:, 0]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path, default=Path("data/raw/h10_surface/h10_sto-3g_surface.jsonl")
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/entanglement_trust"))
    parser.add_argument("--chi", type=int, default=500, help="which DMRG tier supplies S")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    args = parser.parse_args(argv)

    data = load_surface(args.store)
    descriptors = load_descriptors(args.store, args.chi)
    missing = [
        tuple(row) for row in data.x
        if (round(float(row[0]), 6), round(float(row[1]), 6)) not in descriptors
    ]
    if missing:
        raise SystemExit(f"no chi={args.chi} sector profile at {len(missing)} geometries")

    splits = random_splits(
        data.n_points, test_fraction=TEST_FRACTION, n_repeats=N_RANDOM_SPLITS, seed=SPLIT_SEED
    ) + blocked_spacing_splits(data.x, held_out_blocks=BLOCKS)

    evaluations, pooled_inputs, rows = [], defaultdict(list), []
    print(f"chi={args.chi} descriptors, {len(splits)} splits, {args.steps} training steps\n")
    for split in splits:
        cols = held_out_errors(data, split, args.steps)
        keys = [
            (round(float(r), 6), round(float(d), 6)) for r, d in data.x[split.test]
        ]
        ent = {name: np.array([descriptors[k][name] for k in keys]) for name in DESCRIPTORS}
        for model in ("mlp_direct", "mlp_delta"):
            error = cols[f"{model}_error"]
            signals = dict(ent)
            signals["dist_to_train"] = cols["dist_to_train"]
            signals["ensemble_std"] = cols[f"{model}_ensemble_std"]
            signals["spacing_R"] = cols["spacing_R"]
            for i, key in enumerate(keys):
                rows.append({
                    "split_label": split.label, "model": model,
                    "R": key[0], "delta": key[1],
                    "abs_error_hartree": float(error[i]),
                    **{n: float(v[i]) for n, v in signals.items()},
                })
            for name, values in signals.items():
                evaluations.append(
                    evaluate_signal(
                        values, error, signal_name=name, model=model,
                        split_label=split.label,
                    )
                )
                pooled_inputs[(model, name)].append((values, error))
        print(f"  {split.label:<20} n_test={split.test.size:>3}  "
              f"direct MAE {cols['mlp_direct_error'].mean():.3e}  "
              f"delta MAE {cols['mlp_delta_error'].mean():.3e}")

    pooled = {
        f"{model}|{name}": pooled_rank_correlation(pairs)
        for (model, name), pairs in pooled_inputs.items()
    }
    # The decisive comparison: what survives once the spacing the model already
    # consumes is controlled for.
    control = {
        f"{model}|{name}": float(np.median([
            partial_rank_correlation(sig, err, r)
            for (sig, err), (r, _) in zip(pairs, pooled_inputs[(model, "spacing_R")])
        ]))
        for (model, name), pairs in pooled_inputs.items()
    }

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "does an entanglement diagnostic predict surrogate error",
        "chi": args.chi,
        "steps": args.steps,
        "model_seeds": list(MODEL_SEEDS),
        "blocks": [list(b) for b in BLOCKS],
        "descriptors": list(DESCRIPTORS),
        "baseline_signals": list(BASELINE_SIGNALS),
        "pooled_rank_correlation": pooled,
        "partial_rank_correlation_controlling_for_R": control,
        "per_geometry": rows,
        "per_split": [vars(e) for e in evaluations],
        "hardware_id": hardware_id(),
        "software": software_provenance(),
    }
    path = args.out / f"h10_entanglement_trust_chi{args.chi}.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"\nPooled within-split rank correlation with |error| "
          f"(entanglement first, free baselines last):\n")
    print(f"{'signal':>26} {'direct':>9} {'delta':>9} | "
          f"{'direct|R':>9} {'delta|R':>9}")
    for name in list(DESCRIPTORS) + list(BASELINE_SIGNALS):
        g = lambda d, m: d.get(f"{m}|{name}", float("nan"))
        mark = "" if name in DESCRIPTORS else "  (free)"
        print(f"{name:>26} {g(pooled,'mlp_direct'):>9.3f} {g(pooled,'mlp_delta'):>9.3f} | "
              f"{g(control,'mlp_direct'):>9.3f} {g(control,'mlp_delta'):>9.3f}{mark}")
    print("\n  left: raw pooled rank correlation with |error|")
    print("  right: the same, with the spacing R partialled out (median over splits)")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
