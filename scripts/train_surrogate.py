"""Experiment 5: the matched surrogate study on the H10 ``(R, delta)`` surface.

Fits four models -- a constant correction, a tensor-product spline, an MLP on
``E_FCI`` and the same MLP on ``E_FCI - E_HF`` -- across a learning-curve
schedule, on two kinds of split, with several data draws and model seeds. All
four see identical training indices at every point of the schedule.

Two evaluations are reported and never merged:

* **random**: held-out points scattered through the grid. Interpolation on a
  dense two-coordinate surface, and the easy question.
* **blocked-R**: a contiguous block of spacings the model never saw, at each
  end of the range and once in the middle. Extrapolation off the training
  support, and the question that actually discriminates the models.

Neither is evidence of transfer to another molecule.

Splits are blocked on R rather than delta because the coupled-cluster failures
in this dataset cluster at ``delta > 0``; FCI converged everywhere, so the
target is unaffected, but a delta-blocked split would still be a strange thing
to defend. Results are cached by (model, split, size, draw, seed), so an
interrupted run resumes instead of recomputing.

Run with::

    uv run python scripts/train_surrogate.py
    uv run python scripts/train_surrogate.py --quick     # smoke, ~1 min
    uv run python scripts/train_surrogate.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tn_quantum_chemistry.schema import hardware_id, software_provenance
from tn_quantum_chemistry.surrogate import (
    RESULT_SCHEMA_VERSION,
    TRAIN_STEPS,
    Split,
    blocked_spacing_splits,
    evaluate_models,
    fit_mlp,
    gradient_check,
    load_surface,
    random_splits,
)

DEFAULT_STORE = Path("data/raw/h10_surface/h10_sto-3g_surface.jsonl")
DEFAULT_OUT = Path("data/processed/surrogate")

#: Expensive-label counts on the learning curve. The largest is the whole
#: training pool; the smallest is small enough that a spline has almost nothing
#: to work with, which is where a cheap baseline is supposed to fail.
TRAIN_SIZES = (8, 12, 16, 24, 32, 40, 50, 62)

#: Held-out spacing blocks: both ends of the range and one interior gap.
#: The ends ask for extrapolation, the middle for a genuine gap fill.
BLOCKS = ([1.0, 1.2], [1.8, 2.0], [2.8, 3.0])

N_RANDOM_SPLITS = 5
MODEL_SEEDS = (0, 1, 2)
TEST_FRACTION = 0.2


def result_key(row: dict) -> tuple:
    return (
        row["model"],
        row["split_label"],
        row["n_train"],
        row["draw_seed"],
        row["model_seed"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--quick", action="store_true", help="short smoke run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore cached rows")
    args = parser.parse_args()

    data = load_surface(args.store)
    sizes = TRAIN_SIZES
    seeds = MODEL_SEEDS
    steps = args.steps
    n_random = N_RANDOM_SPLITS
    if args.quick:
        sizes, seeds, steps, n_random = (16, 62), (0,), 2000, 2

    splits: list[Split] = random_splits(
        data.n_points, test_fraction=TEST_FRACTION, n_repeats=n_random, seed=7
    ) + blocked_spacing_splits(data.x, held_out_blocks=BLOCKS)

    jobs = [
        (split, n_train)
        for split in splits
        for n_train in sizes
        if n_train <= split.train.size
    ]
    n_fits = len(jobs) * len(seeds) * 2
    print(f"{data.n_points} geometries, {len(splits)} splits, {len(jobs)} "
          f"(split, size) points, {n_fits} MLP fits at {steps} steps")
    for split in splits:
        print(f"  {split.label:<22} pool {split.train.size:>3}  test {split.test.size:>3}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "h10_surrogate_results.jsonl"
    cached: set[tuple] = set()
    if results_path.exists() and not args.force:
        with results_path.open() as handle:
            for line in handle:
                if line.strip():
                    cached.add(result_key(json.loads(line)))
        print(f"cache: {len(cached)} rows already computed")

    started = time.perf_counter()
    written = 0
    with results_path.open("a" if cached else "w") as handle:
        for job_index, (split, n_train) in enumerate(jobs, start=1):
            # Not `hash()`: Python salts string hashing per process, so the
            # draw would change between runs and a resumed run would silently
            # train on different points than the rows already cached.
            digest = hashlib.sha256(split.label.encode()).digest()
            draw_seed = int.from_bytes(digest[:4], "big") % 1000
            pending = [
                key
                for key in (
                    (model, split.label, n_train, draw_seed, seed)
                    for seed in (None, *seeds)
                    for model in (
                        ("constant", "spline") if seed is None
                        else ("mlp_direct", "mlp_delta")
                    )
                )
                if key not in cached
            ]
            if not pending:
                continue
            rows = evaluate_models(
                data,
                split,
                n_train,
                draw_seed=draw_seed,
                model_seeds=seeds,
                steps=steps,
            )
            for row in rows:
                payload = row.to_dict()
                if result_key(payload) in cached:
                    continue
                handle.write(json.dumps(payload) + "\n")
                written += 1
            handle.flush()
            elapsed = time.perf_counter() - started
            print(
                f"[{job_index:>3}/{len(jobs)}] {split.label:<22} n_train={n_train:>3} "
                f"({elapsed / 60:5.1f} min elapsed)",
                flush=True,
            )

    # Derivative check on a model fitted to the full pool, over the whole grid.
    reference_split = splits[0]
    model = fit_mlp(
        data.x[reference_split.train],
        data.y_target[reference_split.train],
        seed=0,
        steps=steps,
    )
    derivative = gradient_check(model, data.x)

    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "n_geometries": data.n_points,
        "basis": data.basis,
        "target_method": data.target_method,
        "baseline_method": data.baseline_method,
        "train_sizes": list(sizes),
        "model_seeds": list(seeds),
        "train_steps": steps,
        "test_fraction": TEST_FRACTION,
        "blocked_R": [list(block) for block in BLOCKS],
        "rows_written": written,
        "wall_seconds": time.perf_counter() - started,
        "label_cost_seconds": {
            "target_total": float(np.nansum(data.target_walltime_seconds)),
            "target_mean_per_point": float(np.nanmean(data.target_walltime_seconds)),
            "baseline_total": float(np.nansum(data.baseline_walltime_seconds)),
            "baseline_mean_per_point": float(
                np.nanmean(data.baseline_walltime_seconds)
            ),
        },
        "gradient_check": derivative,
        "hardware_id": hardware_id(),
        "software": software_provenance(),
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nwrote {written} rows to {results_path}\n"
        f"gradient check: max rel deviation {derivative['max_rel_deviation']:.2e}\n"
        f"total {(time.perf_counter() - started) / 60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
