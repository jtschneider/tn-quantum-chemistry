"""Experiment 5: matched surrogates for the H10 ``(R, delta)`` energy surface.

Four models are fitted to the *same* expensive labels, on the *same* splits,
with the *same* preprocessing discipline:

1. a constant correction, ``E_HF(x) + mean_train(E_FCI - E_HF)``;
2. a tensor-product cubic B-spline, least squares on the training fold;
3. a small JAX MLP trained directly on ``E_FCI``;
4. the same architecture trained on ``E_FCI - E_HF``, with HF restored at
   inference.

Models 3 and 4 differ only in the target. Anything else that differed -- a
tuned architecture, a longer schedule, a different split -- would make the
comparison unfalsifiable, which is precisely the failure mode the plan's ML
gate exists to prevent. Delta-learning is a claim to be *measured* here, not an
assumption to be implemented.

Design decisions worth stating, because none is forced by the data:

**Splits are drawn on R, never on delta.** 64% of geometries at ``delta > 0``
carry an unconverged coupled-cluster method against 33% at ``delta < 0``, so a
delta-blocked holdout loads one fold with the failures. FCI -- the actual
target -- converged at all 77 points, so this constrains how the surface is
*described*, not the target itself; splitting on R keeps the two comparable.

**Normalisation is fitted on the training fold only**, inputs and targets both.
Fitting the scaler on all 77 points would leak the test block's location and
energy scale into every model, and would do it invisibly.

**The constant correction is not a formality.** It is delta-learning with a
zero-capacity model: if the MLP trained on ``E_FCI - E_HF`` cannot beat it, the
correction was learnable by a single number and the neural model contributed
nothing. It is the control that makes the delta-learning claim falsifiable.

**The spline is allowed to extrapolate.** On the blocked-R holdout the test
block lies outside the training knot span, and a cubic tensor-product basis
extrapolates cubically. That is reported rather than clamped: the blocked
holdout exists to show which models degrade off the training support.

Nothing here is transferable to another molecule. ``(R, delta)`` parameterises
one hydrogen chain in one basis; interpolation on this surface is not evidence
about new systems, and is not described as such.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tn_quantum_chemistry.schema import (
    HARTREE_TO_KCAL_PER_MOL,
    RecordStore,
    Status,
)

#: Bumped when the meaning of a stored surrogate result changes.
RESULT_SCHEMA_VERSION = "1.0.0"


def _configure_jax() -> None:
    """Double precision, set once and before any array is created.

    Energies here are compared at the 1e-4 Ha level against totals near -5 Ha;
    float32 carries about 7 decimal digits and would put the noise floor on top
    of the signal.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SurfaceData:
    """The learning problem, extracted from the cached Experiment 4 store.

    ``x`` is ``(n, 2)`` in ``(R, delta)`` with coordinates in Angstrom;
    ``y_target`` and ``y_baseline`` are total energies in Hartree.
    """

    x: np.ndarray
    y_target: np.ndarray
    y_baseline: np.ndarray
    target_method: str
    baseline_method: str
    basis: str
    target_walltime_seconds: np.ndarray
    baseline_walltime_seconds: np.ndarray

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        if self.x.ndim != 2 or self.x.shape[1] != 2:
            raise ValueError(f"x must be (n, 2), got {self.x.shape}")
        for name in ("y_target", "y_baseline"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must be ({n},)")

    @property
    def n_points(self) -> int:
        return self.x.shape[0]

    @property
    def delta_target(self) -> np.ndarray:
        """The delta-learning target, ``E_target - E_baseline``."""
        return self.y_target - self.y_baseline


def load_surface(
    path: str | Path,
    *,
    target_method: str = "FCI",
    baseline_method: str = "RHF",
) -> SurfaceData:
    """Load the target and baseline energies from the Experiment 4 store.

    Only geometries where *both* methods converged are returned. A geometry
    where the target failed is not silently dropped: it raises, because a
    missing label changes the learning problem and must be a deliberate choice
    rather than a quiet one.
    """
    by_point: dict[tuple[float, float], dict[str, Any]] = {}
    bases: set[str] = set()
    for record in RecordStore(path):
        if record.method not in (target_method, baseline_method):
            continue
        key = (
            float(record.geometry_parameters["R"]),
            float(record.geometry_parameters["delta"]),
        )
        by_point.setdefault(key, {})[record.method] = record
        bases.add(record.basis)

    if len(bases) != 1:
        raise ValueError(f"expected one basis in the store, found {sorted(bases)}")

    rows, unusable = [], []
    for key in sorted(by_point):
        pair = by_point[key]
        target, baseline = pair.get(target_method), pair.get(baseline_method)
        if target is None or baseline is None:
            unusable.append((key, "missing method"))
            continue
        if target.status != Status.CONVERGED or baseline.status != Status.CONVERGED:
            unusable.append((key, f"{target.status}/{baseline.status}"))
            continue
        rows.append((key, target, baseline))

    if unusable:
        raise ValueError(
            f"{len(unusable)} geometries lack a converged "
            f"{target_method}/{baseline_method} pair: {unusable[:5]}"
        )

    x = np.array([[k[0], k[1]] for k, _, _ in rows], dtype=float)
    return SurfaceData(
        x=x,
        y_target=np.array([t.e_total for _, t, _ in rows], dtype=float),
        y_baseline=np.array([b.e_total for _, _, b in rows], dtype=float),
        target_method=target_method,
        baseline_method=baseline_method,
        basis=bases.pop(),
        target_walltime_seconds=np.array(
            [t.walltime_seconds or np.nan for _, t, _ in rows], dtype=float
        ),
        baseline_walltime_seconds=np.array(
            [b.walltime_seconds or np.nan for _, _, b in rows], dtype=float
        ),
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Split:
    """One train/test partition. ``train`` is the *pool*; a learning curve
    subsamples it, so every training-set size sees the same test set."""

    kind: str
    label: str
    train: np.ndarray
    test: np.ndarray
    seed: int | None = None

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise ValueError(f"train and test overlap at indices {overlap}")


def random_splits(
    n_points: int, *, test_fraction: float, n_repeats: int, seed: int
) -> list[Split]:
    """Repeated random interpolation splits.

    Interpolation on a dense two-coordinate surface is the *easy* evaluation.
    It is reported beside the blocked holdout, never instead of it.
    """
    n_test = max(1, int(round(test_fraction * n_points)))
    splits = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        permuted = rng.permutation(n_points)
        splits.append(
            Split(
                kind="random",
                label=f"random-{repeat}",
                train=np.sort(permuted[n_test:]),
                test=np.sort(permuted[:n_test]),
                seed=seed + repeat,
            )
        )
    return splits


def blocked_spacing_splits(
    x: np.ndarray, *, held_out_blocks: Sequence[Sequence[float]]
) -> list[Split]:
    """Hold out contiguous blocks of the spacing R.

    On R and not on delta: see the module docstring. Each block is a region of
    the surface the model never saw, which is a different and harder question
    than filling gaps between training points.
    """
    splits = []
    for block in held_out_blocks:
        mask = np.isin(x[:, 0], np.asarray(block, dtype=float))
        if not mask.any():
            raise ValueError(f"no geometries with R in {block}")
        splits.append(
            Split(
                kind="blocked_R",
                label="blocked-R=" + ",".join(f"{r:g}" for r in block),
                train=np.flatnonzero(~mask),
                test=np.flatnonzero(mask),
            )
        )
    return splits


def subsample(split: Split, n_train: int, *, seed: int) -> np.ndarray:
    """Draw ``n_train`` training indices from a split's pool.

    Every model at a given (split, size, draw) receives *these* indices, which
    is what makes the label budgets matched rather than merely similar.
    """
    if n_train > split.train.size:
        raise ValueError(f"asked for {n_train} of {split.train.size} pool points")
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(split.train)[:n_train])


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Standardiser:
    """Mean/standard-deviation scaling fitted on a training fold only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardiser:
        values = np.atleast_2d(values.T).T if values.ndim == 1 else values
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        # A constant column carries no information; dividing by its zero spread
        # would produce NaN rather than a warning.
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and maximum absolute error, in Hartree and kcal/mol."""
    error = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    max_abs = float(np.max(np.abs(error)))
    return {
        "mae_hartree": mae,
        "rmse_hartree": rmse,
        "max_abs_hartree": max_abs,
        "mae_kcal_per_mol": mae * HARTREE_TO_KCAL_PER_MOL,
        "rmse_kcal_per_mol": rmse * HARTREE_TO_KCAL_PER_MOL,
        "max_abs_kcal_per_mol": max_abs * HARTREE_TO_KCAL_PER_MOL,
    }


# --------------------------------------------------------------------------- #
# Model 1: constant correction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConstantCorrection:
    """``E_HF(x) + c``, with ``c`` the mean training correlation energy.

    Delta-learning with zero capacity, and the control the neural delta model
    has to beat before "delta-learning helps" means anything.
    """

    shift: float

    @classmethod
    def fit(cls, y_target: np.ndarray, y_baseline: np.ndarray) -> ConstantCorrection:
        return cls(shift=float(np.mean(y_target - y_baseline)))

    def predict(self, y_baseline: np.ndarray) -> np.ndarray:
        return np.asarray(y_baseline, dtype=float) + self.shift


# --------------------------------------------------------------------------- #
# Model 2: tensor-product cubic B-spline
# --------------------------------------------------------------------------- #


def _clamped_knots(lo: float, hi: float, n_basis: int, degree: int) -> np.ndarray:
    """Open (clamped) uniform knot vector for ``n_basis`` B-splines."""
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_basis={n_basis} too small for degree {degree}")
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    return np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])


@dataclass(frozen=True)
class TensorProductSpline:
    """Least-squares tensor-product B-spline in ``(R, delta)``.

    Fitted with an explicit basis rather than through FITPACK's automatic knot
    placement, so that the number of free parameters is a stated function of
    the training-set size instead of a solver heuristic -- the learning curve
    compares models at matched label counts, and a baseline whose capacity
    moves for its own reasons would not be comparable.

    Capacity rule: the basis is the largest tensor grid with at most
    ``n_train / 2`` functions, capped by the number of distinct coordinate
    values present, and the degree follows from it. So the baseline is bilinear
    where eight labels is all it has, and bicubic once the fold can support it.
    """

    coefficients: np.ndarray
    knots: tuple[np.ndarray, np.ndarray]
    degree: int
    n_basis: tuple[int, int]
    ridge: float

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        max_degree: int = 3,
        ridge: float = 1e-10,
    ) -> TensorProductSpline:
        n_train = x.shape[0]
        budget = max(4, n_train // 2)
        distinct = [np.unique(x[:, axis]).size for axis in (0, 1)]

        # Largest near-square basis inside the budget and the distinct-value cap.
        side = int(np.sqrt(budget))
        n_basis = [max(2, min(side, distinct[0])), max(2, min(side, distinct[1]))]
        while (n_basis[0] + 1) * n_basis[1] <= budget and n_basis[0] + 1 <= distinct[0]:
            n_basis[0] += 1
        while n_basis[0] * (n_basis[1] + 1) <= budget and n_basis[1] + 1 <= distinct[1]:
            n_basis[1] += 1

        degree = int(min(max_degree, n_basis[0] - 1, n_basis[1] - 1))
        knots = tuple(
            _clamped_knots(x[:, axis].min(), x[:, axis].max(), n_basis[axis], degree)
            for axis in (0, 1)
        )
        design = cls._design(x, knots, degree)
        gram = design.T @ design + ridge * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ y)
        return cls(
            coefficients=coefficients,
            knots=knots,
            degree=degree,
            n_basis=(n_basis[0], n_basis[1]),
            ridge=ridge,
        )

    @staticmethod
    def _design(
        x: np.ndarray, knots: tuple[np.ndarray, np.ndarray], degree: int
    ) -> np.ndarray:
        from scipy.interpolate import BSpline

        # extrapolate=True: the blocked-R holdout lies outside the training
        # knot span by construction. Cubic extrapolation is what a spline
        # baseline actually does there, and hiding it behind a clamp would
        # flatter the baseline exactly where the comparison is interesting.
        basis = [
            np.asarray(
                BSpline.design_matrix(
                    x[:, axis], knots[axis], degree, extrapolate=True
                ).todense()
            )
            for axis in (0, 1)
        ]
        return np.einsum("ij,ik->ijk", basis[0], basis[1]).reshape(x.shape[0], -1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._design(np.atleast_2d(x), self.knots, self.degree) @ self.coefficients


# --------------------------------------------------------------------------- #
# Models 3 and 4: one small JAX MLP, two targets
# --------------------------------------------------------------------------- #

#: Fixed for both neural models. No architecture search: the plan forbids it,
#: and a searched direct model against an unsearched delta model would not be
#: the matched comparison this experiment exists to make.
HIDDEN_SIZES = (64, 64)
LEARNING_RATE = 3e-3

#: Measured, not guessed. At 62 training labels (3 splits x 2 seeds) the mean
#: test MAE in Hartree runs
#:
#:     steps    2e3      4e3      1e4      2e4      4e4      8e4      1.6e5
#:     direct   5.4e-3   3.6e-3   3.0e-3   2.6e-3   2.2e-3   2.1e-3   2.5e-3
#:     delta    8.1e-3   2.6e-3   1.9e-3   1.4e-3   9.7e-4   7.9e-4   6.7e-4
#:
#: so a 4000-step budget would have compared two under-trained models and
#: called the difference a result. At 8e4 the training loss is ~1e-8, the
#: direct model is at its minimum and turns back up by 1.6e5, and the delta
#: model has flattened. Both models get this same budget at every training-set
#: size; an 8-label fold is trained exactly as long as a 62-label one, because
#: a per-size budget would be a free parameter fitted to the test set.
TRAIN_STEPS = 80000


@dataclass(frozen=True)
class MLPSurrogate:
    """A trained MLP plus the train-fold scalers it was fitted with.

    The scalers travel with the parameters because a model that is normalised
    at fit time and un-normalised by hand at predict time is one refactor away
    from silently leaking test statistics.
    """

    params: Any
    x_scaler: Standardiser
    y_scaler: Standardiser
    final_loss: float
    steps: int
    seed: int
    hidden_sizes: tuple[int, ...]

    def predict(self, x: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        scaled = self.x_scaler.transform(np.atleast_2d(x))
        raw = np.asarray(_forward(self.params, jnp.asarray(scaled)))
        return self.y_scaler.inverse(raw)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """``dE/d(R, delta)`` in Hartree per Angstrom, by autodiff.

        Generalized-coordinate derivatives of the *model*, not Cartesian atomic
        forces and not reference labels: differentiating a fitted surface is
        cheap and says nothing about the accuracy of the underlying physics.
        """
        import jax
        import jax.numpy as jnp

        scaled = jnp.asarray(self.x_scaler.transform(np.atleast_2d(x)))
        jac = jax.vmap(jax.grad(lambda row: _forward(self.params, row[None])[0]))(scaled)
        chain = self.y_scaler.scale / self.x_scaler.scale
        return np.asarray(jac) * chain


def _forward(params, x):
    import jax.numpy as jnp

    h = x
    for weight, bias in params[:-1]:
        h = jnp.tanh(h @ weight + bias)
    weight, bias = params[-1]
    return (h @ weight + bias)[..., 0]


def _init_params(key, sizes: Sequence[int]):
    import jax
    import jax.numpy as jnp

    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True):
        key, subkey = jax.random.split(key)
        scale = np.sqrt(2.0 / (n_in + n_out))  # Glorot
        params.append(
            (jax.random.normal(subkey, (n_in, n_out)) * scale, jnp.zeros(n_out))
        )
    return params


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    hidden_sizes: Sequence[int] = HIDDEN_SIZES,
    steps: int = TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> MLPSurrogate:
    """Full-batch Adam with a cosine-decayed step size, fixed budget.

    A fixed budget rather than early stopping: early stopping needs a
    validation fold, and carving one out of an 8-point training set would
    change the label budget the learning curve is supposed to hold constant.
    """
    import jax
    import jax.numpy as jnp

    _configure_jax()
    x_scaler = Standardiser.fit(x)
    y_scaler = Standardiser.fit(y)
    x_scaled = jnp.asarray(x_scaler.transform(x))
    y_scaled = jnp.asarray(y_scaler.transform(y))

    params = _init_params(
        jax.random.key(seed), (x.shape[1], *hidden_sizes, 1)
    )
    zeros = jax.tree.map(jnp.zeros_like, params)

    def loss_fn(p):
        return jnp.mean((_forward(p, x_scaled) - y_scaled) ** 2)

    def step(carry, index):
        p, first_moment, second_moment = carry
        loss, grads = jax.value_and_grad(loss_fn)(p)
        count = index + 1
        rate = learning_rate * 0.5 * (1.0 + jnp.cos(jnp.pi * index / steps))
        first_moment = jax.tree.map(
            lambda m, g: 0.9 * m + 0.1 * g, first_moment, grads
        )
        second_moment = jax.tree.map(
            lambda v, g: 0.999 * v + 0.001 * g * g, second_moment, grads
        )
        bias1 = 1.0 - 0.9**count
        bias2 = 1.0 - 0.999**count
        p = jax.tree.map(
            lambda w, m, v: w - rate * (m / bias1) / (jnp.sqrt(v / bias2) + 1e-8),
            p,
            first_moment,
            second_moment,
        )
        return (p, first_moment, second_moment), loss

    (params, _, _), losses = jax.lax.scan(
        step, (params, zeros, zeros), jnp.arange(steps)
    )
    return MLPSurrogate(
        params=jax.tree.map(np.asarray, params),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        final_loss=float(losses[-1]),
        steps=steps,
        seed=seed,
        hidden_sizes=tuple(hidden_sizes),
    )


# --------------------------------------------------------------------------- #
# Matched evaluation
# --------------------------------------------------------------------------- #

#: The four models of the plan, in the order they are reported.
MODEL_NAMES = ("constant", "spline", "mlp_direct", "mlp_delta")


@dataclass(frozen=True)
class SurrogateResult:
    """One (model, split, training-set size, seed) evaluation."""

    model: str
    split_kind: str
    split_label: str
    n_train: int
    n_test: int
    draw_seed: int
    model_seed: int | None
    metrics: dict[str, float]
    train_seconds: float
    predict_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def evaluate_models(
    data: SurfaceData,
    split: Split,
    n_train: int,
    *,
    draw_seed: int,
    model_seeds: Sequence[int],
    steps: int = TRAIN_STEPS,
) -> list[SurrogateResult]:
    """Fit all four models on one shared set of training indices.

    The indices are drawn once and handed to every model, so "matched label
    budget" is a property of the code rather than of the discipline of whoever
    ran it. The deterministic baselines are evaluated once and reported against
    every model seed's row by convention (``model_seed=None``).
    """
    train_idx = subsample(split, n_train, seed=draw_seed)
    test_idx = split.test
    x_train, x_test = data.x[train_idx], data.x[test_idx]
    y_train, y_test = data.y_target[train_idx], data.y_target[test_idx]
    base_train, base_test = data.y_baseline[train_idx], data.y_baseline[test_idx]

    results: list[SurrogateResult] = []

    def record(name, seed, predicted, train_s, predict_s, extra=None):
        results.append(
            SurrogateResult(
                model=name,
                split_kind=split.kind,
                split_label=split.label,
                n_train=int(n_train),
                n_test=int(test_idx.size),
                draw_seed=int(draw_seed),
                model_seed=seed,
                metrics=metrics(predicted, y_test),
                train_seconds=float(train_s),
                predict_seconds=float(predict_s),
                extra=extra or {},
            )
        )

    start = time.perf_counter()
    constant = ConstantCorrection.fit(y_train, base_train)
    fitted = time.perf_counter()
    predicted = constant.predict(base_test)
    record(
        "constant",
        None,
        predicted,
        fitted - start,
        time.perf_counter() - fitted,
        {"shift_hartree": constant.shift},
    )

    start = time.perf_counter()
    spline = TensorProductSpline.fit(x_train, y_train)
    fitted = time.perf_counter()
    predicted = spline.predict(x_test)
    record(
        "spline",
        None,
        predicted,
        fitted - start,
        time.perf_counter() - fitted,
        {"degree": spline.degree, "n_basis": list(spline.n_basis)},
    )

    for seed in model_seeds:
        for name, target, restore in (
            ("mlp_direct", y_train, np.zeros_like(base_test)),
            ("mlp_delta", y_train - base_train, base_test),
        ):
            start = time.perf_counter()
            model = fit_mlp(x_train, target, seed=seed, steps=steps)
            fitted = time.perf_counter()
            # Warm up on the identical call before the clock starts. The
            # first predict traces, compiles and initialises the backend, and
            # JAX recompiles per input shape -- so warming up on a different
            # shape leaves the cost in the measurement. Timing the cold call
            # reported 74 ms against 0.25 ms warm, i.e. the compiler.
            model.predict(x_test)
            fitted = time.perf_counter()
            predicted = model.predict(x_test) + restore
            record(
                name,
                int(seed),
                predicted,
                fitted - start,
                time.perf_counter() - fitted,
                {"final_train_loss": model.final_loss, "steps": model.steps},
            )
    return results


def gradient_check(
    model: MLPSurrogate, x: np.ndarray, *, step: float = 1e-4
) -> dict[str, float]:
    """Autodiff derivatives against central differences of the *same* model.

    This validates the derivative machinery, not the physics: agreement here
    says the chain rule through the scalers is right, and says nothing about
    whether the fitted surface has the correct slope.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    analytic = model.gradient(x)
    numeric = np.empty_like(analytic)
    for axis in range(x.shape[1]):
        offset = np.zeros_like(x)
        offset[:, axis] = step
        numeric[:, axis] = (
            model.predict(x + offset) - model.predict(x - offset)
        ) / (2.0 * step)
    deviation = np.abs(analytic - numeric)
    scale = np.maximum(np.abs(numeric), 1e-12)
    return {
        "max_abs_deviation": float(deviation.max()),
        "max_rel_deviation": float((deviation / scale).max()),
        "finite_difference_step": step,
        "n_points": int(x.shape[0]),
    }
