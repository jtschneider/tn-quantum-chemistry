"""Frozen minimal data schema for every electronic-structure calculation.

Day-1 deliverable: the record layout is fixed *before* any production data is
generated, so that curves, surfaces and ML labels all share one provenance
contract and nothing has to be regenerated to add a missing column.

One :class:`CalculationRecord` is one (system, geometry, method) calculation.
The fields are the union of what Experiments 2-5 need:

* geometry parameters *and* explicit Cartesian positions, with units;
* method, basis, reference type and frozen-core choice;
* total and electronic energy kept separate, with the nuclear repulsion stored
  alongside so the two are always reconstructible;
* FCI discrepancy where a reference exists;
* DMRG bond dimension, discarded weight and sweep metadata where applicable;
* wall time, hardware identifier and software versions.

A failed calculation is a record with ``status`` set and ``error_message``
populated. It is never dropped: a missing row and a failed row mean different
things, and only one of them is honest.

Bumping :data:`SCHEMA_VERSION`
-----------------------------
Adding an optional field with a default is a minor bump. Renaming, removing or
changing the meaning of a field is a major bump and requires regenerating or
explicitly migrating the cached data.

1.2.0 changed how :attr:`CalculationRecord.key` is computed. No stored field
changed meaning and no data was invalidated, so it is a minor bump -- but cache
identity did change, so a store written under an earlier version will not
register as cached and its rows will be recomputed on the next run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.2.0"

#: Energies are Hartree and coordinates Angstrom everywhere unless a record
#: says otherwise. Both are stored per record rather than assumed.
DEFAULT_ENERGY_UNIT = "Hartree"
DEFAULT_COORDINATE_UNIT = "Angstrom"

HARTREE_TO_KCAL_PER_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
#: Deliberately PySCF's value (``pyscf.data.nist.BOHR``, CODATA 2010) rather
#: than the current CODATA 2018 value 0.529177210903. PySCF's constant governs
#: every geometry this project builds, so using a different one here would put
#: two Bohr conventions in the same repository -- a relative difference of
#: 3e-11, harmless in an energy but exactly the kind of silent inconsistency
#: the Day-2 convention work exists to remove. ``test_conventions.py`` asserts
#: the two stay equal.
BOHR_TO_ANGSTROM = 0.52917721092


class Status(StrEnum):
    """Outcome of a calculation. Anything but ``CONVERGED`` must not be plotted
    as if it were a converged result."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReferenceType(StrEnum):
    """The orbital/reference determinant a correlated method is built on."""

    RHF = "RHF"
    UHF = "UHF"
    ROHF = "ROHF"
    CASSCF = "CASSCF"
    NONE = "none"  # semiempirical or otherwise reference-free


# --------------------------------------------------------------------------- #
# Provenance helpers (shared by every module that writes records)
# --------------------------------------------------------------------------- #

_TRACKED_PACKAGES = ("pyscf", "block2", "jax", "numpy", "scipy", "quimb", "tblite")


def software_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Versions, host and timestamp, recorded with every result."""
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
    }
    if extra:
        record.update(extra)
    return record


def hardware_id() -> str:
    """Short, stable identifier for the machine a wall time was measured on.

    Wall times are only comparable within one hardware id.
    """
    return f"{socket.gethostname()}/{platform.machine()}/{os.cpu_count()}cpu"


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


@dataclass
class CalculationRecord:
    """One (system, geometry, method) calculation, converged or not."""

    # --- identity -------------------------------------------------------- #
    system_label: str
    method: str  # "RHF", "MP2", "CCSD", "CCSD(T)", "FCI", "DMRG", "GFN2-xTB", ...
    basis: str

    # --- geometry -------------------------------------------------------- #
    atoms: list[list[Any]]  # [[symbol, x, y, z], ...]
    geometry_parameters: dict[str, float] = field(default_factory=dict)  # {"R":..,"delta":..}
    coordinate_unit: str = DEFAULT_COORDINATE_UNIT
    charge: int = 0
    spin_two_s: int = 0

    # --- Hamiltonian ----------------------------------------------------- #
    reference_type: str = ReferenceType.RHF
    n_frozen_core: int = 0
    n_orbitals: int | None = None
    n_electrons: int | None = None
    point_group_symmetry: bool = False

    # --- results --------------------------------------------------------- #
    status: str = Status.CONVERGED
    energy_unit: str = DEFAULT_ENERGY_UNIT
    e_total: float | None = None
    e_electronic: float | None = None
    e_nuclear_repulsion: float | None = None
    e_correlation: float | None = None  # e_total - e_reference, where meaningful
    fci_discrepancy: float | None = None  # e_total - e_fci, signed

    # --- DMRG-specific (None for every other method) --------------------- #
    dmrg: dict[str, Any] | None = None  # bond dims, discarded weights, sweeps, seed
    #: Symmetry-resolved bipartition data from the converged MPS, as produced by
    #: ``SectorProfile.to_dict()``. Carries the full ``(N, 2S, pg)``-labelled
    #: Schmidt spectrum, which is free to obtain once DMRG has run and cannot be
    #: recovered later without re-running the solver. Roughly 16 kB of JSON per
    #: H10 geometry.
    sector_profile: dict[str, Any] | None = None

    # --- cost and provenance --------------------------------------------- #
    walltime_seconds: float | None = None
    hardware_id: str = field(default_factory=hardware_id)
    software: dict[str, Any] = field(default_factory=software_provenance)
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        # Keep the energy decomposition self-consistent rather than trusting
        # callers to fill in two of three fields correctly.
        if (
            self.e_electronic is None
            and self.e_total is not None
            and self.e_nuclear_repulsion is not None
        ):
            self.e_electronic = self.e_total - self.e_nuclear_repulsion

        if self.status == Status.CONVERGED and self.e_total is None:
            raise ValueError(
                f"{self.system_label}/{self.method}: a converged record must carry e_total"
            )
        if self.status == Status.FAILED and not self.error_message:
            raise ValueError(
                f"{self.system_label}/{self.method}: a failed record must carry an error_message"
            )

    @property
    def key(self) -> str:
        """Deterministic cache key: identical physics gives an identical key.

        Built only from what *specifies* the calculation, never from what it
        produced. ``n_orbitals`` used to be included and had to be removed: it
        is determined by the basis, geometry and frozen core, so it adds no
        identifying information, but it is ``None`` on a failed record and set
        on a successful one. A failure and its successful retry therefore hashed
        differently, `replace=True` never matched, and superseded failures
        accumulated in the store beside the results that had replaced them.
        """
        payload = json.dumps(
            {
                "system": self.system_label,
                "method": self.method,
                "basis": self.basis,
                "atoms": [[a[0], *(round(float(x), 10) for x in a[1:])] for a in self.atoms],
                "unit": self.coordinate_unit,
                "charge": self.charge,
                "spin": self.spin_two_s,
                "frozen_core": self.n_frozen_core,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def converged(self) -> bool:
        return self.status == Status.CONVERGED

    def energy_in(self, unit: str) -> float | None:
        """Total energy converted out of Hartree. Never mutates the record."""
        if self.e_total is None:
            return None
        factors = {"Hartree": 1.0, "kcal/mol": HARTREE_TO_KCAL_PER_MOL, "eV": HARTREE_TO_EV}
        try:
            return self.e_total * factors[unit]
        except KeyError:
            raise ValueError(f"unknown energy unit {unit!r}, expected one of {sorted(factors)}")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculationRecord:
        data = {k: v for k, v in data.items() if k != "key"}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"record has fields absent from schema {SCHEMA_VERSION}: {sorted(unknown)}"
            )
        return cls(**data)

    @classmethod
    def failed(
        cls, system_label: str, method: str, basis: str, atoms: list[list[Any]],
        error_message: str, **kwargs: Any,
    ) -> CalculationRecord:
        """Construct the record for a calculation that did not produce an energy."""
        return cls(
            system_label=system_label, method=method, basis=basis, atoms=atoms,
            status=Status.FAILED, error_message=error_message, **kwargs,
        )


# --------------------------------------------------------------------------- #
# Append-only cache
# --------------------------------------------------------------------------- #


class RecordStore:
    """JSON-lines store of :class:`CalculationRecord`, append-only by default.

    Expensive calculations are written the moment they finish and are never
    silently overwritten: re-running a script skips keys that are already
    present unless the caller explicitly asks to replace them.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[CalculationRecord]:
        if not self.path.exists():
            return
        with self.path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield CalculationRecord.from_dict(json.loads(line))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: unreadable record: {exc}") from exc

    def keys(self) -> set[str]:
        return {record.key for record in self}

    def completed_keys(self) -> set[str]:
        """Keys that need no recomputation.

        A failed record is *not* completed. Scripts skip work by consulting this
        rather than :meth:`keys`, so re-running a generator retries the points
        that failed last time instead of treating the failure as a cached
        answer -- while still skipping everything that succeeded.
        """
        return {record.key for record in self if record.status != Status.FAILED}

    def append(self, record: CalculationRecord, *, replace: bool = False) -> bool:
        """Append ``record``. Returns False if the key was already cached.

        With ``replace=True`` the file is rewritten with the new record in place
        of the old one, so superseding a *result* is possible but never
        implicit.

        One case replaces without being asked: a record that is not a failure
        always supersedes a stored failure with the same key. A result is
        strictly better information than the failure it replaces, and keeping
        both would leave two contradictory rows for one calculation.
        """
        existing = list(self)
        by_key = {r.key: r for r in existing}
        if record.key in by_key:
            supersedes_failure = (
                by_key[record.key].status == Status.FAILED
                and record.status != Status.FAILED
            )
            if not replace and not supersedes_failure:
                return False
            rewritten = [record if r.key == record.key else r for r in existing]
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w") as handle:
                for r in rewritten:
                    handle.write(json.dumps(r.to_dict()) + "\n")
            tmp.replace(self.path)
            return True

        with self.path.open("a") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
        return True

    #: Fields too bulky to belong in a flat table; dropped by :meth:`to_table`.
    BULK_FIELDS = ("sector_profile",)

    def prune_superseded_failures(self) -> list[CalculationRecord]:
        """Drop failed records that a later successful record has replaced.

        Deliberate and reported, never automatic: the returned list is exactly
        what was removed, so a caller can print it. A failure with no live
        counterpart is always kept -- a missing row and a failed row mean
        different things, and only one of them is honest.
        """
        records = list(self)
        live = {
            record.key for record in records if record.status != Status.FAILED
        }
        keep, dropped = [], []
        for record in records:
            if record.status == Status.FAILED and record.key in live:
                dropped.append(record)
            else:
                keep.append(record)

        if dropped:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w") as handle:
                for record in keep:
                    handle.write(json.dumps(record.to_dict()) + "\n")
            tmp.replace(self.path)
        return dropped

    def to_table(self, *, include_bulk: bool = False) -> list[dict[str, Any]]:
        """Flat rows for pandas/CSV; nested dicts are JSON-encoded in place.

        The full sector profile is omitted by default: it is archival bulk, not
        a table column, and JSON-encoding it into a cell produces rows tens of
        kilobytes wide. Read it from the records themselves when it is needed.
        """
        rows = []
        for record in self:
            row = record.to_dict()
            if not include_bulk:
                for field_name in self.BULK_FIELDS:
                    row.pop(field_name, None)
            for key, value in list(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value)
            rows.append(row)
        return rows
