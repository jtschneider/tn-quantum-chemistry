"""Record identity and store behaviour.

The cache key decides which calculations count as "the same". Getting it wrong
does not raise -- it quietly changes what a re-run recomputes and what a store
ends up containing -- so the rules are pinned here.
"""

from __future__ import annotations

import pytest

from tn_quantum_chemistry.schema import CalculationRecord, RecordStore, Status

ATOMS = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 1.0]]


def _record(method="FCI", **kwargs):
    defaults = {
        "system_label": "H2_R1.0",
        "method": method,
        "basis": "sto-3g",
        "atoms": ATOMS,
        "e_total": -1.1,
        "e_nuclear_repulsion": 0.5,
    }
    defaults.update(kwargs)
    return CalculationRecord(**defaults)


# --------------------------------------------------------------------------- #
# Key identity
# --------------------------------------------------------------------------- #


def test_key_ignores_results_and_depends_only_on_the_specification():
    """A failure and its successful retry must be the same calculation.

    ``n_orbitals`` was once part of the key. Being ``None`` on a failed record
    and set on a successful one, it gave the two different keys, so
    ``replace=True`` never matched and superseded failures accumulated in the
    store beside the results that had replaced them.
    """
    failed = CalculationRecord.failed(
        "H2_R1.0", "FCI", "sto-3g", ATOMS, error_message="SCF did not converge"
    )
    succeeded = _record(n_orbitals=2, n_electrons=2, walltime_seconds=1.0)
    assert failed.key == succeeded.key

    # Results must not enter the key at all.
    assert _record(e_total=-1.1).key == _record(e_total=-99.0).key
    assert _record(walltime_seconds=1.0).key == _record(walltime_seconds=500.0).key


def test_key_separates_calculations_that_really_differ():
    assert _record(method="FCI").key != _record(method="CCSD").key
    assert _record(basis="sto-3g").key != _record(basis="sto-6g").key
    assert _record(n_frozen_core=0).key != _record(n_frozen_core=1).key
    assert _record(spin_two_s=0).key != _record(spin_two_s=2).key
    moved = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 1.5]]
    assert _record(atoms=ATOMS).key != _record(atoms=moved).key


# --------------------------------------------------------------------------- #
# Store behaviour
# --------------------------------------------------------------------------- #


def test_append_is_idempotent_and_replace_is_explicit(tmp_path):
    store = RecordStore(tmp_path / "records.jsonl")
    record = _record()
    assert store.append(record) is True
    assert store.append(record) is False  # already cached, not silently doubled
    assert len(list(store)) == 1

    updated = _record(e_total=-1.2)
    assert store.append(updated, replace=True) is True
    (only,) = list(store)
    assert only.e_total == pytest.approx(-1.2)


def test_a_result_supersedes_a_stored_failure(tmp_path):
    """Re-running must retry a failed point, and its result must win.

    With the key shared between a failure and its retry, a plain append would
    otherwise reject the retry as already cached -- meaning a failed geometry
    could never be recomputed without ``--replace``.
    """
    store = RecordStore(tmp_path / "records.jsonl")
    failure = CalculationRecord.failed(
        "H2_R1.0", "FCI", "sto-3g", ATOMS, error_message="SCF did not converge"
    )
    store.append(failure)
    assert store.keys() == {failure.key}
    assert store.completed_keys() == set()  # a failure is not completed work

    assert store.append(_record(n_orbitals=2)) is True  # no replace= needed
    (only,) = list(store)
    assert only.status == Status.CONVERGED
    assert store.completed_keys() == {failure.key}

    # A result still does not silently overwrite another result.
    assert store.append(_record(e_total=-9.9)) is False


def test_prune_removes_superseded_failures_and_keeps_real_ones(tmp_path):
    """A failure with no live counterpart survives pruning.

    A missing row and a failed row mean different things; only one of them is
    honest about what happened.
    """
    store = RecordStore(tmp_path / "records.jsonl")
    superseded = CalculationRecord.failed(
        "H2_R1.0", "FCI", "sto-3g", ATOMS, error_message="transient"
    )
    still_failing = CalculationRecord.failed(
        "H2_R9.0", "CCSD", "sto-3g",
        [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 9.0]],
        error_message="amplitudes diverged",
    )
    # Write the contradictory pair directly: append() now resolves this case on
    # its own, so the state pruning exists to clean can only be built by hand --
    # it is what stores written before the key fix actually contain.
    with store.path.open("w") as handle:
        import json

        for record in (superseded, still_failing, _record(n_orbitals=2)):
            handle.write(json.dumps(record.to_dict()) + "\n")

    dropped = store.prune_superseded_failures()
    remaining = list(store)

    assert [r.method for r in dropped] == ["FCI"]
    assert len(remaining) == 2
    assert any(
        r.status == Status.FAILED and r.system_label == "H2_R9.0" for r in remaining
    )
    assert store.prune_superseded_failures() == []  # idempotent


def test_pruning_an_untouched_store_rewrites_nothing(tmp_path):
    store = RecordStore(tmp_path / "records.jsonl")
    store.append(_record())
    before = store.path.read_bytes()
    assert store.prune_superseded_failures() == []
    assert store.path.read_bytes() == before
