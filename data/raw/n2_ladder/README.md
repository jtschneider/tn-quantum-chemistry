# `data/raw/n2_ladder/`

`n2_active_space_ladder.jsonl` — the live Phase 1 store, written by
`scripts/n2_active_space_ladder.py`.

`superseded_ramped_noise_defect.jsonl` — **kept as evidence, not as a result.**
The 81 records of the first Phase 1 run, made with the version of
`DMRGSchedule.ramped` whose noise ramp finished before the sweep reached its own
bond dimension. In `CAS(10e,16o)` with Fiedler orbital ordering that schedule
converged to a state 0.19 Ha above the ground state — at every bond dimension
and both seeds at 1.600 Å, and at two of four runs at 2.400 Å — while reporting
a final discarded weight of 5e-10.

The records are correct records of what was run, and each carries its full
schedule (`dmrg.noises`, `dmrg.schedule_bond_dims`), so the defect is legible in
the data. They are preserved because the failure is the most useful thing Phase 1
produced: no diagnostic available without an exact reference detected it.

The controls that identified it are in `notes/n2-tensor-network-experiment.md`.
