# Architecture

## Data flow

The artifact separates immutable experiment definitions, generated working
data, and paper-facing outputs.

```text
configs + fixed programs
        |
        v
prepare_inputs.py --> point clouds + representative indices
        |                         |
        +------------+------------+
                     v
          upper and lower runners
                     |
                     v
            compact pair outputs
                     |
                     v
          aggregators and notebooks
                     |
                     v
              claims checker
```

`artifact.py` is the orchestration layer. It invokes the same underlying
scripts that a follow-on project can call directly. It never embeds numerical
semantics itself.

## Core modules

- `core.geometry` implements convex hulls, polygon areas, axis-aligned boxes,
  affine polygon images, and half-space clipping.
- `core.guards` defines the shared guard representation used by concrete and
  abstract execution.
- `core.programs` generates seeded affine layers and fingerprints program
  collections.
- `core.three_assert` defines the paper's control-flow shape and executes it
  on point clouds and convex polygons.
- `core.three_assert_feedback` checks useful feedback, pulls restrictions
  backward, and propagates posterior corrections to the outer join.
- `core.branch_feedback` contains deterministic program-specific half-space
  generation and reusable inverse-affine helpers.

## Reproducibility contracts

Program collections have a schema version, execution-order string, and SHA-256
fingerprint. Selections record the input-point fingerprint and complete
selection configuration. Existing results are resumed only when these fields
match.

Randomness is explicit:

- input sample `i` uses seed `283 + i`;
- representative selection for sample `i` uses seed `283 + i`;
- affine programs use seed 283;
- feedback constraints use seed 283 plus a program-specific offset.

Each parallel worker is assigned disjoint distributions/sample ranges and is
restricted to one BLAS/OpenMP thread.

## Background execution and progress

`scripts/background.py` supervises detached evaluations and holds a per-mode
exclusive lock. `artifact.py` records stage transitions atomically in
`results/<mode>/logs/progress.json`, including start/end timestamps, a unique
run ID, and failures. Worker log headers carry the same run ID.
`scripts/progress.py` reads those transitions and the current run's worker logs;
completed and validated-reused units are deduplicated, while logs from older
runs are excluded. No numerical work or large pair-summary files are read to
compute progress, and the monitor does not change experimental outputs.

The same snapshot powers one-shot CLI output, `--watch`, `--json`, and optional
loopback-only HTTP access (`/` and `/status.json`). The HTTP service is read-only
and separate from the experiment. Remaining-time estimates refer only to the
current stage; there is no misleading total percentage across unequal stages.
Legacy background runs without a stage ledger can still be inspected, but do
not receive a stage ETA.

## Output policy

Full constraint traces are disabled by default. A compact pair summary stores
final metrics and the requested checkpoints. This is sufficient for every
paper table and figure while allowing interrupted runs to resume. Large
aggregate rows are stored as gzip-compressed CSV so they remain inspectable
with standard tools and pandas.

Reference data are immutable inputs to `artifact.py check --source reference`.
New runs write only under `work/` and `results/`.
