# Abstract Interpretation with Error Bounds: Artifact

This artifact accompanies the paper *Abstract Interpretation with Error Bounds*.
It implements the two-dimensional numerical evaluation of representative-based
upper bounds, forward lower bounds, and posterior lower-bound learning.

Experimental commands perform no network access, telemetry, privileged
operations, or modification outside the artifact directory. They may use
multiple local CPU processes and write generated data under `work/` and
`results/`. An optional read-only progress monitor listens on the local
loopback interface only; it makes no outbound network requests.

## 1. Claims Supported

The artifact supports RQ1--RQ3 from Section 7 of the paper:

- representative states tighten upper-bound estimates relative to generators;
- the feedback-free forward logic computes sound but conservative lower bounds;
- useful false-positive feedback improves those lower bounds.

See [CLAIMS.md](CLAIMS.md) for the complete claim-to-command mapping, expected
numbers, and the distinction between fast trend validation and full numerical
reproduction. Formal theorems are not machine checked by this artifact.

## 2. Requirements and Installation

### Tested environment

- Ubuntu Linux, x86-64
- Python 3.10.20
- NumPy 2.2.5
- pandas 2.3.3
- Matplotlib 3.10.9

Other operating systems and Python versions have not been validated. The code
uses only standard Python scientific-computing packages, but reviewers should
use the versions in `requirements-lock.txt` when reproducing paper numbers.

### Installation

Run all commands from the artifact root, the directory containing
`artifact.py`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

No package installation outside the virtual environment is required.

### Sanity test

```bash
python artifact.py sanity
```

Expected final lines include:

```text
smoke_core: ok
smoke_branching: ok
smoke_branch_feedback: ok
smoke_three_assert: ok
Claim check: PASS
Sanity checks: PASS
```

The command checks the main program fingerprint, concrete/abstract execution,
the polygon-complexity threshold, representative selectors, feedback
soundness, source portability, and every paper-level claim against the bundled
reference rows. It does not rerun the large experiments.

Expected time is a few minutes or less. If this command fails, do not begin the
full evaluation; inspect the reported test or missing dependency first.

## 3. Evaluation Instructions

### Fast evaluation

The recommended reviewer workflow is:

```bash
python artifact.py evaluate --mode fast
```

The fast configuration evaluates all six input distributions, five input sets
per distribution, ten programs, all three assertion sites, all three
representative-selection policies, and 2000 feedback constraints at each of
the two feedback sites. This gives 300 input-set--program pairs.

The command executes five resumable stages:

1. generate deterministic 100,000-point input sets and select representatives;
2. run the upper-bound experiment;
3. run the initial and posterior lower-bound experiment;
4. aggregate rows and generate figures/notebooks;
5. check the qualitative RQ1--RQ3 claims.

On the authors' 2.6 GHz Linux machine, this workflow completed in about 6 minutes; allow up to 30 minutes on a typical multicore Linux workstation.
All worker processes are restricted to one BLAS/OpenMP thread. Override the
number of concurrent workers with, for example:

```bash
python artifact.py evaluate --mode fast --jobs 4
```

Expected outputs:

```text
results/fast/
  upper/
  lower/
  figures/upper/
  figures/lower/
  notebooks/
  paper/                    Table 1/2 and Figure 9/11, with source manifest
  logs/
  claim_check_current.json
```

The final claim check should report `PASS` for every row. Fast evaluation
checks stable directions of improvement rather than exact full-study means.

### Stage-by-stage execution

Each stage can be run independently:

```bash
python artifact.py evaluate --mode fast --stage prepare
python artifact.py evaluate --mode fast --stage upper
python artifact.py evaluate --mode fast --stage lower
python artifact.py evaluate --mode fast --stage report
python artifact.py evaluate --mode fast --stage check
```

A successful stage may be run again. Matching outputs are skipped. A metadata,
schema, seed, or fingerprint mismatch stops with an error instead of silently
mixing experiments. Use `--force` only when intentionally replacing the
selected mode's generated results.

### Full paper reproduction

```bash
python artifact.py evaluate --mode full --jobs 12 --background
```

This returns immediately and continues after the terminal or SSH session closes.
It uses the same Python interpreter and environment as the launch command.
Check progress with:

```bash
python artifact.py status --mode full
python artifact.py status --mode full --watch --interval 10
tail -f results/full/logs/evaluation.log
```

`status` shows the run state, elapsed time, current stage, completed/expected
counts and percentages for each stage, and the last recorded activity. During
a running stage it also shows per-distribution counts, reused/computed work,
and a rough estimate of that stage's remaining time. Use `--json` for a
machine-readable snapshot including supervisor/worker PIDs and exit status.
`--watch` refreshes the display and exits on completion, failure, or interruption.
`Ctrl-C` stops monitoring, not the experiment.

Counts come from this run's completed or validated-reused worker log entries,
not from the presence of old output files. A `--force` rerun therefore does not
start at 100% because of previous results. Upper counts advance after a full
input/selection batch finishes (300 rows for the full configuration); lower
counts advance one input/program pair at a time. Percentages are **per stage**,
not an overall runtime percentage. The ETA excludes later stages, may fluctuate,
and is unavailable before enough work completes or for older runs without
stage timestamps. A quiet log while a batch computes does not by itself
indicate a stall. Completion is successful only when the recorded exit code is
zero; the report and claim-check stages must finish too.

The master log records completed shards; individual worker logs in the same
directory contain finer-grained progress. Only one evaluation per
mode can run at a time. To resume an interrupted or failed run, use the same
launch command without `--force`; completed matching results are reused.
Omit `--background` for foreground execution. The flag also works with fast mode
and individual stages.

#### Optional browser access

While the experiment runs in the background, start a separate read-only monitor:

```bash
python artifact.py status --mode full --serve 8765
```

Visit `http://127.0.0.1:8765/` for a plain-text progress view that refreshes every
10 seconds, or `http://127.0.0.1:8765/status.json` for JSON. Choose another port
if 8765 is already occupied. The monitor remains available after the experiment
finishes and cannot launch, stop, or alter it. `Ctrl-C` stops only the monitor.
It serves no files, and does not listen on public interfaces.

To keep the monitor itself alive after closing the terminal, run the following
after starting the background experiment (which creates the log directory):

```bash
nohup python -u artifact.py status --mode full --serve 8765 \
  > results/full/logs/monitor.log 2>&1 < /dev/null &
```

For an experiment on a remote host, forward the port from your own computer
using `ssh -L 8765:127.0.0.1:8765 USER@HOST`, then open the same local URL.

#### Workload and expected runtime

The full configuration uses six distributions, 100 input sets per distribution,
100 programs, three representative policies, and 2000 constraints per feedback
site. It produces the 60,000 lower-bound pairs and 540,000 upper-bound
program/site rows used in the paper.

The complete clean rerun on 2026-10-12, with the locked environment, **12 workers**
and one numerical-library thread per worker, took **5 hours 56 minutes** from
launch through the final successful claim check. This includes input generation,
representative selection, both experiments, aggregation, notebooks, and paper
tables/figures. It ran from 14:25:57 to 20:22:05 (Asia/Shanghai) and reproduced
all 52 printed numerical entries in Tables 1 and 2.

Budget approximately **6 hours on a comparable machine**, and allow **6--8 hours
or longer** on a slower or shared workstation; this is planning guidance, not a
guarantee or an additional measured range. Fewer workers, CPU contention and
storage performance can increase elapsed time. Earlier measurements on the
authors' 2.6 GHz Linux machine were approximately 1 hour 16 minutes for the
upper stage and 4 hours 12 minutes for the lower stage alone, excluding setup
and final reporting. The compact artifact runner
does not save full feedback traces and is expected to use a few GB of disk,
including approximately 0.9 GB of regenerated point clouds.

The final checker requires exact schemas, row counts, and fingerprints. Numeric
results are compared at the precision printed in the paper, avoiding
inappropriate byte-level requirements across platforms.

### Reference-data evaluation

Reviewers do not need to rerun the full experiment to inspect all 60,000 pairs.
The bundled compressed row data can be checked directly:

```bash
python artifact.py check --mode full --source reference
```

Tables, figures, and notebooks can be regenerated from those rows:

```bash
python artifact.py report --source reference
```

Outputs are written under `results/reference_report/`. The source reference
files are never overwritten.

### Reproducing the paper's tables and figures

To regenerate just **Table 1, Table 2, Figure 9, and Figure 11** from the bundled
full-study row data, without running experiments:

```bash
python scripts/reproduce_paper.py --source reference
```

After a fresh full evaluation, reproduce them from the newly computed rows:

```bash
python scripts/reproduce_paper.py --source current --mode full
```

The normal `evaluate` report stage and `artifact.py report` also generate these
four items automatically. Reference outputs go to
`results/reference_report/paper/`; fresh outputs go to `results/<mode>/paper/`.
Fast-mode outputs use the same layout but are not expected to match full-study
numbers. Use `--out-dir PATH` to choose another destination for the standalone
script.

| File | Paper item |
| --- | --- |
| `table1.csv`, `table1.tex` | Upper bounds, retention, slack, and slack reduction at the three assertions |
| `table2.csv`, `table2.tex` | Mean/median true loss, initial/posterior lower bounds and recovery at `p_t` and `q_3` |
| `figure9.png`, `figure9.pdf` | Upper-bound slack arrows relative to true loss |
| `figure11.png`, `figure11.pdf` | Two panels: recovery and useful feedback versus checks per site |
| `figure11_data.csv` | Aggregated checkpoints used by Figure 11 |
| `manifest.json` | Source paths, SHA-256 hashes, row counts, configuration, and aggregation conventions |

The tables are recalculated from individual rows, not copied from the paper.
CSV files retain numerical precision; LaTeX files use the paper's printed
precision and require `booktabs`. Table 1 averages across all three selection
policies; its slack reduction is one minus the ratio of the two mean slacks.
Table 2 aggregates per-pair recovery ratios rather than dividing mean areas.
Figure 11 counts checks **per site** (2000 per site, 4000 total per pair).
The script rejects incomplete or duplicate pair coverage.

One historical convention is retained for exact reproduction: the initial
table treats zero-true-loss recovery as zero, whereas the `p_t` curve at zero
checks excludes undefined `0/0` entries (600 of 60,000 reference pairs).
Consequently, that curve starts at approximately 0.468 rather than the table's
0.463. `figure11_data.csv` includes valid recovery counts, and `manifest.json`
records the convention. Original experiment data are not rewritten.

To test release tooling, background supervision, and the printed paper values:

```bash
python tests/smoke_release.py
python tests/smoke_progress.py
```

### Optional full traces

The default lower-bound output contains final metrics and checkpoints, but not
the 2000-step per-site trace. A trace is useful for debugging an individual
pair and expensive at scale. Enable it only for a deliberately small run by
calling the underlying runner with `--save-trace`; see
`python scripts/run_three_assert_lower_bound_batch.py --help`.

## 4. Artifact Contents and Reuse

```text
artifact.py                 unified reviewer entry point
CLAIMS.md                   paper claim checklist
paper.pdf                   paper snapshot associated with this artifact
configs/                    fixed fast and full configurations
core/                       reusable numerical semantics and feedback logic
scripts/                    generators, selectors, runners, and aggregators
tests/                      lightweight semantic and soundness checks
data/programs/              fixed 100-program collection and metadata
data/reference_selections/  archive of selected representative indices
reference_results/          fixed full-study row data and reference summaries
docs/                       architecture and extension guidance
work/                       generated inputs and pair results; not distributed
results/                    regenerated summaries, figures, logs, and checks
```

The program studied by the main experiment is:

```text
affine1;
if G1 then
    affine2;
    if G2 then assert1
    else affine3; assert2
else
    affine4;
assert3
```

Here `G1` is `x >= 0`, `G2` is `y >= 0`, and each affine command is
`v := A v + b`. Abstract states are convex polygons. After every affine,
guard, or join, a polygon with more than five edges is replaced by its smallest
axis-aligned enclosing box.

### Changing the benchmark size

Copy one of the JSON files under `configs/` and adjust its sample or program
range. The fixed seeds are part of the experiment definition. To add a
configuration as a new CLI mode, register its name in `artifact.py`.

### Adding distributions, selectors, programs, or feedback

See [docs/extending.md](docs/extending.md). The main reusable boundaries are:

- `core/geometry.py`: convex hulls, clipping, affine images, and areas;
- `core/three_assert.py`: concrete and abstract program execution;
- `core/three_assert_feedback.py`: posterior correction and propagation;
- `scripts/select_space_filling_points.py`: representative policies;
- `scripts/prepare_inputs.py`: deterministic input and selection preparation.

New experiments should preserve schema versions and record fingerprints for
every generated program collection and selection configuration.

### Reference representative selections

`data/reference_selections/selected_representatives.tar.gz` contains the
indices and metadata for all 600 input sets and three selection policies used in
the paper. The normal full pipeline recomputes them from fixed seeds. The
archive is supplied for independent equality checks or follow-on analyses.

## Troubleshooting

- **A worker fails:** inspect the corresponding file in
  `results/<mode>/logs/`. Other completed workers remain resumable.
- **Existing-result mismatch:** generated data were produced with another
  seed, schema, program range, or configuration. Use a new mode/output
  directory, or explicitly use `--force` if replacement is intended.
- **Matplotlib cannot open a display:** the scripts only save files and do not
  require an interactive display. On unusual systems set `MPLBACKEND=Agg`.
- **Out of memory:** lower `--jobs`; each worker independently loads a
  100,000-point cloud.
- **Disk growth:** verify that `--save-trace` was not enabled for a large run.

## Building the Zenodo Archive

After verification, build a clean release archive with:

```bash
python scripts/package_release.py
```

The command writes `dist/error-bounds-artifact.zip` and
`dist/error-bounds-artifact.zip.sha256`. It refreshes `SHA256SUMS` for every
included file (except the manifest itself). Generated workspaces, run results,
notebooks, figure copies, caches, virtual environments, and editor backup files
are excluded. The bundled reference rows and summaries remain
in the archive because they are the paper-result evidence used by the artifact
checks and regenerated reports. The package script does not delete local runs.
Upload the ZIP to Zenodo; the archive checksum can also be uploaded.

After downloading and extracting the ZIP, verify its contents from the artifact
root with `sha256sum -c SHA256SUMS`, then follow the installation and sanity-test
steps above. The separate scalability study is not part of this release.

## License and Citation

Source code is released under the MIT License. Citation metadata is provided in
`CITATION.cff`. The Zenodo DOI should be added to that file when the final
archive is published.
