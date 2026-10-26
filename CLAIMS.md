# Artifact Claims

This artifact supports the empirical claims in Section 7 of the accompanying
paper, *Abstract Interpretation with Error Bounds*. It does not claim to
machine-check the paper's formal theorems; those proofs are pen-and-paper
arguments in the paper and appendix.

## Claim Checklist

| ID | Paper location | Claim | Artifact evidence | Evaluation command |
| --- | --- | --- | --- | --- |
| C1 | Section 7.2, Table 1, Figure 9 (RQ1) | Representative sets preserve substantially more concrete output area and produce tighter upper bounds than the generator baseline. | `reference_results/upper/`; the full experiment evaluates 600 input sets, 100 programs, three assertion sites, and three selection policies. | `python artifact.py check --mode full --source reference`; optionally rerun with `python artifact.py evaluate --mode full --jobs 12 --background`. |
| C2 | Section 7.3, Table 2 (RQ2) | Without feedback, the forward lower-bound logic has mean/median recovery 46.3%/47.7% at (p_t), while the lower bound at (q_3) is zero after the outer join. | `reference_results/lower/three_assert_lower_bound_pair_final_metrics.csv.gz`. | Same claim-check command; a full rerun regenerates the row data. |
| C3 | Section 7.4, Table 2 and Figure 11 (RQ3) | False-positive feedback increases mean/median recovery at (p_t) to 64.4%/71.1% and mean recovery at (q_3) to 22.8%. | Lower-bound pair-final and checkpoint rows under `reference_results/lower/`. | Same claim-check command; `python scripts/reproduce_paper.py --source reference` regenerates the tables and figures. |
| C4 | Section 7.4 | Useful feedback is sparse but effective: mean/median useful-constraint counts are 242.3/68.0 after 4000 checks per pair. | Lower-bound pair-final and checkpoint rows. | Same claim-check command. |
| C5 | Section 7.1 | The evaluation covers six input distributions, 100 input sets per distribution, 100 programs, and fixed seeds. | Configurations, program fingerprint, input generators, and summary metadata. | `python artifact.py sanity` checks the program collection; full evaluation checks row counts and fingerprints. |

## Expected Full-Experiment Values

Upper-bound values are averages over the three representative-selection
policies and all six input distributions.

| Site | Representative retention | Generator retention | Representative slack | Generator slack | Generator slack removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `assert1` | 93.2% | 33.2% | 0.35 | 9.18 | 96.2% |
| `assert2` | 91.3% | 33.8% | 0.32 | 9.30 | 96.6% |
| `assert3` | 98.0% | 74.5% | 2.56 | 31.23 | 91.8% |

Lower-bound values aggregate 60,000 input-set--program pairs.

| Location | Initial recovery mean/median | Posterior recovery mean/median |
| --- | ---: | ---: |
| (p_t) | 46.3% / 47.7% | 64.4% / 71.1% |
| (q_3) | 0.0% / 0.0% | 22.8% / 12.9% |

The checker compares rounded paper values rather than requiring byte-identical
floating-point output. Fingerprints, program counts, sample counts, and schemas
must match exactly.

## Fast-Evaluation Interpretation

The fast configuration uses all six distributions but only five input sets and
ten programs per distribution, for 300 input-set--program pairs. It is intended
to reproduce the direction of each result, not the exact full-study mean:

- representative retention exceeds generator retention at every assertion;
- representative slack is below generator slack at every assertion;
- the initial lower bound at (p_t) is nonzero on average;
- posterior recovery exceeds initial recovery at (p_t) and (q_3);
- at least one useful feedback constraint is discovered.

These properties are checked automatically by
`python artifact.py check --mode fast --source current`.
