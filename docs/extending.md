# Extending the Artifact

The source is organized so new numerical experiments can reuse the semantics
without modifying the reviewer-facing pipeline.

## Add an input distribution

1. Add a seeded sampler to `scripts/generate_triangle_points.py`.
2. Register its name in `DISTRIBUTIONS`.
3. Return all distribution parameters in the generated metadata.
4. Add the distribution to a new JSON configuration.
5. Run `python artifact.py sanity` and a small custom evaluation first.

A distribution must return a finite `n x 2` NumPy array. It need not be
uniform; representative selection operates on the concrete point locations.

## Add a representative selector

1. Implement the selector in `scripts/select_space_filling_points.py` or a
   new module.
2. Register a `SelectionAlgorithm` in
   `scripts/run_point_subset_area_batch.py`.
3. Return original point indices and metadata including selected count,
   support count, achieved covering radius, and all random parameters.
4. Keep support points unless the new policy explicitly studies another
   invariant.

Selections are computed once per input set and reused across programs.

## Change the program family

Program semantics belong in a versioned module under `core/`. A new program
schema should define:

- concrete point execution;
- abstract polygon execution;
- execution-order metadata;
- validation and collection fingerprinting;
- assertion-site names.

Do not reuse the current schema version for a different control-flow shape.
Add a separate runner and tests, then register an orchestration mode rather
than weakening existing fingerprint checks.

## Change the polygon complexity policy

The current implementation applies `refine_polygon` after every affine,
guard, and join and replaces polygons with more than five edges by their
smallest axis-aligned box. A follow-on experiment can parameterize
`edge_threshold` in the program JSON or replace the refinement operator.
Record the policy in every generated program and result schema.

## Add posterior feedback

Half-space generation is in `core.branch_feedback`; learning for the current
program is in `core.three_assert_feedback`. New feedback should keep three
operations separate:

1. determine whether the abstract result triggers an alarm;
2. determine whether concrete states establish that it is a false positive;
3. update the posterior restriction and certified lower bound without changing
   the original abstract result.

Always test the invariant `corrected_lower_bound <= true_loss` on synthetic
and generated examples.

## Use custom experiment sizes

The supplied `fast.json` and `full.json` are immutable paper configurations.
For exploratory work, copy one under `configs/`, change sample/program ranges,
and add the mode name to the CLI choices. Write results to a distinct mode
directory to avoid mixing fingerprints or partial runs.

## Debug one pair

Invoke the batch runner directly with one distribution, one sample, and one
program. Add `--save-trace` only in this small setting. The resulting trace
contains every checked constraint and is useful for studying learning dynamics,
but it scales poorly and is deliberately excluded from normal runs.
