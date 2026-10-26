"""Reusable core utilities for the POPL precision-loss experiments.

The package keeps geometry, guard semantics, affine-program generation,
lower-bound metrics, and posterior-correction logic separate from the command
line scripts. New experiments should prefer these modules over copying helper
functions into another runner.
"""
