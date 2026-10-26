"""Unit checks for fixed-bank then/else posterior learning."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.branch_feedback import (
    constraint_bank_fingerprint,
    constraints_for_program,
    evaluate_two_branch_feedback_trace,
    generate_fixed_constraint_bank,
    validate_constraint_bank,
)
from core.geometry import polygon_area
from core.guards import quadrant_guard_spec


def main() -> None:
    bank = generate_fixed_constraint_bank(
        count=100,
        seed=283,
        num_blocks=2,
        projection_radius=10.0,
        central_radius=2.0,
    )
    validate_constraint_bank(bank)
    fingerprint = constraint_bank_fingerprint(bank)
    bank["fingerprint"] = fingerprint
    assert constraint_bank_fingerprint(bank) == fingerprint
    block0, block_id0 = constraints_for_program(bank, 0, 50)
    block1, block_id1 = constraints_for_program(bank, 1, 50)
    assert block_id0 == 0 and block_id1 == 1
    assert {item["constraint_id"] for item in block0}.isdisjoint(
        {item["constraint_id"] for item in block1}
    )

    intermediate_polygon = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    intermediate_true_points = np.array(
        [[0.8, 0.8], [0.2, 0.8], [0.8, 0.2]], dtype=float
    )
    second_layer = {
        "A": np.eye(2),
        "b": np.array([-0.5, -0.5]),
        "abs_det_A": 1.0,
        "guard": quadrant_guard_spec(),
    }
    constraints = [
        {
            "constraint_id": 0,
            "block_id": 0,
            "normal": [1.0, 1.0],
            "threshold": 0.5,
        },
        {
            "constraint_id": 1,
            "block_id": 0,
            "normal": [1.0, 1.0],
            "threshold": -0.1,
        },
    ]
    trace, final = evaluate_two_branch_feedback_trace(
        intermediate_polygon=intermediate_polygon,
        intermediate_true_points=intermediate_true_points,
        old_intermediate_lower_bound=0.0,
        second_layer=second_layer,
        constraints=constraints,
    )
    assert len(trace) == 2
    assert trace[0]["then_useful"] and not trace[0]["else_useful"]
    assert trace[1]["else_useful"] and not trace[1]["then_triggered"]
    assert final["then_concrete_count"] == 1
    assert final["else_concrete_count"] == 2
    assert final["combined_corrected_lower_bound"] >= final[
        "then_corrected_lower_bound"
    ]
    assert final["combined_corrected_lower_bound"] >= final[
        "else_corrected_lower_bound"
    ]
    assert final["combined_corrected_lower_bound"] <= final["true_loss_area"] + 1e-9
    assert final["true_loss_area"] <= polygon_area(intermediate_polygon)
    print("smoke_branch_feedback: ok")


if __name__ == "__main__":
    main()
