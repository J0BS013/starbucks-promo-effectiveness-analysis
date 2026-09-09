import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal import (
    evaluate_randomized_offer_experiment,
    plan_randomized_experiment,
    simulate_randomized_offer_experiment,
)


def test_synthetic_experiment_is_balanced_and_reproducible():
    first = simulate_randomized_offer_experiment(n_customers=2_000, seed=7)
    second = simulate_randomized_offer_experiment(n_customers=2_000, seed=7)
    assert first.equals(second)
    assert first["assignment"].value_counts().to_dict() == {"control": 1_000, "treatment": 1_000}


def test_randomized_fixture_recovers_a_positive_assigned_effect():
    experiment = simulate_randomized_offer_experiment(
        n_customers=20_000, baseline_conversion=0.20, treatment_effect=0.08, seed=7
    )
    result = evaluate_randomized_offer_experiment(experiment)
    assert result["uplift_absolute"] > 0.05
    assert result["significant"]


def test_power_plan_has_consistent_total_sample_size():
    plan = plan_randomized_experiment(0.20, 0.05)
    assert plan["required_per_group"] > 0
    assert plan["required_total"] == plan["required_per_group"] * 2


def test_synthetic_experiment_rejects_unbalanced_population_size():
    with pytest.raises(ValueError, match="even"):
        simulate_randomized_offer_experiment(n_customers=101)
