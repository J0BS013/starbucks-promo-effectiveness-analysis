"""Synthetic randomized-experiment utilities kept separate from transcript analysis."""

import numpy as np
import pandas as pd

from utils.stats import ab_test_proportions, required_sample_size


def simulate_randomized_offer_experiment(
    n_customers: int = 10_000,
    baseline_conversion: float = 0.20,
    treatment_effect: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced RCT fixture with known randomized assignment.

    ``treatment_effect`` is an absolute conversion-rate difference. This data
    is intentionally synthetic and must not be combined with Starbucks
    transcript results, which do not include random assignment.
    """
    if n_customers < 2 or n_customers % 2:
        raise ValueError("n_customers must be an even number of at least 2")
    if not 0 <= baseline_conversion <= 1:
        raise ValueError("baseline_conversion must be between 0 and 1")
    if not 0 <= baseline_conversion + treatment_effect <= 1:
        raise ValueError("treatment conversion must be between 0 and 1")

    rng = np.random.default_rng(seed)
    assignment = np.repeat(["control", "treatment"], n_customers // 2)
    rng.shuffle(assignment)
    probabilities = np.where(
        assignment == "treatment", baseline_conversion + treatment_effect, baseline_conversion
    )
    return pd.DataFrame(
        {
            "customer_id": [f"synthetic_{i}" for i in range(n_customers)],
            "assignment": assignment,
            "converted": rng.binomial(1, probabilities).astype(bool),
        }
    )


def evaluate_randomized_offer_experiment(experiment: pd.DataFrame) -> dict:
    """Estimate the assigned-offer effect for a synthetic randomized fixture."""
    required = {"assignment", "converted"}
    missing = required - set(experiment.columns)
    if missing:
        raise ValueError(f"Experiment is missing columns: {sorted(missing)}")
    control = experiment.loc[experiment["assignment"] == "control", "converted"]
    treatment = experiment.loc[experiment["assignment"] == "treatment", "converted"]
    if control.empty or treatment.empty:
        raise ValueError("Experiment must contain control and treatment records")
    return ab_test_proportions(control, treatment)


def plan_randomized_experiment(
    baseline_conversion: float, minimum_detectable_effect: float
) -> dict:
    """Return a transparent two-group power-analysis plan."""
    per_group = required_sample_size(baseline_conversion, minimum_detectable_effect)
    return {
        "baseline_conversion": baseline_conversion,
        "minimum_detectable_effect": minimum_detectable_effect,
        "alpha": 0.05,
        "power": 0.80,
        "required_per_group": per_group,
        "required_total": per_group * 2,
    }
