import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import zt_ind_solve_power

log = logging.getLogger(__name__)


def ab_test_proportions(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
    alternative: str = "larger",
) -> dict:
    """Z-test for two proportions (e.g. conversion rates).

    Used to test whether the treatment group has a higher conversion rate
    than the control group.
    """
    n_ctrl = len(control)
    n_trt = len(treatment)
    conv_ctrl = int(control.sum())
    conv_trt = int(treatment.sum())

    stat, p_value = proportions_ztest(
        count=[conv_trt, conv_ctrl],
        nobs=[n_trt, n_ctrl],
        alternative=alternative,
    )
    rate_ctrl = conv_ctrl / n_ctrl
    rate_trt = conv_trt / n_trt
    uplift_abs = rate_trt - rate_ctrl
    uplift_rel = uplift_abs / rate_ctrl if rate_ctrl > 0 else np.nan

    ci_low = uplift_abs - 1.96 * np.sqrt(
        rate_ctrl * (1 - rate_ctrl) / n_ctrl + rate_trt * (1 - rate_trt) / n_trt
    )
    ci_high = uplift_abs + 1.96 * np.sqrt(
        rate_ctrl * (1 - rate_ctrl) / n_ctrl + rate_trt * (1 - rate_trt) / n_trt
    )

    return {
        "n_control": n_ctrl,
        "n_treatment": n_trt,
        "conversion_control": round(rate_ctrl, 4),
        "conversion_treatment": round(rate_trt, 4),
        "uplift_absolute": round(uplift_abs, 4),
        "uplift_relative_pct": round(uplift_rel * 100, 2) if not np.isnan(uplift_rel) else np.nan,
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "z_stat": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }


def ab_test_means(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Welch's t-test for continuous metrics (e.g. average spend).

    Welch's variant does not assume equal variances between groups,
    making it more robust for real-world A/B tests.
    """
    ctrl_clean = control.dropna()
    trt_clean = treatment.dropna()
    stat, p_value = stats.ttest_ind(trt_clean, ctrl_clean, equal_var=False)

    mean_ctrl = ctrl_clean.mean()
    mean_trt = trt_clean.mean()
    uplift_abs = mean_trt - mean_ctrl
    uplift_rel = uplift_abs / mean_ctrl if mean_ctrl > 0 else np.nan

    se = np.sqrt(ctrl_clean.var() / len(ctrl_clean) + trt_clean.var() / len(trt_clean))
    ci_low = uplift_abs - 1.96 * se
    ci_high = uplift_abs + 1.96 * se

    return {
        "n_control": len(ctrl_clean),
        "n_treatment": len(trt_clean),
        "mean_control": round(mean_ctrl, 2),
        "mean_treatment": round(mean_trt, 2),
        "uplift_absolute": round(uplift_abs, 2),
        "uplift_relative_pct": round(uplift_rel * 100, 2) if not np.isnan(uplift_rel) else np.nan,
        "ci_95_low": round(ci_low, 2),
        "ci_95_high": round(ci_high, 2),
        "t_stat": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }


def chi_squared_test(
    contingency_table: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Chi-squared test of independence between two categorical variables.

    Tests whether the observed distribution differs from what we'd expect
    if the two variables were independent.
    """
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {
        "chi2_stat": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum sample size per group to detect a given MDE with specified power.

    Args:
        baseline_rate: Current conversion rate (proportion).
        mde: Minimum detectable effect in absolute percentage points.
        alpha: Type I error rate (false positive).
        power: 1 - Type II error rate (1 - false negative).
    """
    effect_size = (
        2 * np.arcsin(np.sqrt(baseline_rate + mde))
        - 2 * np.arcsin(np.sqrt(baseline_rate))
    )
    n = zt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative="larger",
    )
    return int(np.ceil(n))


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni correction for multiple hypothesis testing.

    Adjusts the significance threshold when running multiple tests
    to control the family-wise error rate (FWER).
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    return [
        {
            "p_value": p,
            "adjusted_alpha": round(adjusted_alpha, 6),
            "significant_corrected": p < adjusted_alpha,
        }
        for p in p_values
    ]


def calculate_roi(
    incremental_revenue: float,
    campaign_cost: float,
) -> dict:
    """Return on Investment for a marketing campaign."""
    net_profit = incremental_revenue - campaign_cost
    roi = net_profit / campaign_cost if campaign_cost > 0 else np.nan
    return {
        "incremental_revenue": round(incremental_revenue, 2),
        "campaign_cost": round(campaign_cost, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi * 100, 2) if not np.isnan(roi) else np.nan,
    }
