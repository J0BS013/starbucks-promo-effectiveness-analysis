# Promotional Offer Decision Memo

## Decision

Prioritize discount offers for the next controlled test, retain BOGO only as a challenger with stricter economic guardrails, and do not treat informational offers as conversion-generating promotions.

This is a test recommendation rather than a full-scale rollout decision. The Starbucks transcript supports reliable temporal attribution and observational prioritization, but it does not identify incremental causal lift.

## Evidence

- Discount offers show the strongest observed completion association and a positive reward-to-revenue scenario in the analyzed population.
- BOGO offers generate completions but have a negative observed reward-to-revenue scenario under the documented proxy assumptions.
- Viewing is associated with higher completion, but viewing is a post-exposure behavior and cannot define a causal treatment group.
- The exposure-grain model prevents repeated sends from multiplying views, completions, or attributed rewards.

## Recommended experiment

Randomly assign eligible customers to control and discount-treatment groups before exposure. Use intention-to-treat as the primary analysis and pre-register:

- **Primary metric:** incremental contribution margin per eligible customer.
- **Secondary metrics:** offer completion, transaction conversion, average order value, and take-up.
- **Guardrails:** reward cost, contact frequency, unsubscribe behavior, and adverse segment concentration.
- **Decision rule:** scale only when the lower confidence bound for incremental contribution margin remains positive and guardrails stay within their predefined limits.

The synthetic experiment module in `src/causal.py` demonstrates balanced random assignment, treatment-effect estimation, and power planning without mixing simulated results with the observational Starbucks data.

## Risks and limitations

- There is no randomized holdout in the source dataset.
- Observed revenue is a prioritization proxy, not incremental revenue.
- Reward cost does not capture every operational or margin component.
- Segment findings may reflect customer composition and selection effects.
- Historical app behavior may not generalize to a future campaign or channel mix.

## Monitoring plan

Track assignment balance, exposure delivery, completion, incremental contribution margin, reward cost, and guardrails by offer type and pre-specified customer segment. Review results at the planned sample size instead of stopping when significance first appears. Roll back the treatment if contribution margin becomes negative or a guardrail breaches its agreed threshold.
