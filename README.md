# Promo Effectiveness Analysis — Starbucks Capstone

**Stack:** Python · pandas · scipy · statsmodels · matplotlib/seaborn · Jupyter

---

## Business Question

> *Which promotional offer type drives the highest incremental conversion and revenue, in which customer segments, and at what ROI?*

---

## Executive Summary

| Offer Type | Uplift (vs. control) | Best Segment | ROI |
|---|---|---|---|
| **Discount** | **+84.6%** | Low-income / Age 35–44 | **+89%** |
| BOGO | +33.9% | Age 35–44 | −59% |
| Informational | n/s | — | n/a |

**Key findings:**

1. **Discount is the best campaign** — statistically significant uplift of +84.6% over organic baseline and the only offer type with positive ROI (+89%). For every $1 spent on rewards, it returned $1.89 in incremental revenue.
2. **BOGO drives conversion but destroys value** — uplift of +33.9% is real (p < 0.05) but the reward cost ($234K) exceeds incremental revenue ($96K), resulting in −59% ROI.
3. **Informational offers have zero measurable effect** — no statistically significant conversion uplift, zero reward cost, no ROI case. Budget should be reallocated.
4. **Viewing the offer is the key driver** — customers who viewed any offer converted at 62.2% vs. 35.4% organic (+75.8% uplift, p ≈ 0). Ensuring visibility is more important than offer type selection.
5. **Exposed customers spend more** — average total spend of $107.64 vs. $95.47 for non-exposed (+12.8%, p ≈ 0), confirming offers lift basket size beyond just completion.

**Recommendation:** Concentrate Discount offers on low-income customers aged 25–44, where uplift exceeds +140%. Pause BOGO pending a reward structure review — the mechanic works behaviorally but is currently unprofitable. Eliminate Informational from the budget.

---

## Dataset

**Starbucks Capstone Challenge** — simulated data mirroring real Starbucks customer behavior.

| File | Rows | Description |
|---|---|---|
| `portfolio.json` | 10 | Offer metadata: type (BOGO/discount/informational), spend requirement, reward, duration, distribution channels |
| `profile.json` | 17,000 | Customer demographics: age, gender, income, membership start date |
| `transcript.json` | ~306,000 | Event log: offer received, offer viewed, offer completed, transaction (amount) |

After cleaning (removing 2,175 customers with missing demographics):
- **14,825 valid customers**, **16,994 with at least one offer event**
- **115,609 offer-level rows** across 10 distinct offers
- **Average transaction value: $12.78**

**Download:** [Kaggle — Starbucks Capstone Dataset](https://www.kaggle.com/datasets/blacktile/starbucks-app-customer-reward-program-data)  
Place the 3 JSON files in `data/raw/`.

---

## What Are We Doing?

### The Core Problem

Starbucks sends promotional offers to app users. Not every customer sees every offer, and not every offer type works equally well for every customer. We want to measure:

1. Does viewing a promotion **cause** customers to convert more?
2. Which offer type is **most effective**?
3. Which customer segments **respond best**?
4. Is the campaign **profitable** after accounting for rewards paid out?

### Pipeline

```
data/raw/*.json
     │
     ▼ python main.py
  Extract + Transform + Save
  (src/extract.py → src/transform.py → data/processed/*.parquet)
     │
     ▼ jupyter lab
  Notebooks 01 → 05
```

---

## Statistical Methods Explained

### A/B Testing vs. Hypothesis Testing

These two concepts are **related but distinct**:

| | A/B Testing | Hypothesis Testing |
|---|---|---|
| **What it is** | An *experimental design* | A *statistical inference framework* |
| **Question answered** | What did we observe? | Can we trust what we observed? |
| **Output** | Conversion rates, lift, funnels | p-values, test statistics, decisions |
| **Notebook** | 02 | 03 |

**A/B Testing** is the experiment — you design two groups, collect data, and compare results.  
**Hypothesis Testing** is the validation — you formally determine whether the observed difference is real or due to random chance.

*A/B Testing tells you what happened. Hypothesis Testing tells you whether to believe it.*

---

### Control and Treatment Groups

There is no pure holdout group in this dataset. We use the closest proxy:

| Group | Definition | Rationale |
|---|---|---|
| **Control (A)** | Received offer but **did not view** it | Behaves as if no offer exists — organic baseline |
| **Treatment (B)** | **Viewed** the offer | Aware of and influenced by the promotion |

---

### Statistical Tests Used

#### Z-Test for Proportions
Used to compare two conversion rates.

- **When to use:** Binary metric (converted/not), large samples (n > 30 per group)
- **H₀:** The two conversion rates are equal
- **p-value:** Probability of observing this difference if H₀ were true. If p < 0.05, reject H₀.

#### Welch's t-Test
Used to compare average spend between groups.

- **When to use:** Continuous metric, groups may have unequal variances
- **Why Welch's:** Does not assume equal variance — more robust than Student's t-test
- **H₀:** Mean spend is equal in both groups

#### Chi-Squared Test of Independence
Used to test whether offer completion is associated with gender.

- **When to use:** Two categorical variables
- **H₀:** The variables are independent (no association)

#### Bonferroni Correction
Running 4 tests at α = 0.05 gives ~18.5% chance of at least one false positive. Bonferroni divides the threshold by the number of tests: adjusted α = 0.05 / 4 = **0.0125**.

---

### Uplift

$$\text{Uplift}_{\text{relative}} = \frac{\text{Rate}_{\text{treatment}} - \text{Rate}_{\text{control}}}{\text{Rate}_{\text{control}}} \times 100\%$$

---

### ROI

$$\text{ROI} = \frac{\text{Incremental Revenue} - \text{Campaign Cost}}{\text{Campaign Cost}} \times 100\%$$

- **Incremental Revenue** = uplift × treated customers × avg ticket ($12.78)
- **Campaign Cost** = total rewards paid to completers
- Only statistically significant uplift is used — non-significant results produce $0 incremental revenue

---

## Results

### Hypothesis Tests

| # | Hypothesis | Test | Result | p-value | Decision |
|---|---|---|---|---|---|
| H1 | Viewing offer → higher conversion (35.4% → 62.2%) | Z-test | +75.8% uplift | ≈ 0 | **Reject H₀** |
| H2 | BOGO > Discount conversion | Z-test | Discount wins (79.7% vs 67.3%) | 1.0 | Fail to Reject H₀ |
| H3 | Exposed customers spend more ($95 → $108) | Welch's t | +$12.17 / +12.8% | ≈ 0 | **Reject H₀** |
| H4 | Completion depends on gender | Chi-squared | χ²=1712.8 | ≈ 0 | **Reject H₀** |

All 4 remain significant after Bonferroni correction (adjusted α = 0.0125). H2 is not significant regardless.

### Uplift by Segment (Top 5 — Discount offers only)

| Segment | Control Conv. | Treatment Conv. | Uplift |
|---|---|---|---|
| Low income | 29.0% | 78.0% | **+169.4%** |
| Age 35–44 | 34.7% | 85.5% | **+146.2%** |
| Age 25–34 | 33.5% | 81.6% | **+143.8%** |
| Age < 25 | 34.1% | 79.2% | **+132.4%** |
| Mid income | 36.0% | 81.5% | **+126.5%** |

### ROI by Offer Type

| Offer Type | Incremental Revenue | Campaign Cost | Net Profit | ROI |
|---|---|---|---|---|
| Discount | $185,890 | $98,220 | **+$87,670** | **+89%** |
| BOGO | $96,285 | $234,765 | **−$138,480** | **−59%** |
| Informational | $0 | $0 | $0 | n/a |

---

## Notebooks

| Notebook | Focus | Key Output |
|---|---|---|
| `01_eda.ipynb` | Data structure, quality, distributions | Funnel rates, demographic charts |
| `02_ab_testing.ipynb` | Experiment design, group comparison, observed lift | Conversion by group and offer type |
| `03_hypothesis_testing.ipynb` | Z-test, t-test, χ², Bonferroni correction | p-values, significance decisions |
| `04_uplift_segmentation.ipynb` | Uplift by gender, age, income, heatmaps | Top-responding segments ranking |
| `05_roi_analysis.ipynb` | ROI by offer type, channel, segment | Business profitability per campaign |

---

## Project Structure

```
promo-effectiveness-analysis/
├── data/
│   ├── raw/                    ← Starbucks JSON files (not versioned)
│   └── processed/              ← Parquet outputs from main.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ab_testing.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_uplift_segmentation.ipynb
│   └── 05_roi_analysis.ipynb
├── src/
│   ├── constants.py            ← Paths, alpha, bins
│   ├── extract.py              ← Load raw JSON files
│   ├── transform.py            ← Clean, merge, feature engineering
│   └── utils/
│       ├── stats.py            ← Z-test, t-test, chi-squared, ROI, Bonferroni
│       └── plot.py             ← Reusable chart helpers
├── tests/
│   └── test_transform.py       ← 12 unit tests for transform logic
├── reports/
│   └── figures/                ← Charts exported from notebooks
├── main.py                     ← Run ETL once before opening notebooks
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python main.py          # ETL: raw JSON → data/processed/*.parquet
python -m jupyter lab   # open notebooks 01 → 05 in order
```

To run tests:
```bash
python -m pytest tests/ -v
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy, pyarrow |
| Statistical tests | scipy, statsmodels |
| Visualization | matplotlib, seaborn |
| Notebooks | JupyterLab |
| Testing | pytest |
