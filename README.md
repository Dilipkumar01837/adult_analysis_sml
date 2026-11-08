# Gender Income Gap Analysis Using Permutation Tests and Odds Ratios
SDG 5 - Gender Equality | Statistical Machine Learning project

## Problem
Pay gaps are usually quoted as raw averages that ignore education, hours and occupation, so policy
cannot tell how much of the gap is unexplained. This project answers three questions:
1. **Permutation p-value for the raw gap** (plus a stratified permutation test)
2. **Adjusted odds ratios with confidence intervals** (nested logistic regressions)
3. **Gender x education interaction** (likelihood-ratio test + odds ratio per education group)
A predictive check (CV AUC with and without gender) is included as the ML component.

## Data
UCI Adult / Census Income (outcome: income > 50K, binary). Two ways to supply it:
- Put the UCI file `adult.data` in `data/` (https://archive.ics.uci.edu/dataset/2/adult), or
- Do nothing: the code tries `fetch_openml("adult", version=2)` (needs internet).
`--synthetic` generates FAKE data to smoke-test the code only. Never report those numbers.

## Run
```
pip install -r requirements.txt
python run_all.py                    # real data, 10,000 permutations (~1-3 min)
python run_all.py --n-perm 2000      # faster
python run_all.py --data path/to/adult.data
```
Outputs in `results/`: **`dashboard.html`** (open in any browser, works offline; plain-language
explanations, charts and tables), `summary.md` (all key numbers, auto-written), `results.json`, `table_*.csv`,
`fig1_permutation_null.png`, `fig2_odds_ratios_models.png`, `fig3_or_by_education.png`.

## Structure
- `src/data.py` loading/cleaning (drops missing occupation/workclass, merges occupations with <100 rows)
- `src/permutation.py` permutation tests (plain and within-strata) and bootstrap CI
- `src/logit.py` NumPy logistic regression (IRLS), Wald CIs, LR tests; validated against
  scikit-learn (coefficients agree to ~1e-6)
- `src/ml_model.py` 5-fold CV AUC, logistic regression and gradient boosting, with/without gender
- `src/build_dashboard.py` + `src/dashboard_template.html` build the dashboard from `results.json`
  (rebuild alone with `python -m src.build_dashboard`)
- `run_all.py` runs everything and writes outputs
- `docs/REPORT_TEMPLATE.md` report skeleton

## Method notes
- Raw statistic: male minus female rate of >50K. P-value = (#{|null| >= |obs|} + 1) / (B + 1).
- Stratified permutation shuffles gender within education-group x occupation cells. Its null is
  centred on the gap expected from the cells' gender mix, so extremeness is measured around that centre.
- Models: M0 unadjusted, M1 + age (quadratic) and hours, M2 + education years, M3 + occupation.
  Age, hours scaled per 10 units; education centred.
- Interaction: `female x edu_c` (1 df LR test) and `female x education group` (3 df LR test).
  Group-specific ORs come from the sum of the main effect and interaction coefficients.
- `relationship` and `marital_status` are deliberately excluded: `relationship` encodes
  husband/wife, so it is gender-coded.

## Limitations (state these in the report)
- Adult data is from the 1994 US census; outcome is binary (>50K), not a continuous wage.
- Association, not causation. Education and occupation can themselves be shaped by gender
  (mediators), so "unexplained" does not equal discrimination, and "explained" does not equal fair.
- Survey weights (`fnlwgt`) are not used. Wald CIs assume large samples. Hours are self-reported.
