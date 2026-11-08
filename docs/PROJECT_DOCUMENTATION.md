# Gender Income Gap Analysis

## 1. Project Overview

Gender Income Gap Analysis is a statistical machine-learning application that investigates whether gender is associated with the probability of earning more than $50,000 per year.

The project uses the UCI Adult Census Income dataset. It combines descriptive statistics, permutation tests, bootstrap confidence intervals, logistic regression, interaction analysis, and predictive machine learning in one reproducible workflow.

The project is designed to answer a practical question:

> After considering observable factors such as age, education, work hours, and occupation, how different are the income outcomes associated with gender?

The application does not claim to measure discrimination directly. It measures patterns and associations in the available data and presents those results with statistical uncertainty and methodological caveats.

## 2. Who This Project Is For

### Students and learners

The project demonstrates how several common data-science techniques work together in a complete analysis:

- Data loading and cleaning
- Feature preparation
- Statistical hypothesis testing
- Logistic regression
- Confidence intervals
- Interaction terms
- Cross-validation and AUC
- Automated reporting
- Offline dashboard generation

### Researchers and policy analysts

The workflow provides a starting point for examining gender-related income disparities while separating:

- The raw difference in income rates
- Differences associated with education and occupation composition
- The adjusted association remaining after model controls

### Data scientists

The repository is a compact example of a reproducible tabular-analysis pipeline. It includes a custom logistic-regression implementation, model comparison, generated tables, figures, JSON results, and a browser-readable dashboard.

### Educators and reviewers

The generated report and dashboard can be used to explain why an observed group difference should not automatically be interpreted as causal evidence.

## 3. What the Application Does

The application runs four analytical sections.

### 3.1 Raw income gap

The application calculates the proportion of men and women whose income category is above $50K. The raw gap is defined as:

```text
male rate of income >50K - female rate of income >50K
```

It also calculates a bootstrap 95% confidence interval for that difference.

### 3.2 Permutation tests

The first permutation test shuffles gender labels across the complete dataset. It tests whether the observed income gap is unusual under a null hypothesis in which gender labels are unrelated to income outcomes.

The second test shuffles gender labels within education-group and occupation strata. This asks whether the gap remains unusually large when people are compared within similar education and occupation groups.

The stratified null distribution is centered on its simulated mean because different gender compositions across strata can produce a non-zero expected gap even under the within-stratum null.

The p-value uses the plus-one correction:

```text
(number of null statistics at least as extreme as observed + 1)
/ (number of permutations + 1)
```

### 3.3 Adjusted odds ratios

The application fits nested logistic regression models for the binary outcome `income > 50K`.

The models are:

| Model | Variables |
|---|---|
| M0 | Gender only |
| M1 | Gender, age, age squared, and work hours |
| M2 | M1 plus education years |
| M3 | M2 plus occupation indicators |

For each model, the application reports the female-versus-male odds ratio, Wald confidence interval, and p-value.

An odds ratio below 1 means that the modeled odds of income above $50K are lower for women than for men, conditional on the variables included in that model. It does not represent a percentage difference in income and should not be read as a causal effect.

### 3.4 Gender and education interaction

The application tests whether the gender association changes across education levels.

It includes:

- A continuous `female x education` interaction
- A likelihood-ratio test for the continuous interaction
- A categorical interaction by education group
- Female-versus-male odds ratios for each education group

The education groups are:

- `<=HS`
- `SomeCollege`
- `Bachelors`
- `Postgrad`

### 3.5 Predictive machine-learning check

The optional machine-learning stage compares five-fold cross-validated ROC AUC with and without gender as a feature.

The models are:

- Logistic regression
- Histogram gradient boosting

This analysis asks whether gender adds predictive information after age, hours, education, and occupation are already available. It is a predictive comparison, not a causal fairness test.

## 4. Why the Project Is Valuable

### It distinguishes raw and adjusted differences

A raw gap may reflect differences in education, occupation, work hours, age, or other factors. Comparing nested models shows how the estimated gender association changes as observable covariates are added.

### It uses uncertainty instead of a single number

Permutation p-values, bootstrap confidence intervals, Wald confidence intervals, and likelihood-ratio tests show how strongly the data support each result and how precise the estimates are.

### It compares people within relevant groups

The stratified permutation test reduces the risk of interpreting group-composition differences as if they were differences among otherwise similar observations.

### It makes interaction effects visible

The gender association may not be identical at every education level. The interaction analysis makes that possibility testable rather than assuming one common gender coefficient is sufficient.

### It produces accessible results

The application automatically creates:

- A Markdown summary for reports
- CSV tables for further analysis
- PNG figures for presentations
- A self-contained offline HTML dashboard
- JSON output for downstream tools

### It is reproducible

The full workflow can be rerun from the command line with a selected input file and permutation count. The dashboard is generated from the saved JSON results, so the displayed analysis can be regenerated consistently.

## 5. System Architecture

The application is a command-line analytical pipeline with a static reporting layer.

```text
Input data
    |
    v
src/data.py
    Load, clean, encode, and derive analysis variables
    |
    v
run_all.py
    Orchestrate the complete analysis
    |
    +--> src/permutation.py
    |       Raw and stratified permutation tests
    |       Bootstrap confidence interval
    |
    +--> src/logit.py
    |       Design matrices
    |       NumPy IRLS logistic regression
    |       Odds ratios, Wald intervals, LR tests
    |
    +--> src/ml_model.py (optional)
    |       Five-fold cross-validation
    |       Logistic regression and gradient boosting AUC
    |
    v
results/
    Tables, figures, JSON results, Markdown summary
    |
    v
src/build_dashboard.py
    Insert results JSON into the dashboard template
    |
    v
results/dashboard.html
    Self-contained offline browser dashboard
```

## 6. Repository Structure

```text
README.md
requirements.txt
run_all.py

data/
    Adult Census input data

docs/
    REPORT_TEMPLATE.md
    PROJECT_DOCUMENTATION.md

results/
    dashboard.html
    results.json
    summary.md
    CSV analysis tables
    PNG figures

src/
    __init__.py
    data.py
    permutation.py
    logit.py
    ml_model.py
    build_dashboard.py
    dashboard_template.html
```

## 7. Module Responsibilities

### `run_all.py`

The main orchestration script. It:

1. Parses command-line arguments
2. Loads the dataset
3. Runs raw and stratified permutation tests
4. Calculates the bootstrap interval
5. Fits nested logistic models
6. Runs interaction analyses
7. Optionally runs the ML comparison
8. Writes tables, figures, JSON, Markdown, and the dashboard

### `src/data.py`

Responsible for input loading and analysis-ready data preparation.

It:

- Reads a local Adult dataset when available
- Falls back to OpenML when a local file is not found
- Supports synthetic data for smoke testing only
- Removes missing occupation, workclass, sex, and income records
- Converts the income label to a binary outcome
- Converts sex to a binary female indicator
- Creates education groups
- Merges rare occupations into `Other-rare`
- Centers and scales selected numeric variables

### `src/permutation.py`

Contains the statistical resampling functions:

- Raw permutation test
- Within-strata permutation test
- Bootstrap confidence interval for the raw gap

### `src/logit.py`

Contains the custom logistic-regression implementation.

The `Logit` class uses iteratively reweighted least squares, implemented with NumPy. The module also provides:

- Dummy-variable design-matrix construction
- Interaction-term construction
- Odds ratios
- Wald confidence intervals
- Coefficient p-values
- Likelihood-ratio tests

### `src/ml_model.py`

Contains the optional predictive modeling stage. It builds preprocessing pipelines with one-hot encoding for occupation and numeric preprocessing for continuous variables, then evaluates models through stratified five-fold cross-validation.

### `src/build_dashboard.py`

Reads `results.json`, safely inserts the results into the dashboard template, and writes the final self-contained HTML file.

### `src/dashboard_template.html`

The presentation layer for the offline dashboard. It contains the HTML, CSS, and JavaScript needed to display the generated analysis without a server or external runtime dependency.

## 8. Data Processing

The input data is expected to contain the UCI Adult fields, including:

- Age
- Work class
- Education and education number
- Occupation
- Sex
- Hours per week
- Income class

The analysis retains records with usable values for occupation, workclass, sex, and income. The target is created from the income class:

```text
income = 1 when class is >50K
income = 0 otherwise
```

The female indicator is created from the sex field:

```text
female = 1 for Female
female = 0 for Male
```

Age and work hours are scaled per 10 units for regression stability. Age is also represented quadratically to allow a non-linear relationship with income.

Occupation categories with fewer than 100 observations are combined into `Other-rare`.

## 9. Generated Results

The main output files are:

### `results/dashboard.html`

A static dashboard that can be opened directly in a browser. It contains the key findings, charts, model results, and plain-language explanations.

### `results/summary.md`

A report-ready summary containing sample counts, income rates, gaps, p-values, confidence intervals, model odds ratios, interaction tests, and optional AUC results.

### `results/results.json`

A structured representation of all calculated results. This is the data source for the dashboard.

### CSV tables

The generated CSV files are useful for spreadsheets, reports, or additional analysis:

- `table_odds_ratios_female.csv`
- `table_full_model_M3.csv`
- `table_female_OR_by_education.csv`
- `table_ml_auc.csv` when the ML stage is completed

### PNG figures

The generated figures show:

- Permutation null distributions
- Odds ratios across nested models
- Gender odds ratios by education group

## 10. Typical Use Cases

### Educational demonstration

Run the project with a reduced permutation count to demonstrate resampling, regression, interactions, and model evaluation in a classroom or notebook discussion.

### Exploratory inequality analysis

Use the dashboard to inspect whether a raw gender income gap remains after accounting for a selected set of observable characteristics.

### Reproducible report generation

Use the generated Markdown, tables, figures, and JSON to prepare a report or presentation without manually copying values from statistical output.

### Method comparison

Compare conclusions from:

- Descriptive group comparisons
- Permutation-based inference
- Regression-based adjustment
- Interaction analysis
- Predictive cross-validation

### Software engineering practice

Use the repository as an example of separating data preparation, statistical methods, machine learning, orchestration, and presentation code into focused modules.

### Future dataset extensions

The same pipeline can be adapted to a newer or organization-specific dataset if it contains compatible fields and clearly documented definitions. Any such adaptation should revalidate preprocessing, variable meanings, fairness assumptions, and statistical interpretation.

## 11. Running the Application

Install dependencies:

```bash
py -3.12 -m pip install -r requirements.txt
```

Run the default analysis:

```bash
py -3.12 run_all.py
```

Run a faster analysis with fewer permutations:

```bash
py -3.12 run_all.py --n-perm 2000
```

Run without the optional machine-learning stage:

```bash
py -3.12 run_all.py --n-perm 2000 --skip-ml
```

Use a specific input file:

```bash
py -3.12 run_all.py --data path/to/adult.data
```

Run a synthetic smoke test:

```bash
py -3.12 run_all.py --synthetic --n-perm 100
```

Synthetic output is for testing the code only and must never be presented as a real finding.

Open the generated dashboard:

```text
results/dashboard.html
```

The dashboard is static and works offline. No web server is required.

## 12. Interpretation Guide

### Rate difference

The raw rate difference is easy to understand but does not adjust for other variables.

### Odds ratio

An odds ratio compares odds, not probabilities. It should be interpreted with its confidence interval and the variables included in the model.

### P-value

A p-value describes how unusual the observed statistic is under a specified null hypothesis. It does not measure the size, importance, or cause of an effect.

### Confidence interval

A confidence interval describes estimation uncertainty under the assumptions of the method. It does not guarantee that the true value lies within the interval for a particular dataset.

### AUC

AUC measures ranking performance for prediction. A small AUC gain from adding gender means gender contributes little additional predictive ranking information in that model. It does not establish that gender is fair or unfair to use.

## 13. Limitations and Responsible Use

The results must be interpreted carefully.

- The Adult dataset is based on 1994 United States census data.
- The income outcome is binary and does not represent continuous wages.
- The analysis is observational and identifies association, not causation.
- Education and occupation may themselves be shaped by gender-related social conditions.
- A gap explained by observed variables is not automatically fair or acceptable.
- A remaining gap is not proof of discrimination.
- Survey weights such as `fnlwgt` are not used in the current implementation.
- Hours are self-reported.
- Wald confidence intervals rely on large-sample approximations.
- Unobserved variables may affect the estimates.
- The binary sex field in the source dataset does not represent the full range of gender identities.
- Results from this historical dataset should not automatically be generalized to modern populations or other countries.
- The application should not be used by itself for employment, compensation, lending, admissions, or other high-impact decisions.

## 14. Current Runtime Note

The optional histogram-gradient-boosting model requires dense input in some installed scikit-learn versions, while the preprocessing pipeline currently produces sparse one-hot encoded data. On such environments, the main statistical analysis can be run with `--skip-ml` while the data, permutation, regression, interaction, tables, figures, and dashboard stages continue to work.

This compatibility issue affects the optional predictive stage, not the core income-gap analysis.

## 15. Extension Ideas

Potential future improvements include:

- Add a configuration file for dataset paths and analysis parameters
- Support survey weights
- Add robust or clustered standard errors
- Add calibration and precision-recall metrics to the ML stage
- Convert sparse features to dense only for models that require it
- Add automated tests for edge cases and numerical stability
- Add support for more detailed gender categories when the input data supports them
- Add sensitivity analyses for occupation and education definitions
- Add a web-hosted version of the dashboard
- Add a reproducible environment lock file

## 16. Summary

Gender Income Gap Analysis is an end-to-end statistical machine-learning project for examining gender-associated differences in high-income outcomes. Its value comes from combining simple descriptive comparisons with resampling-based inference, adjusted logistic regression, interaction tests, predictive checks, and an accessible offline dashboard.

The project is best understood as an educational and exploratory analysis framework. It helps users ask better questions about income inequality, quantify uncertainty, compare alternative models, and communicate results without presenting correlation as proof of causation.
