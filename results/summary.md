# Results summary (n = 46,033; 14,919 women, 31,114 men)

## 1. Permutation test (raw gap)
- P(>50K): men 0.313, women 0.114; gap = 0.199 (bootstrap 95% CI 0.192 to 0.206)
- Permutation p-value (2000 shuffles): < 0.0005
- Stratified permutation (within education x occupation) p-value: < 0.0005 (gap expected from education/occupation mix alone: 0.046)

## 2. Female vs male odds ratios (95% CI)

| Model | OR | 95% CI | p |
|---|---|---|---|
| M0 unadjusted | 0.282 | 0.266 to 0.298 | 0 |
| M1 + age, hours | 0.341 | 0.321 to 0.361 | 8.1e-279 |
| M2 + education | 0.304 | 0.286 to 0.323 | 7.5e-307 |
| M3 + occupation | 0.295 | 0.276 to 0.315 | 8.3e-283 |
- Reduction in the log-odds gap after adding all covariates: 3.6% (negative = gap widens). What remains is unexplained by these variables, which is not proof of discrimination.

## 3. Gender x education interaction
- Continuous: ratio of ORs per extra year of education = 1.028 (95% CI 0.998 to 1.058); LR test chi2 = 3.39, df = 1, p = 0.0655
- By education group (3 df LR test p = 0.263):

| Education | Female OR | 95% CI | n |
|---|---|---|---|
| <=HS | 0.305 | 0.268 to 0.346 | 20742 |
| SomeCollege | 0.270 | 0.240 to 0.304 | 13543 |
| Bachelors | 0.317 | 0.281 to 0.358 | 7772 |
| Postgrad | 0.288 | 0.247 to 0.337 | 3976 |