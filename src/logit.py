"""Logistic regression via IRLS (NumPy only): odds ratios, Wald CIs, likelihood-ratio tests."""
import numpy as np
import pandas as pd
from scipy import stats


class Logit:
    def __init__(self, X: pd.DataFrame, y, max_iter=100, tol=1e-9):
        self.names = ["const"] + list(X.columns)
        A = np.column_stack([np.ones(len(X)), X.values.astype(float)])
        y = np.asarray(y, float)
        beta = np.zeros(A.shape[1])
        for _ in range(max_iter):
            eta = A @ beta
            mu = 1 / (1 + np.exp(-eta))
            w = np.clip(mu * (1 - mu), 1e-10, None)
            z = eta + (y - mu) / w
            AtW = A.T * w
            new = np.linalg.solve(AtW @ A, AtW @ z)
            done = np.max(np.abs(new - beta)) < tol
            beta = new
            if done:
                break
        mu = np.clip(1 / (1 + np.exp(-(A @ beta))), 1e-12, 1 - 1e-12)
        w = mu * (1 - mu)
        self.beta = pd.Series(beta, index=self.names)
        self.cov = pd.DataFrame(np.linalg.inv((A.T * w) @ A), index=self.names, columns=self.names)
        self.se = pd.Series(np.sqrt(np.diag(self.cov)), index=self.names)
        self.llf = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        self.k = A.shape[1]
        self.n = len(y)

    def pvalues(self):
        return pd.Series(2 * stats.norm.sf(np.abs(self.beta / self.se)), index=self.names)

    def odds_ratio(self, term, alpha=0.05):
        return self.odds_ratio_sum([term], alpha)

    def odds_ratio_sum(self, terms, alpha=0.05):
        """OR (Wald CI) for a sum of coefficients, e.g. main effect + interaction."""
        z = stats.norm.ppf(1 - alpha / 2)
        b = sum(self.beta[t] for t in terms)
        s = np.sqrt(sum(self.cov.loc[a, c] for a in terms for c in terms))
        return {"OR": float(np.exp(b)), "ci_low": float(np.exp(b - z * s)),
                "ci_high": float(np.exp(b + z * s)), "p": float(2 * stats.norm.sf(abs(b / s)))}


def design(df, numeric, categorical=(), interact=()):
    """Design matrix. interact = [(col_a, col_b)] multiplies two columns that are already in X
    (use numeric columns or dummy prefixes, e.g. ('female', 'edu_group') -> all dummies)."""
    X = df[list(numeric)].astype(float).copy()
    for c in categorical:
        X = X.join(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=False)
                   .drop(columns=lambda_ref(df, c)).astype(float))
    for a, b in interact:
        cols_a = [c for c in X.columns if c == a or c.startswith(a + "_")]
        cols_b = [c for c in X.columns if c == b or c.startswith(b + "_")]
        for ca in cols_a:
            for cb in cols_b:
                X[f"{ca}:{cb}"] = X[ca] * X[cb]
    return X


def lambda_ref(df, col):
    """Reference level (dropped dummy) = first level in sorted order, except edu_group (<=HS)."""
    if col == "edu_group":
        return [f"{col}_<=HS"]
    return [f"{col}_{sorted(df[col].astype(str).unique())[0]}"]


def lr_test(m_small: Logit, m_big: Logit):
    stat = 2 * (m_big.llf - m_small.llf)
    dof = m_big.k - m_small.k
    return {"LR_stat": float(stat), "df": int(dof), "p": float(stats.chi2.sf(stat, dof))}
