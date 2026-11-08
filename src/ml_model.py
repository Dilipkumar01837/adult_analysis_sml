"""Predictive check: does knowing gender add predictive power once other features are known?"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def cv_auc(df, use_female, seed=42):
    num = ["age", "hours", "edu_num"] + (["female"] if use_female else [])
    rows = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for name, scale in [("LogisticRegression", True), ("HistGradientBoosting", False)]:
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["occ"]),
                                 ("num", StandardScaler() if scale else "passthrough", num)])
        clf = (LogisticRegression(max_iter=1000) if scale
               else HistGradientBoostingClassifier(random_state=seed))
        s = cross_val_score(make_pipeline(pre, clf), df[num + ["occ"]], df["income"],
                            cv=cv, scoring="roc_auc")
        rows[name] = (s.mean(), s.std())
    return rows


def compare(df):
    a, b = cv_auc(df, False), cv_auc(df, True)
    return pd.DataFrame([{"model": m, "AUC_without_gender": a[m][0],
                          "AUC_with_gender": b[m][0], "AUC_gain": b[m][0] - a[m][0],
                          "AUC_sd_with": b[m][1]} for m in a])
