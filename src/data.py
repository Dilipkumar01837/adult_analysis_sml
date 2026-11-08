"""Data loading and cleaning (UCI Adult / Census Income dataset)."""
import numpy as np
import pandas as pd
from pathlib import Path

UCI_COLS = ["age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "class"]


def _read_raw(csv_path=None):
    """Read from a local CSV (UCI adult.data), else try OpenML (needs internet)."""
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path, header=None, names=UCI_COLS, skipinitialspace=True)
        if str(df.iloc[0, 0]).strip().lower() == "age":  # file had its own header
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = [c.replace("-", "_") for c in df.columns]
        return df
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    raw.columns = [c.replace("-", "_") for c in raw.columns]
    return raw


def make_synthetic(n=20000, seed=0):
    """Synthetic data for TESTING THE CODE ONLY. Never report these results."""
    rng = np.random.default_rng(seed)
    female = rng.random(n) < 0.33
    edu = rng.integers(5, 17, n)
    age = rng.integers(18, 70, n)
    hours = np.clip(rng.normal(40, 10, n) - 3 * female, 5, 90)
    occ = rng.choice(["Sales", "Craft-repair", "Exec-managerial", "Adm-clerical",
                      "Prof-specialty", "Other-service"], n)
    occ_eff = pd.Series(occ).map({"Exec-managerial": 1.0, "Prof-specialty": 0.8,
                                  "Sales": 0.2, "Craft-repair": 0.0,
                                  "Adm-clerical": -0.3, "Other-service": -0.8}).values
    logit = (-9 + 0.4 * edu + 0.04 * age - 0.0004 * (age - 40) ** 2
             + 0.03 * hours + occ_eff - 0.5 * female + 0.06 * female * (edu - 10))
    p = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "age": age, "workclass": "Private", "education_num": edu,
        "occupation": occ, "sex": np.where(female, "Female", "Male"),
        "hours_per_week": hours.round(),
        "class": np.where(rng.random(n) < p, ">50K", "<=50K"),
    })


def load_data(csv_path=None, synthetic=False, seed=0):
    df = make_synthetic(seed=seed) if synthetic else _read_raw(csv_path)
    df = df.copy()
    for c in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    df = df.dropna(subset=["occupation", "workclass", "sex", "class"]).reset_index(drop=True)

    out = pd.DataFrame()
    out["income"] = df["class"].astype(str).str.replace(".", "", regex=False).eq(">50K").astype(int)
    out["female"] = df["sex"].eq("Female").astype(int)
    out["age"] = df["age"].astype(float)
    out["hours"] = df["hours_per_week"].astype(float)
    out["edu_num"] = df["education_num"].astype(float)
    out["edu_group"] = pd.cut(out["edu_num"], [0, 9, 12, 13, 16],
                              labels=["<=HS", "SomeCollege", "Bachelors", "Postgrad"]).astype(str)
    occ = df["occupation"].astype(str)
    rare = occ.value_counts()[lambda s: s < 100].index  # merge tiny categories
    out["occ"] = occ.where(~occ.isin(rare), "Other-rare")
    out["age_c"] = (out["age"] - out["age"].mean()) / 10
    out["age_c2"] = out["age_c"] ** 2
    out["hours_c"] = (out["hours"] - out["hours"].mean()) / 10
    out["edu_c"] = out["edu_num"] - out["edu_num"].mean()
    return out
