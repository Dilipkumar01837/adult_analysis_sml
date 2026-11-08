"""Run the full analysis:  python run_all.py [--data data/adult.data] [--synthetic] [--n-perm 10000]"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_data
from src.permutation import permutation_test, bootstrap_gap_ci
from src.logit import Logit, design, lr_test
from src.ml_model import compare
from src.build_dashboard import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/adult.data")
    ap.add_argument("--synthetic", action="store_true", help="TEST RUN ONLY (fake data)")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-ml", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    df = load_data(a.data, synthetic=a.synthetic)
    R = {"synthetic_data_warning": bool(a.synthetic), "n": int(len(df)),
         "n_female": int(df.female.sum()), "n_male": int((1 - df.female).sum())}

    # ---- 1. Permutation test on the raw gap -------------------------------------------
    rate_m = df.loc[df.female == 0, "income"].mean()
    rate_f = df.loc[df.female == 1, "income"].mean()
    perm = permutation_test(df.income, df.female, a.n_perm)
    ci = bootstrap_gap_ci(df.income, df.female)
    strata = df.edu_group + "|" + df.occ
    perm_s = permutation_test(df.income, df.female, a.n_perm, seed=7, strata=strata)
    R["raw"] = {"rate_male": rate_m, "rate_female": rate_f,
                "gap": perm["observed_gap"], "gap_boot_ci95": ci,
                "perm_p": perm["p_value"], "n_perm": a.n_perm}
    R["stratified_perm"] = {"p": perm_s["p_value"], "null_sd": perm_s["null_sd"], "expected_gap_from_composition": perm_s["null_mean"],
                            "note": "gender shuffled within education-group x occupation strata"}

    # ---- 2. Adjusted odds ratios (nested models) ---------------------------------------
    specs = {
        "M0 unadjusted": (["female"], []),
        "M1 + age, hours": (["female", "age_c", "age_c2", "hours_c"], []),
        "M2 + education": (["female", "age_c", "age_c2", "hours_c", "edu_c"], []),
        "M3 + occupation": (["female", "age_c", "age_c2", "hours_c", "edu_c"], ["occ"]),
    }
    rows, models = [], {}
    for name, (num, cat) in specs.items():
        m = Logit(design(df, num, cat), df.income)
        models[name] = m
        rows.append({"model": name, **m.odds_ratio("female"), "n_params": m.k})
    or_tab = pd.DataFrame(rows)
    b0, b3 = np.log(or_tab.OR.iloc[0]), np.log(or_tab.OR.iloc[-1])
    R["odds_ratios_female_vs_male"] = or_tab.to_dict("records")
    R["share_of_log_odds_gap_explained_by_covariates"] = float(1 - b3 / b0)
    or_tab.to_csv(out / "table_odds_ratios_female.csv", index=False)

    m3 = models["M3 + occupation"]
    full = pd.DataFrame([{"term": t, **m3.odds_ratio(t)} for t in m3.names
                         if t != "const" and not t.startswith("occ_")])
    full.to_csv(out / "table_full_model_M3.csv", index=False)

    # ---- 3. Gender x education interaction ---------------------------------------------
    base_num = ["female", "age_c", "age_c2", "hours_c"]
    m_main = Logit(design(df, base_num + ["edu_c"], ["occ"]), df.income)
    m_int = Logit(design(df, base_num + ["edu_c"], ["occ"], [("female", "edu_c")]), df.income)
    R["interaction_continuous"] = {
        "term": "female x edu_c (per extra year of education)",
        **m_int.odds_ratio("female:edu_c"), "LR_test": lr_test(m_main, m_int)}

    g_main = Logit(design(df, base_num, ["occ", "edu_group"]), df.income)
    g_int = Logit(design(df, base_num, ["occ", "edu_group"], [("female", "edu_group")]), df.income)
    R["interaction_by_group_LR"] = lr_test(g_main, g_int)
    grp = [{"edu_group": "<=HS", **g_int.odds_ratio("female")}]
    for lvl in ["SomeCollege", "Bachelors", "Postgrad"]:
        grp.append({"edu_group": lvl,
                    **g_int.odds_ratio_sum(["female", f"female:edu_group_{lvl}"])})
    grp = pd.DataFrame(grp)
    grp["n"] = [int((df.edu_group == g).sum()) for g in grp.edu_group]
    grp.to_csv(out / "table_female_OR_by_education.csv", index=False)
    R["female_OR_by_education"] = grp.to_dict("records")

    R["interaction_continuous"]["female_OR_at_mean_education"] = m_int.odds_ratio("female")

    def _hist(res):
        c, e = np.histogram(res["null"], bins=50)
        return {"counts": c.tolist(), "edges": e.tolist(), "observed": res["observed_gap"]}
    R["perm_hist"] = {"raw": _hist(perm), "strat": _hist(perm_s)}

    # ---- 4. Predictive check -----------------------------------------------------------
    if not a.skip_ml:
        ml = compare(df)
        ml.to_csv(out / "table_ml_auc.csv", index=False)
        R["ml_auc"] = ml.to_dict("records")

    # ---- Figures -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for x, res, t in [(ax[0], perm, "Raw gap: labels shuffled overall"),
                      (ax[1], perm_s, "Shuffled within education x occupation")]:
        x.hist(res["null"], bins=50, color="#9ab")
        x.axvline(res["observed_gap"], color="r", lw=2, label=f"observed = {res['observed_gap']:.3f}")
        x.set_title(t, fontsize=10); x.set_xlabel("Male - female rate of >50K"); x.legend()
    fig.suptitle(f"Permutation null distributions ({a.n_perm} shuffles)")
    fig.tight_layout(); fig.savefig(out / "fig1_permutation_null.png", dpi=150); plt.close(fig)

    fig, x = plt.subplots(figsize=(7, 3.5))
    y = np.arange(len(or_tab))[::-1]
    x.errorbar(or_tab.OR, y, xerr=[or_tab.OR - or_tab.ci_low, or_tab.ci_high - or_tab.OR],
               fmt="o", color="k", capsize=3)
    x.axvline(1, ls="--", color="gray"); x.set_yticks(y); x.set_yticklabels(or_tab.model)
    x.set_xlabel("Odds ratio, female vs male (95% CI)"); x.set_title("Gender odds ratio as controls are added")
    fig.tight_layout(); fig.savefig(out / "fig2_odds_ratios_models.png", dpi=150); plt.close(fig)

    fig, x = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(grp))[::-1]
    x.errorbar(grp.OR, y, xerr=[grp.OR - grp.ci_low, grp.ci_high - grp.OR], fmt="s", color="#c44", capsize=3)
    x.axvline(1, ls="--", color="gray"); x.set_yticks(y); x.set_yticklabels(grp.edu_group)
    x.set_xlabel("Adjusted odds ratio, female vs male (95% CI)")
    x.set_title("Gender x education interaction")
    fig.tight_layout(); fig.savefig(out / "fig3_or_by_education.png", dpi=150); plt.close(fig)

    # ---- Save results + auto summary ---------------------------------------------------
    (out / "results.json").write_text(json.dumps(R, indent=2, default=float))
    r, ic = R["raw"], R["interaction_continuous"]
    p_txt = lambda p: f"< {1/(a.n_perm+1):.4f}" if p <= 1.5 / (a.n_perm + 1) else f"{p:.4f}"
    lines = []
    if a.synthetic:
        lines += ["**WARNING: SYNTHETIC DATA - test run only. Do not report these numbers.**", ""]
    lines += [
        f"# Results summary (n = {R['n']:,}; {R['n_female']:,} women, {R['n_male']:,} men)", "",
        "## 1. Permutation test (raw gap)",
        f"- P(>50K): men {r['rate_male']:.3f}, women {r['rate_female']:.3f}; gap = {r['gap']:.3f} "
        f"(bootstrap 95% CI {ci[0]:.3f} to {ci[1]:.3f})",
        f"- Permutation p-value ({a.n_perm} shuffles): {p_txt(r['perm_p'])}",
        f"- Stratified permutation (within education x occupation) p-value: {p_txt(perm_s['p_value'])} "
        f"(gap expected from education/occupation mix alone: {perm_s['null_mean']:.3f})", "",
        "## 2. Female vs male odds ratios (95% CI)", "",
        "| Model | OR | 95% CI | p |", "|---|---|---|---|"]
    for _, q in or_tab.iterrows():
        lines.append(f"| {q.model} | {q.OR:.3f} | {q.ci_low:.3f} to {q.ci_high:.3f} | {q.p:.2g} |")
    lines += [f"- Reduction in the log-odds gap after adding all covariates: "
              f"{R['share_of_log_odds_gap_explained_by_covariates']:.1%} (negative = gap widens). "
              f"What remains is unexplained by these variables, which is not proof of discrimination.", "",
              "## 3. Gender x education interaction",
              f"- Continuous: ratio of ORs per extra year of education = {ic['OR']:.3f} "
              f"(95% CI {ic['ci_low']:.3f} to {ic['ci_high']:.3f}); LR test chi2 = {ic['LR_test']['LR_stat']:.2f}, "
              f"df = 1, p = {ic['LR_test']['p']:.3g}",
              f"- By education group (3 df LR test p = {R['interaction_by_group_LR']['p']:.3g}):", "",
              "| Education | Female OR | 95% CI | n |", "|---|---|---|---|"]
    for _, q in grp.iterrows():
        lines.append(f"| {q.edu_group} | {q.OR:.3f} | {q.ci_low:.3f} to {q.ci_high:.3f} | {q.n} |")
    if "ml_auc" in R:
        lines += ["", "## 4. Predictive check (5-fold CV AUC)", "",
                  "| Model | Without gender | With gender | Gain |", "|---|---|---|---|"]
        for q in R["ml_auc"]:
            lines.append(f"| {q['model']} | {q['AUC_without_gender']:.4f} | "
                         f"{q['AUC_with_gender']:.4f} | {q['AUC_gain']:+.4f} |")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nDashboard:", build(out))


if __name__ == "__main__":
    main()
