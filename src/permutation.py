"""Permutation tests and bootstrap CI for the raw gender gap in P(income > 50K)."""
import numpy as np


def _gap(y, f):
    """Male rate minus female rate."""
    return y[f == 0].mean() - y[f == 1].mean()


def permutation_test(y, female, n_perm=10000, seed=42, strata=None):
    """Two-sided permutation test of H0: gender is unrelated to income.

    strata=None   -> shuffle gender labels over the whole sample (raw gap).
    strata=array  -> shuffle gender only *within* strata (e.g. education x occupation),
                     testing whether the gap survives when comparing like with like.
    For strata, the 'expected gap from composition' is the null mean.
    p-value uses the +1 correction: (#{|null| >= |obs|} + 1) / (n_perm + 1).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    f = np.asarray(female, int)
    obs = _gap(y, f)
    null = np.empty(n_perm)
    if strata is None:
        for i in range(n_perm):
            null[i] = _gap(y, rng.permutation(f))
    else:
        strata = np.asarray(strata)
        groups = [np.where(strata == s)[0] for s in np.unique(strata)]
        for i in range(n_perm):
            fp = f.copy()
            for idx in groups:
                fp[idx] = rng.permutation(f[idx])
            null[i] = _gap(y, fp)
    # Stratified null is NOT centred at 0 (men/women are spread differently across strata),
    # so measure extremeness around the null mean. For the plain test the mean is ~0.
    centre = null.mean() if strata is not None else 0.0
    p = (np.sum(np.abs(null - centre) >= abs(obs - centre)) + 1) / (n_perm + 1)
    return {"observed_gap": float(obs), "p_value": float(p), "n_perm": n_perm,
            "null": null, "null_sd": float(null.std()), "null_mean": float(null.mean())}


def bootstrap_gap_ci(y, female, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    f = np.asarray(female, int)
    ym, yf = y[f == 0], y[f == 1]
    gaps = np.array([rng.choice(ym, len(ym)).mean() - rng.choice(yf, len(yf)).mean()
                     for _ in range(n_boot)])
    return tuple(float(q) for q in np.quantile(gaps, [alpha / 2, 1 - alpha / 2]))
