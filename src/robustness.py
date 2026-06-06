"""Robustness for the VIX vol-carry engine: crisis stress tests (does the crash
filter actually protect when it matters?) and block-bootstrap of the Sharpe."""
import numpy as np
import pandas as pd

from .metrics import performance, vol_target

TD = 252
# The episodes that matter for a short-volatility strategy
CRISES = {
    "Volmageddon Feb-2018": ("2018-01-26", "2018-02-12"),
    "COVID crash 2020":     ("2020-02-19", "2020-03-23"),
    "2022 selloff":         ("2022-01-01", "2022-06-30"),
    "Aug-2024 vol spike":   ("2024-07-25", "2024-08-09"),
}


def crisis_table(ret_naive: pd.Series, ret_carry: pd.Series) -> pd.DataFrame:
    """Cumulative return per crisis window, vol-targeted to 10% for comparability."""
    n, c = vol_target(ret_naive), vol_target(ret_carry)
    rows = {}
    for name, (a, b) in CRISES.items():
        wn, wc = n.loc[a:b], c.loc[a:b]
        if len(wc) < 2:
            continue
        rows[name] = {"Naive short-vol": np.exp(wn.sum()) - 1,
                      "Gated (crash filter)": np.exp(wc.sum()) - 1}
    return pd.DataFrame(rows).T


def block_bootstrap(ret: pd.Series, n=2000, block=21, seed=0):
    r = vol_target(ret).dropna().values
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(r) / block))
    sh, dd = np.empty(n), np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(r) - block, nb)
        samp = np.concatenate([r[s:s + block] for s in starts])[:len(r)]
        sh[i] = samp.mean() / samp.std() * np.sqrt(TD)
        eq = np.cumprod(1 + samp); dd[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return sh, dd
