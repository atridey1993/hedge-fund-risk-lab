from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_test(exceptions: pd.Series, alpha: float = 0.99) -> dict:
    x = int(exceptions.sum())
    n = int(exceptions.count())
    p = 1 - alpha

    if x == 0 or x == n:
        return {"exceptions": x, "observations": n, "lr_pof": np.nan, "p_value": np.nan}

    phat = x / n
    lr = -2 * (
        (n - x) * np.log((1 - p) / (1 - phat))
        + x * np.log(p / phat)
    )
    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "exceptions": x,
        "observations": n,
        "expected_exceptions": n * p,
        "exception_rate": phat,
        "lr_pof": float(lr),
        "p_value": float(p_value),
    }


def christoffersen_independence_test(exceptions: pd.Series) -> dict:
    e = exceptions.astype(int).values
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(e)):
        if e[i-1] == 0 and e[i] == 0: n00 += 1
        if e[i-1] == 0 and e[i] == 1: n01 += 1
        if e[i-1] == 1 and e[i] == 0: n10 += 1
        if e[i-1] == 1 and e[i] == 1: n11 += 1

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    eps = 1e-12
    log_l_uncond = (
        (n00 + n10) * np.log(1 - pi + eps)
        + (n01 + n11) * np.log(pi + eps)
    )
    log_l_cond = (
        n00 * np.log(1 - pi01 + eps)
        + n01 * np.log(pi01 + eps)
        + n10 * np.log(1 - pi11 + eps)
        + n11 * np.log(pi11 + eps)
    )

    lr_ind = -2 * (log_l_uncond - log_l_cond)
    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": pi01,
        "pi11": pi11,
        "lr_independence": float(lr_ind),
        "p_value": float(p_value),
    }
