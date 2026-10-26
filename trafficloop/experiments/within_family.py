"""E3e: the one-step versus traffic-level rank agreement inside each model family.

The anti-correlation between the one-step ranking and the composite (E3) mixes two effects: the
mechanistic entries are poor one step ahead and good at the traffic level, and inside the learned
family the best imitators are also the least stable. This module computes Kendall's tau-b between
the P0 ranking and the composite, the P3 unstable fraction and the whole-event P1 error inside
every family and over all entries, per dataset (point estimates with the two-sided p-value of
``scipy.stats.kendalltau``; the sample is at most 17 entries, so the intervals of E3 do not apply).
"""

from __future__ import annotations

import pandas as pd

from trafficloop.stats.ranks import kendall_tau

GROUPS = ("all", "network", "mechanistic")
TARGETS = {"composite": "composite", "unstable fraction": "p3_unstable_fraction", "P1 whole": "p1_whole_rmse_s"}
COLUMNS = ["dataset", "group", "n_models", "against", "tau", "p"]


def within_family_taus(rankings: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau-b between the P0 ranking and each target inside every group of :data:`GROUPS`.

    ``rankings``: the E3 table (``dataset, model, family, p0_rmse_a, composite, p3_unstable_fraction,
    p1_whole_rmse_s``). A group with fewer than three entries with both values is left out.
    """
    rows = []
    for dataset, block in rankings.groupby("dataset", sort=True):
        for group in GROUPS:
            part = block if group == "all" else block[block["family"] == group]
            for name, column in TARGETS.items():
                keep = part[["model", "p0_rmse_a", column]].dropna()
                if len(keep) < 3:
                    continue
                tau, p = kendall_tau(dict(zip(keep["model"], keep["p0_rmse_a"])), dict(zip(keep["model"], keep[column])))
                rows.append({"dataset": dataset, "group": group, "n_models": int(len(keep)), "against": name,
                             "tau": float(tau), "p": float(p)})
    return pd.DataFrame(rows, columns=COLUMNS)
