"""E3e: the one-step versus traffic-level rank agreement inside each model family.

The anti-correlation between the one-step ranking and the composite (E3) mixes two effects: the
mechanistic entries are poor one step ahead and good at the traffic level, and inside the learned
family the best imitators are also the least stable. This module computes Kendall's tau-b between
the P0 ranking and the composite, the P3 unstable fraction and the whole-event P1 error inside
every family and over all entries, per dataset (point estimates with the two-sided p-value of
``scipy.stats.kendalltau``; the sample is at most 17 entries, so the intervals of E3 do not apply).
``p_holm`` is the Holm-adjusted p-value over the datasets of one group and one target: the six
datasets are one family of tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from trafficloop.stats.paired import holm
from trafficloop.stats.ranks import _tau_b, kendall_tau

GROUPS = ("all", "network", "mechanistic")
TARGETS = {"composite": "composite", "unstable fraction": "p3_unstable_fraction", "P1 whole": "p1_whole_rmse_s"}
COLUMNS = ["dataset", "group", "n_models", "against", "tau", "p", "p_holm"]


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
    return _with_holm(pd.DataFrame(rows, columns=COLUMNS[:-1]), ["group", "against"])


def _with_holm(frame: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    """Add ``p_holm``: Holm's adjustment of ``p`` inside every combination of the ``family`` columns
    (the rows of one combination are the datasets)."""
    frame = frame.copy()
    frame["p_holm"] = float("nan")
    for _, index in frame.groupby(family, sort=False).groups.items():
        frame.loc[index, "p_holm"] = holm(frame.loc[index, "p"].to_numpy(dtype=float))
    return frame


# ----------------------------------------------------------------------------- E3g: what an entry is fitted to

#: what every reference entry is fitted to (the column "training or calibration" of the entries table):
#: the families of E3e differ in it, so the correlation over all entries mixes family and objective.
#: "one-step objective": the objective is defined on the next acceleration alone, as a squared error (mlp,
#: gru, lstm, maml_lstm, perl, pidl with its physics term, stability_mlp with its stability penalty), a
#: likelihood (mdn_lstm), a denoising loss (diffusion) or nearest neighbours of the reference states (knn).
#: The certified hybrid is a group of its own: a core calibrated in closed loop and a bounded residual
#: trained on the one-step error.
OBJECTIVES = {
    "closed-loop calibration": ("idm", "idm_median", "gipps", "gipps_median", "newell", "newell_median"),
    "one-step objective": ("knn", "mlp", "gru", "lstm", "mdn_lstm", "maml_lstm", "diffusion", "pidl", "perl", "stability_mlp"),
    "rollout or multi-step loss": ("mlp_loop", "gru_loop", "lstm_loop", "seq2seq"),
    "policy (reward, adversarial)": ("ddpg", "gail"),
    "calibrated core, certified residual": ("residual_idm_certified",),
    "none": ("persistence", "constant_speed"),
}
UNASSIGNED = "unassigned"
#: sets of objectives inside which tau(P0, composite) is computed
OBJECTIVE_SETS = {
    "one-step objective": ("one-step objective",),
    "one-step objective, rollout or multi-step loss": ("one-step objective", "rollout or multi-step loss"),
    "closed-loop calibration, policy or certified hybrid": ("closed-loop calibration", "policy (reward, adversarial)",
                                                            "calibrated core, certified residual"),
}
OBJECTIVE_RANK_COLUMNS = ["dataset", "objective", "n_models", "models", "p0_rank_median", "p0_rank_min", "p0_rank_max",
                          "composite_rank_median", "composite_rank_min", "composite_rank_max"]
OBJECTIVE_TAU_COLUMNS = ["dataset", "set", "n_models", "tau", "p", "p_holm"]


def objective_of(model: str) -> str:
    """The objective of :data:`OBJECTIVES` of a reference entry (``unassigned`` for any other label)."""
    for name, members in OBJECTIVES.items():
        if model in members:
            return name
    return UNASSIGNED


def objective_tables(rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E3g: the places of the entries under P0 and under the composite by training objective, and
    Kendall's tau-b between the two rankings inside sets of entries with a common kind of objective.

    Ranks are average ranks among all entries of the dataset in ``rankings`` (1 = best; a missing
    value last). The taus use the composite over all entries, as E3e does; a set with fewer than
    three entries is left out.
    """
    ranks, taus = [], []
    for dataset, block in rankings.groupby("dataset", sort=False):
        block = block.assign(objective=[objective_of(m) for m in block["model"]],
                             rank_p0=block["p0_rmse_a"].rank(method="average", na_option="bottom"),
                             rank_c=block["composite"].rank(method="average", na_option="bottom"))
        for name in (*OBJECTIVES, UNASSIGNED):
            part = block[block["objective"] == name]
            if part.empty:
                continue
            ranks.append({"dataset": dataset, "objective": name, "n_models": int(len(part)), "models": ", ".join(part["model"]),
                          "p0_rank_median": float(part["rank_p0"].median()), "p0_rank_min": float(part["rank_p0"].min()),
                          "p0_rank_max": float(part["rank_p0"].max()),
                          "composite_rank_median": float(part["rank_c"].median()),
                          "composite_rank_min": float(part["rank_c"].min()), "composite_rank_max": float(part["rank_c"].max())})
        for name, objectives in OBJECTIVE_SETS.items():
            keep = block[block["objective"].isin(objectives)][["model", "p0_rmse_a", "composite"]].dropna()
            if len(keep) < 3:
                continue
            tau, p = kendall_tau(dict(zip(keep["model"], keep["p0_rmse_a"])), dict(zip(keep["model"], keep["composite"])))
            taus.append({"dataset": dataset, "set": name, "n_models": int(len(keep)), "tau": float(tau), "p": float(p)})
    return (pd.DataFrame(ranks, columns=OBJECTIVE_RANK_COLUMNS),
            _with_holm(pd.DataFrame(taus, columns=OBJECTIVE_TAU_COLUMNS[:-1]), ["set"]))


# ----------------------------------------------------------------------------- E3h (c): contrast between the groups (REV5-1)

N_PERMUTATIONS = 10_000
#: tolerance of "at or below the observed tau" (the observed tau comes from scipy, the permuted ones from ``_tau_b``)
TIE_TOL = 1e-12
CONTRAST_COLUMNS = ["dataset", "n_models", "groups", "tau_observed", "p_observed", "tau_perm_mean", "tau_perm_low", "tau_perm_high",
                    "share_at_or_below", "ratio_mean_observed", "n_perm", "seed"]


def _last(values) -> np.ndarray:
    """NaN -> +inf: a missing value ranks last (as ``robustness.point_tau``)."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isnan(values), np.inf, values)


def group_indices(models: Sequence[str], groups: Mapping[str, Sequence[str]] = OBJECTIVES) -> list[tuple[str, np.ndarray]]:
    """Positions in ``models`` of the members of every group of ``groups`` (in its order, empty groups left
    out); an entry in no group forms a group of its own (named after it), after the groups."""
    out, placed = [], set()
    for name, members in groups.items():
        index = np.array([i for i, m in enumerate(models) if m in set(members)], dtype=np.int64)
        if index.size:
            out.append((name, index))
            placed.update(index.tolist())
    out += [(f"{UNASSIGNED}: {m}", np.array([i])) for i, m in enumerate(models) if i not in placed]
    return out


def group_permutation_taus(p0: np.ndarray, composite: np.ndarray, index: Sequence[np.ndarray], n_perm: int = N_PERMUTATIONS,
                           seed: int = 0) -> np.ndarray:
    """Kendall's tau-b between ``p0`` and ``composite`` with the composite values permuted inside every group
    of ``index`` (arrays of positions), ``n_perm`` times; a missing value ranks last and moves with its entry's
    value. One Generator ``numpy.random.default_rng(seed)`` permutes the groups in the order of ``index``
    (``Generator.permuted`` along the rows of the tiled group values); a group of one entry draws nothing."""
    rng = np.random.default_rng(seed)
    base = _last(composite)
    matrix = np.tile(base, (n_perm, 1))
    for positions in index:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size > 1:
            matrix[:, positions] = rng.permuted(np.tile(base[positions], (n_perm, 1)), axis=1)
    return _tau_b(np.tile(_last(p0), (n_perm, 1)), matrix)


def group_contrast_table(rankings: pd.DataFrame, n_perm: int = N_PERMUTATIONS, seed: int = 0,
                         groups: Mapping[str, Sequence[str]] = OBJECTIVES) -> pd.DataFrame:
    """REV5-1 (c), exploratory: how much of tau(P0, composite) the contrast between the groups of
    ``groups`` (the training objectives) produces alone.

    Per dataset of ``rankings`` (the E3 table: ``p0_rmse_a`` and ``composite``, i.e. the ``last`` rule over
    all its entries): the observed tau-b and p (``kendall_tau`` with a missing value last, as
    ``robustness.point_tau``), and over ``n_perm`` permutations of the composite values inside every group
    (:func:`group_permutation_taus`, one Generator seeded with ``seed`` per dataset): the mean, the 2.5 % and
    97.5 % quantiles, ``share_at_or_below`` = (k + 1) / (n_perm + 1) with k the permutations whose tau is at
    or below the observed one (the observed arrangement counted once), and mean / observed.
    """
    rows = []
    for dataset, block in rankings.groupby("dataset", sort=False):
        models = block["model"].tolist()
        p0, composite = block["p0_rmse_a"].to_numpy(dtype=np.float64), block["composite"].to_numpy(dtype=np.float64)
        p0_last, composite_last = _last(p0), _last(composite)
        tau, p = kendall_tau(dict(zip(models, map(float, p0_last))), dict(zip(models, map(float, composite_last))))
        index = group_indices(models, groups)
        taus = group_permutation_taus(p0, composite, [i for _, i in index], n_perm, seed)
        finite = taus[np.isfinite(taus)]
        k = int(np.sum(finite <= tau + TIE_TOL)) if np.isfinite(tau) else 0
        mean = float(finite.mean()) if finite.size else float("nan")
        low, high = (float(v) for v in np.quantile(finite, [0.025, 0.975])) if finite.size else (float("nan"),) * 2
        rows.append({"dataset": dataset, "n_models": len(models), "groups": "; ".join(f"{name} {len(i)}" for name, i in index),
                     "tau_observed": tau, "p_observed": p, "tau_perm_mean": mean, "tau_perm_low": low, "tau_perm_high": high,
                     "share_at_or_below": (k + 1) / (finite.size + 1) if np.isfinite(tau) else float("nan"),
                     "ratio_mean_observed": mean / tau if np.isfinite(tau) and tau != 0 else float("nan"),
                     "n_perm": int(finite.size), "seed": int(seed)})
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)
