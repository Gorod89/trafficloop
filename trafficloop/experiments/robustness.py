"""E3f: robustness of the rank correlation between one-step accuracy and the composite; exploratory,
added after the results on 2026-11-16.

The headline of E3 is Kendall's tau-b between the P0 ranking (``p0_rmse_a``) and the composite
(``v1``: mean normalised rank of the four inputs of :data:`~trafficloop.experiments.analysis.COMPOSITE_COLUMNS`,
a missing input ranked last) over the fixed set of reference entries. This module recomputes it

* under two rules for a missing input (``last``: the rule of the composite; ``skip``: an entry is
  averaged over the inputs it has) and on three sets of entries (all, without the baselines, the
  entries with all four inputs), with percentile and bias-corrected percentile intervals
  (:func:`missing_rule_table`, :func:`tau_bias_table`);
* with a collision gate, an admissibility threshold on the share of colliding followers in front
  of the composite (:func:`collision_gate_table`), and with the collision input replaced by
  measures that do not reward followers that never react (:func:`collision_input_table`), and with
  other rules for the stability verdict, one of which counts neutral stability as instability
  (:func:`stability_rule_table`);
* leaving one entry out (:func:`leave_one_out`), on other training seeds and folds
  (:func:`seed_fold_table`), and leaving one of the four inputs out (:func:`input_ablation_table`);
* and evaluates the registered rules H5.3 (:func:`h53_table`) and H5.1 (:func:`h51_table`), and
  tabulates their statistics (with those of H5.4) in the other training runs and the block run
  (:func:`run_statistics_table`, decision HD-5).

Everything comes from the E3 rankings and the replicates of
:func:`~trafficloop.experiments.analysis.e3_replicates` (one call per dataset, so that every
treatment shares the same draws); nothing is written here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from trafficloop.experiments import analysis as an
from trafficloop.experiments import within_family as wf
from trafficloop.protocols.composite import INPUTS, _aligned, _mean_over, _sum_over, composite_matrix
from trafficloop.stats.bootstrap import draw_counts, percentile_ci
from trafficloop.stats.ranks import _tau_b, kendall_tau

P0, P1_WHOLE, STABILITY, COLLISIONS = "p0_rmse_a", "p1_whole_rmse_s", "p3_unstable_fraction", "p2_collisions_100"
#: the E3 columns whose replicates E3f needs (one call of e3_replicates per dataset)
REPLICATE_COLUMNS = [P0, P1_WHOLE, *an.COMPOSITE_COLUMNS.values(), "composite"]
SETS = ("all", "no baselines", "complete")
RULES = ("last", "skip")
#: treatment -> (set, rule); "published" is the composite of E3
TREATMENTS = {
    "published": ("all", "last"),
    "no baselines, last": ("no baselines", "last"),
    "all, skip": ("all", "skip"),
    "no baselines, skip": ("no baselines", "skip"),
    "complete, last": ("complete", "last"),
}
#: the pairs of the bias table: (name a, column a, name b, column b)
BIAS_PAIRS = (("P0", P0, "composite", "composite"), ("P0", P0, "P3 unstable", STABILITY),
              ("P1 whole", P1_WHOLE, "composite", "composite"))
THETAS = (0.0, 0.01)
GATE_SETS = ("all", "no baselines")
#: (fold, seed) of the driver split in the seed and fold check
SEED_FOLD_RUNS = ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 0), (4, 0))
#: files every entry needs in the seed and fold check (P0 and the four composite inputs)
NEEDED_FILES = ("train.json", "p2.json", "p3.json")
TRAJECTORY = {"P0": P0, "P1 whole": P1_WHOLE}
Z975 = 1.959964
#: tolerance of "strictly below the point tau" (the replicate taus and the point tau are computed
#: by two implementations that may differ in the last bits for the same ordering)
TIE_TOL = 1e-12

STAT_COLUMNS = ["tau", "p", "low", "high", "tau_boot_mean", "tau_boot_median", "share_below", "bc_low", "bc_high", "n_boot"]
MISSING_COLUMNS = ["dataset", "treatment", "set", "rule", "n_models", "n_missing_stability", "missing_stability",
                   *STAT_COLUMNS, "top3", "note"]
BIAS_COLUMNS = ["dataset", "ranking_a", "ranking_b", "n_models", *STAT_COLUMNS, "note"]
GATE_COLUMNS = ["dataset", "set", "theta", "n_models", "n_profiles_in_range", "n_followers", "n_admissible", "admissible",
                "top3", *STAT_COLUMNS, "note"]
LOO_COLUMNS = ["dataset", "n_models", "tau_all", "loo_min", "loo_max", "entry_at_min", "entry_at_max"]
LOO_ENTRY_COLUMNS = ["dataset", "left_out", "family", "n_models", "tau", "p", "delta"]
SEED_FOLD_COLUMNS = ["dataset", "fold", "seed", "set", "n_models", "missing", "tau", "p", "top3"]
INPUT_COLUMNS = ["dataset", "set", "left_out", "inputs", "n_models", *STAT_COLUMNS, "top3", "note"]
#: measures that can stand for the collision input (:func:`collision_measures`); the first is the input itself
COLLISION_INPUTS = ("followers", "blocked", "profiles")
COLLISION_INPUT_COLUMNS = ["dataset", "set", "input", "n_models", "n_profiles_in_range", *STAT_COLUMNS, "top3", "note"]
#: rules for the verdict on an equilibrium (:func:`stability_measures`); the first is the protocol's
STABILITY_RULES = ("published", "tolerance 1e-3", "neutral counts as unstable")
#: a gain at 0.1 Hz above 1 - NEUTRAL_EPS is "not damped" under the third rule
NEUTRAL_GAIN, NEUTRAL_EPS = "gain_0.1", 1e-3
STABILITY_RULE_COLUMNS = ["dataset", "set", "rule", "n_models", "n_changed", "changed", *STAT_COLUMNS, "top3", "note"]
COLLISION_ENTRY_COLUMNS = ["dataset", "model", "family", *COLLISION_INPUTS, *(f"composite_{k}" for k in COLLISION_INPUTS),
                           *(f"rank_{k}" for k in COLLISION_INPUTS)]
H53_COLUMNS = ["dataset", "set", "trajectory", "n_models", "top3_trajectory", "top3_composite", "dropped", "p_boot_ge2",
               "p_boot_coincide", "n_boot", "note"]
#: runs of other splits in the run statistics, ``(split, fold, seed)`` per dataset (decision HD-5)
EXTRA_RUNS = {"highd": (("block", 0, 0),)}
#: files every entry needs in the run statistics (NEEDED_FILES and P1 for H5.4 at 10 s and 30 s)
RUN_FILES = (*NEEDED_FILES, "p1.json")
RUN_STAT_COLUMNS = ["dataset", "split", "fold", "seed", "n_models", "missing", "tau", "p", "top3", "p0_top3", "whole_top3",
                    "h53_dropped_p0", "h53_dropped_whole", "h51_n", "h51_within10", "h51_beyond30", "h51_better30", "h51_worse30",
                    "h51_best", "h51_best_ratio", "h54_tau_h10", "h54_tau_h30", "h54_tau_whole"]
#: integer columns of the run statistics (missing in a run without entries)
RUN_COUNT_COLUMNS = ("h53_dropped_p0", "h53_dropped_whole", "h51_n", "h51_within10", "h51_beyond30", "h51_better30", "h51_worse30")
H51_COLUMNS = ["dataset", "group", "n_models", "n_within_10", "n_ci_within_10", "n_better_30", "n_better_ci", "n_worse_30",
               "best_model", "best_ratio", "best_low", "best_high"]


# ----------------------------------------------------------------------------- sets and composites


def entry_set(rankings: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of the E3 table in the set ``name`` of :data:`SETS` (order kept)."""
    if name == "all":
        return rankings
    if name == "no baselines":
        return rankings[rankings["family"] != "baseline"]
    if name == "complete":
        values = rankings[list(an.COMPOSITE_COLUMNS.values())].to_numpy(dtype=np.float64)
        return rankings[np.isfinite(values).all(axis=1)]
    raise KeyError(f"unknown set {name!r}; sets: {list(SETS)}")


def skip_matrix(values: Mapping[str, np.ndarray]) -> np.ndarray:
    """The ``skip`` composite of every row of ``values[input]`` (``[rows, models]``).

    Per input the models with a finite value are ranked (average ranks on ties) and normalised to
    ``(rank - 1) / (n_valid - 1)`` (0 when one model has a value); the composite of a model is the
    mean of its normalised ranks over the inputs it has, NaN when it has none.
    """
    parts = []
    for name in INPUTS:
        m = np.atleast_2d(np.asarray(values[name], dtype=np.float64))
        m = np.where(np.isfinite(m), m, np.nan)
        rank = pd.DataFrame(m).rank(axis=1, method="average", na_option="keep").to_numpy(dtype=np.float64)
        n_valid = (~np.isnan(m)).sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            nrank = np.where(n_valid > 1, (rank - 1.0) / (n_valid - 1), 0.0)
        parts.append(np.where(np.isnan(m), np.nan, nrank))
    stack = np.stack(parts, axis=-1)
    count = (~np.isnan(stack)).sum(axis=-1)
    total = np.nansum(stack, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _inputs_of(block: pd.DataFrame) -> dict[str, np.ndarray]:
    """The four inputs of the rows of ``block`` as one row ``[1, models]`` each."""
    return {name: block[col].to_numpy(dtype=np.float64)[None, :] for name, col in an.COMPOSITE_COLUMNS.items()}


def composite_point(block: pd.DataFrame, rule: str) -> np.ndarray:
    """Composite of the rows of ``block`` under ``rule`` (``last``: :func:`an._composite`)."""
    if rule == "last":
        return an._composite(block.reset_index(drop=True))
    if rule == "skip":
        return skip_matrix(_inputs_of(block))[0]
    raise KeyError(f"unknown rule {rule!r}; rules: {list(RULES)}")


def _matrix(samples: Mapping[str, Mapping[str, np.ndarray]], column: str, models: Sequence[str]) -> np.ndarray | None:
    """Replicates ``[n_boot, models]`` of ``column``, or None without replicates."""
    if column not in samples:
        return None
    return np.column_stack([np.asarray(samples[column][m], dtype=np.float64) for m in models])


def composite_replicates(samples: Mapping, models: Sequence[str], rule: str) -> np.ndarray | None:
    """Composite of ``models`` in every replicate, from the replicates of the four inputs
    restricted to ``models``; None when an input has no replicates."""
    values = {}
    for name, column in an.COMPOSITE_COLUMNS.items():
        matrix = _matrix(samples, column, models)
        if matrix is None:
            return None
        values[name] = matrix
    if rule == "last":
        return composite_matrix(values, INPUTS)
    if rule == "skip":
        return skip_matrix(values)
    raise KeyError(f"unknown rule {rule!r}; rules: {list(RULES)}")


# ----------------------------------------------------------------------------- tau statistics


def _inf(x: np.ndarray) -> np.ndarray:
    """NaN -> +inf: a missing value ranks last."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(x), np.inf, x)


def point_tau(models: Sequence[str], a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Kendall's tau-b and its p-value of two value vectors over ``models``, a missing value last."""
    a, b = _inf(a), _inf(b)
    return kendall_tau(dict(zip(models, map(float, a))), dict(zip(models, map(float, b))))


def replicate_taus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kendall's tau-b of every replicate (rows of ``[n_boot, models]``), a missing value last."""
    return _tau_b(_inf(a), _inf(b))


def tau_statistics(tau: float, taus: np.ndarray | None) -> dict:
    """Percentile interval, mean, median, share below the point and the bias-corrected percentile
    interval of the replicate taus around the point ``tau`` (NaN without replicates)."""
    out = dict.fromkeys(("low", "high", "tau_boot_mean", "tau_boot_median", "share_below", "bc_low", "bc_high"), float("nan"))
    out["n_boot"] = 0
    if taus is None:
        return out
    finite = np.asarray(taus, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return out
    b = finite.size
    low, high = percentile_ci(finite)
    share = float(np.mean(finite < tau - TIE_TOL)) if np.isfinite(tau) else float("nan")
    out.update(low=low, high=high, tau_boot_mean=float(finite.mean()), tau_boot_median=float(np.median(finite)),
               share_below=share, n_boot=int(b))
    if np.isfinite(share):
        z0 = stats.norm.ppf(np.clip(share, 1.0 / (b + 1), b / (b + 1.0)))
        q = stats.norm.cdf([2 * z0 - Z975, 2 * z0 + Z975])
        out["bc_low"], out["bc_high"] = (float(v) for v in np.percentile(finite, 100.0 * q))
    return out


def _tau_row(models, a, b, rep_a, rep_b) -> dict:
    tau, p = point_tau(models, a, b)
    taus = None if rep_a is None or rep_b is None else replicate_taus(rep_a, rep_b)
    return {"tau": tau, "p": p, **tau_statistics(tau, taus)}


def top_k(models: Sequence[str], values: np.ndarray, k: int = 3) -> list[str]:
    """The ``k`` lowest values (a missing value last), ties broken by model name (as :func:`an._ordered`)."""
    block = pd.DataFrame({"model": list(models), "v": np.asarray(values, dtype=np.float64)})
    return an._ordered(block, "v")["model"].head(k).tolist()


def replicate_top_k(models: Sequence[str], matrix: np.ndarray, k: int = 3) -> list[set[str]]:
    """:func:`top_k` of every replicate (rows of ``[n_boot, models]``)."""
    names = np.asarray(list(models))
    by_name = np.argsort(np.argsort(names, kind="stable"), kind="stable")
    values = _inf(matrix)
    return [set(names[np.lexsort((by_name, row))[:k]]) for row in values]


def _names(models: Sequence[str]) -> str:
    return ", ".join(models)


# ----------------------------------------------------------------------------- B: missing rule and bias


def missing_rule_table(rankings: pd.DataFrame, samples: Mapping) -> pd.DataFrame:
    """tau(P0, composite) under every treatment of :data:`TREATMENTS` with its bootstrap statistics.

    Replicate ``r``: the composite recomputed with the treatment from the ``r``-th replicates of
    the four inputs restricted to the set, against the ``r``-th P0 replicate.
    """
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    rows = []
    for treatment, (name, rule) in TREATMENTS.items():
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        missing = [m for m, v in zip(models, block[STABILITY].to_numpy(dtype=np.float64)) if not np.isfinite(v)]
        composite = composite_point(block, rule)
        rep_p0 = _matrix(samples, P0, models)
        rep_c = composite_replicates(samples, models, rule)
        note = "" if rep_p0 is not None and rep_c is not None else "no replicates"
        rows.append({"dataset": dataset, "treatment": treatment, "set": name, "rule": rule, "n_models": len(models),
                     "n_missing_stability": len(missing), "missing_stability": _names(missing),
                     **_tau_row(models, block[P0].to_numpy(dtype=np.float64), composite, rep_p0, rep_c),
                     "top3": _names(top_k(models, composite)), "note": note})
    return pd.DataFrame(rows, columns=MISSING_COLUMNS)


def tau_bias_table(rankings: pd.DataFrame, samples: Mapping) -> pd.DataFrame:
    """The statistics of :func:`tau_statistics` for the pairs of :data:`BIAS_PAIRS` over all entries
    (the published composite, a missing value last, as ``an._tau_row``)."""
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    models = rankings["model"].tolist()
    rows = []
    for name_a, col_a, name_b, col_b in BIAS_PAIRS:
        rep_a, rep_b = _matrix(samples, col_a, models), _matrix(samples, col_b, models)
        rows.append({"dataset": dataset, "ranking_a": name_a, "ranking_b": name_b, "n_models": len(models),
                     **_tau_row(models, rankings[col_a].to_numpy(dtype=np.float64), rankings[col_b].to_numpy(dtype=np.float64),
                                rep_a, rep_b),
                     "note": "" if rep_a is not None and rep_b is not None else "no replicates"})
    return pd.DataFrame(rows, columns=BIAS_COLUMNS)


def reproduction_check(missing_rule: pd.DataFrame, kendall: pd.DataFrame) -> pd.DataFrame:
    """Absolute differences of tau, low and high between the ``published`` rows and the
    (P0, composite) rows of the E3 Kendall table, per dataset."""
    published = missing_rule[missing_rule["treatment"] == "published"].set_index("dataset")
    reference = kendall[(kendall["ranking_a"] == "P0") & (kendall["ranking_b"] == "composite")].set_index("dataset")
    rows = []
    for dataset in published.index:
        if dataset not in reference.index:
            rows.append({"dataset": dataset, "tau": float("nan"), "low": float("nan"), "high": float("nan")})
            continue
        rows.append({"dataset": dataset, **{c: abs(float(published.at[dataset, c]) - float(reference.at[dataset, c]))
                                            for c in ("tau", "low", "high")}})
    return pd.DataFrame(rows, columns=["dataset", "tau", "low", "high"])


# ----------------------------------------------------------------------------- C: collision gate


def in_range_profiles(root, dataset: str, labels: Sequence[str], fold: int = 0, seed: int = 0) -> int:
    """Number of in-range P2 profiles (``n_profiles_in_range`` of ``p2.json``), the same for every entry
    and equal to the in-range rows of the largest platoon in ``p2.profiles.parquet`` (the synthetic
    pulse included, as in ``collisions_100_in_range``)."""
    found = {}
    for label in labels:
        directory = an.run_dir(root, dataset, label, fold=fold, seed=seed)
        summary = an._summary(directory, "p2")
        if summary is None:
            continue
        n = int(summary["n_profiles_in_range"])
        profiles = an._read(directory / "p2.profiles.parquet", ["profile", "n_vehicles", "in_range"])
        if profiles is not None:
            rows = profiles[profiles["in_range"].astype(bool) & (profiles["n_vehicles"] == profiles["n_vehicles"].max())]
            if len(rows) != n:
                raise ValueError(f"{dataset}/{label}: n_profiles_in_range {n}, {len(rows)} in-range rows in p2.profiles.parquet")
        found[label] = n
    if not found:
        raise ValueError(f"{dataset}: no p2.json")
    if len(set(found.values())) != 1:
        raise ValueError(f"{dataset}: n_profiles_in_range differs between entries: {sorted(set(found.values()))}")
    return next(iter(found.values()))


def admissible(collisions: np.ndarray, followers: int, theta: float) -> np.ndarray:
    """Share of colliding followers ``collisions / followers <= theta``; a missing value is inadmissible."""
    collisions = np.asarray(collisions, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(collisions) & (collisions <= theta * followers + 1e-9)


def collision_gate_table(rankings: pd.DataFrame, samples: Mapping, n_profiles: int, thetas: Sequence[float] = THETAS,
                         sets: Sequence[str] = GATE_SETS, vehicles: int = 100) -> pd.DataFrame:
    """tau(P0, gated score): composite (``last`` rule on the set) + 1 for the inadmissible entries.

    ``n_profiles`` in-range profiles of ``vehicles`` followers each give the denominator of the
    collision share; the gate is re-evaluated in every replicate from the replicate collisions.
    """
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    followers = int(vehicles * n_profiles)
    rows = []
    for name in sets:
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        composite = composite_point(block, "last")
        rep_p0, rep_c = _matrix(samples, P0, models), composite_replicates(samples, models, "last")
        rep_coll = _matrix(samples, COLLISIONS, models)
        for theta in thetas:
            ok = admissible(block[COLLISIONS].to_numpy(dtype=np.float64), followers, theta)
            gated = composite + (~ok).astype(np.float64)
            rep_g = None
            if rep_c is not None and rep_coll is not None:
                rep_g = rep_c + (~admissible(rep_coll, followers, theta)).astype(np.float64)
            rows.append({"dataset": dataset, "set": name, "theta": float(theta), "n_models": len(models),
                         "n_profiles_in_range": int(n_profiles), "n_followers": followers, "n_admissible": int(ok.sum()),
                         "admissible": _names([m for m, a in zip(models, ok) if a]), "top3": _names(top_k(models, gated)),
                         **_tau_row(models, block[P0].to_numpy(dtype=np.float64), gated, rep_p0, rep_g),
                         "note": "" if rep_p0 is not None and rep_g is not None else "no replicates"})
    return pd.DataFrame(rows, columns=GATE_COLUMNS)


# ----------------------------------------------------------------------------- C2: the collision input


def collision_measures(vehicles: pd.DataFrame, profiles: Sequence[str]) -> pd.DataFrame:
    """Three collision measures of every profile of ``profiles`` from ``p2.vehicles.parquet``
    (``profile, index, collided``; index 0 is the leader, the others are the followers in order).

    ``followers``: followers with a collision, the collision input of the composite. A collided
    vehicle keeps driving and a follower counts when its own gap closes, so a platoon of identical
    followers that do not react to their leader (persistence, constant speed) counts one collision
    per profile, that of the first follower: the measure rewards the absence of a reaction.
    ``blocked``: followers at or behind the first colliding one (everything from the first collision
    back is simulated behind a vehicle that has already crashed). ``profiles``: 1 when at least one
    follower collides.
    """
    table = vehicles[vehicles["index"].to_numpy() > 0]
    names = table["profile"].astype(str).to_numpy()
    rows = []
    for name in profiles:
        part = table[names == name].sort_values("index")
        if part.empty:
            raise KeyError(f"no followers of the profile {name!r} in the vehicle table")
        collided = part["collided"].to_numpy(dtype=bool)
        n = int(collided.sum())
        blocked = int(len(collided) - np.argmax(collided)) if n else 0
        rows.append({"profile": name, "followers": float(n), "blocked": float(blocked), "profiles": float(n > 0)})
    return pd.DataFrame(rows, columns=["profile", *COLLISION_INPUTS])


def collision_inputs(root, dataset: str, labels: Sequence[str], fold: int = 0, seed: int = 0) -> dict[str, pd.DataFrame]:
    """:func:`collision_measures` of the in-range profiles of every entry (the in-range rows of the
    largest platoon in ``p2.profiles.parquet``, as the composite uses them); ``followers`` must
    reproduce the ``collisions`` of those rows."""
    out = {}
    for label in labels:
        directory = an.run_dir(root, dataset, label, fold=fold, seed=seed)
        profiles = an._read(directory / "p2.profiles.parquet", ["profile", "n_vehicles", "in_range", "collisions"])
        vehicles = an._read(directory / "p2.vehicles.parquet", ["profile", "index", "collided"])
        if profiles is None or vehicles is None:
            raise FileNotFoundError(f"{dataset}/{label}: p2.profiles.parquet or p2.vehicles.parquet is missing")
        rows = profiles[profiles["in_range"].astype(bool) & (profiles["n_vehicles"] == profiles["n_vehicles"].max())]
        frame = collision_measures(vehicles, rows["profile"].astype(str).tolist())
        if not np.array_equal(frame["followers"].to_numpy(), rows["collisions"].to_numpy(dtype=np.float64)):
            raise ValueError(f"{dataset}/{label}: the vehicle table does not reproduce the collisions of the profiles")
        out[label] = frame
    return out


def collision_input_table(rankings: pd.DataFrame, samples: Mapping, measures: Mapping[str, pd.DataFrame], n_boot: int = 1000,
                          seed: int = 0, sets: Sequence[str] = GATE_SETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """tau(P0, composite) with the collision input replaced by each measure of :data:`COLLISION_INPUTS`
    (``followers`` is the composite itself), and the values and composite ranks of every entry.

    The profile counts of the replicates are drawn as in ``an._profile_unit`` (generator
    ``[seed, 1]``, first draw), so that replicate ``r`` of a replaced input belongs to the ``r``-th
    replicates of the other inputs; the table has intervals only when the ``followers`` replicates
    reproduce those of the collision input in ``samples`` exactly.
    """
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    every = rankings["model"].tolist()
    aligned = _aligned({m: measures[m] for m in every}, "profile", COLLISION_INPUTS, "collision measures")
    n_profiles = aligned["followers"].shape[1]
    counts = draw_counts(np.random.default_rng([seed, 1]), n_profiles, n_boot).astype(np.float64)
    point = {key: aligned[key].sum(axis=1) for key in COLLISION_INPUTS}
    replicates = {key: _sum_over(counts, aligned[key]) for key in COLLISION_INPUTS}
    published = _matrix(samples, COLLISIONS, every)
    reproduced = (published is not None and published.shape == replicates["followers"].shape
                  and np.array_equal(published, replicates["followers"], equal_nan=True))
    position = {m: i for i, m in enumerate(every)}
    rows, entries = [], {"dataset": dataset, "model": every, "family": rankings["family"].tolist()}
    for name in sets:
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        index = [position[m] for m in models]
        base, base_rep = _inputs_of(block), {key: _matrix(samples, column, models) for key, column in an.COMPOSITE_COLUMNS.items()}
        p0, rep_p0 = block[P0].to_numpy(dtype=np.float64), _matrix(samples, P0, models)
        have = reproduced and rep_p0 is not None and all(base_rep[key] is not None for key in INPUTS if key != "collisions_100")
        for key in COLLISION_INPUTS:
            composite = composite_matrix({**base, "collisions_100": point[key][index][None, :]}, INPUTS)[0]
            rep_c = composite_matrix({**base_rep, "collisions_100": replicates[key][:, index]}, INPUTS) if have else None
            if name == "all":
                entries[key] = point[key]
                entries[f"composite_{key}"] = composite
                entries[f"rank_{key}"] = pd.Series(composite).rank(method="average").to_numpy()
            rows.append({"dataset": dataset, "set": name, "input": key, "n_models": len(models), "n_profiles_in_range": int(n_profiles),
                         **_tau_row(models, p0, composite, rep_p0, rep_c), "top3": _names(top_k(models, composite)),
                         "note": "" if have else "no replicates"})
    return pd.DataFrame(rows, columns=COLLISION_INPUT_COLUMNS), pd.DataFrame(entries, columns=COLLISION_ENTRY_COLUMNS)


# ----------------------------------------------------------------------------- C3: the stability verdict


def stability_measures(speeds: pd.DataFrame) -> pd.DataFrame:
    """Per equilibrium speed of ``p3.speeds.parquet`` the verdict of every rule of :data:`STABILITY_RULES`
    (1 unstable, 0 stable, NaN where the speed has no equilibrium).

    ``published``: the verdict of the protocol, maximal gain above ``1 + 1e-5``. ``tolerance 1e-3``: the
    same with ``1e-3``. ``neutral counts as unstable``: unstable under the published rule or passing a
    perturbation of 0.1 Hz on with a gain above ``1 - NEUTRAL_EPS``; a model that neither amplifies nor
    damps (a pure time shift such as Newell's, whose gain is one at every frequency) is stable under the
    published rule and unstable under this one.
    """
    equilibrium = speeds["converged"].to_numpy(dtype=bool, na_value=False) & speeds["tracks"].to_numpy(dtype=bool, na_value=False)
    published = speeds["unstable_numerical"].to_numpy(dtype=np.float64, na_value=np.nan)
    loose = speeds["unstable_1e3"].to_numpy(dtype=np.float64, na_value=np.nan)
    gain = speeds[NEUTRAL_GAIN].to_numpy(dtype=np.float64, na_value=np.nan)
    neutral = np.where((published > 0.5) | (gain > 1.0 - NEUTRAL_EPS), 1.0, 0.0)
    values = {"published": published, "tolerance 1e-3": loose, "neutral counts as unstable": neutral}
    out = {"v_e": speeds["v_e"].to_numpy(dtype=np.float64)}
    for rule in STABILITY_RULES:
        out[rule] = np.where(equilibrium, values[rule], np.nan)
    return pd.DataFrame(out)


def stability_inputs(root, dataset: str, labels: Sequence[str], fold: int = 0, seed: int = 0) -> dict[str, pd.DataFrame]:
    """:func:`stability_measures` of every entry that has ``p3.speeds.parquet``."""
    out = {}
    for label in labels:
        speeds = an._read(an.run_dir(root, dataset, label, fold=fold, seed=seed) / "p3.speeds.parquet")
        if speeds is not None:
            out[label] = stability_measures(speeds)
    return out


def stability_rule_table(rankings: pd.DataFrame, samples: Mapping, measures: Mapping[str, pd.DataFrame], n_profiles: int,
                         n_boot: int = 1000, seed: int = 0, sets: Sequence[str] = GATE_SETS) -> pd.DataFrame:
    """tau(P0, composite) with the unstable fraction recomputed under every rule of :data:`STABILITY_RULES`.

    The speed counts of the replicates are drawn as in ``an._profile_unit`` (generator ``[seed, 1]``: the
    profile counts for ``n_profiles`` profiles first, then the speed counts); intervals are given only
    when the ``published`` replicates reproduce those of the unstable fraction in ``samples`` exactly.
    ``changed`` lists the entries of the set whose unstable fraction differs from the published one.
    """
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    every = rankings["model"].tolist()
    have = [m for m in every if m in measures]
    aligned = _aligned({m: measures[m] for m in have}, "v_e", STABILITY_RULES, "stability measures")
    rng = np.random.default_rng([seed, 1])
    draw_counts(rng, n_profiles, n_boot)  # the profile counts come first in an._profile_unit
    counts = draw_counts(rng, aligned["published"].shape[1], n_boot).astype(np.float64)
    position = {m: i for i, m in enumerate(have)}

    def of(rule: str, models: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Point values [models] and replicates [n_boot, models] of the unstable fraction under ``rule``."""
        with np.errstate(invalid="ignore"):
            point_all = np.array([np.nanmean(row) if np.isfinite(row).any() else np.nan for row in aligned[rule]])
        rep_all = _mean_over(counts, aligned[rule])
        point = np.array([point_all[position[m]] if m in position else np.nan for m in models])
        rep = np.column_stack([rep_all[:, position[m]] if m in position else np.full(n_boot, np.nan) for m in models])
        return point, rep

    published_point, published_rep = of("published", every)
    reference = _matrix(samples, STABILITY, every)
    reproduced = (reference is not None and reference.shape == published_rep.shape
                  and np.array_equal(reference, published_rep, equal_nan=True)
                  and np.allclose(published_point, rankings[STABILITY].to_numpy(dtype=np.float64), equal_nan=True, atol=1e-12))
    rows = []
    for name in sets:
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        base, base_rep = _inputs_of(block), {key: _matrix(samples, column, models) for key, column in an.COMPOSITE_COLUMNS.items()}
        p0, rep_p0 = block[P0].to_numpy(dtype=np.float64), _matrix(samples, P0, models)
        ok = reproduced and rep_p0 is not None and all(base_rep[key] is not None for key in INPUTS if key != "unstable_fraction")
        reference_point, _ = of("published", models)
        for rule in STABILITY_RULES:
            point, rep = of(rule, models)
            composite = composite_matrix({**base, "unstable_fraction": point[None, :]}, INPUTS)[0]
            rep_c = composite_matrix({**base_rep, "unstable_fraction": rep}, INPUTS) if ok else None
            changed = [m for m, a, b in zip(models, reference_point, point) if not (np.isnan(a) and np.isnan(b)) and abs(a - b) > 1e-12]
            rows.append({"dataset": dataset, "set": name, "rule": rule, "n_models": len(models), "n_changed": len(changed),
                         "changed": _names(changed), **_tau_row(models, p0, composite, rep_p0, rep_c),
                         "top3": _names(top_k(models, composite)), "note": "" if ok else "no replicates"})
    return pd.DataFrame(rows, columns=STABILITY_RULE_COLUMNS)


# ----------------------------------------------------------------------------- D, E: entries, seeds, folds


def leave_one_out(rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """tau(P0, composite ``last``) recomputed without each entry (point values): the per-dataset
    range and the per-entry table (``delta`` = tau without the entry - tau over all)."""
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    models = rankings["model"].tolist()
    tau_all, _ = point_tau(models, rankings[P0].to_numpy(dtype=np.float64), composite_point(rankings, "last"))
    entries = []
    for i, (left, family) in enumerate(zip(models, rankings["family"])):
        block = rankings.drop(index=rankings.index[i])
        tau, p = point_tau(block["model"].tolist(), block[P0].to_numpy(dtype=np.float64), composite_point(block, "last"))
        entries.append({"dataset": dataset, "left_out": left, "family": family, "n_models": len(block), "tau": tau, "p": p,
                        "delta": tau - tau_all})
    entries = pd.DataFrame(entries, columns=LOO_ENTRY_COLUMNS)
    finite = entries[np.isfinite(entries["tau"].to_numpy(dtype=np.float64))]
    summary = {"dataset": dataset, "n_models": len(models), "tau_all": tau_all, "loo_min": float("nan"), "loo_max": float("nan"),
               "entry_at_min": "", "entry_at_max": ""}
    if len(finite):
        low, high = finite.loc[finite["tau"].idxmin()], finite.loc[finite["tau"].idxmax()]
        summary.update(loo_min=float(low["tau"]), loo_max=float(high["tau"]), entry_at_min=low["left_out"],
                       entry_at_max=high["left_out"])
    return pd.DataFrame([summary], columns=LOO_COLUMNS), entries


def seed_fold_table(root, dataset: str, labels: Sequence[str], runs: Sequence[tuple[int, int]] = SEED_FOLD_RUNS,
                    sets: Sequence[str] = GATE_SETS) -> pd.DataFrame:
    """Point tau(P0, composite ``last``) on other folds and seeds of the driver split, over the entries
    of ``labels`` that have every file of :data:`NEEDED_FILES` (the composite recomputed on them)."""
    rows = []
    for fold, seed in runs:
        present = [m for m in labels if all((an.run_dir(root, dataset, m, fold=fold, seed=seed) / f).exists() for f in NEEDED_FILES)]
        missing = [m for m in labels if m not in present]
        rankings = an.e3_rankings(root, dataset, present, fold=fold, seed=seed) if present else None
        for name in sets:
            row = {"dataset": dataset, "fold": fold, "seed": seed, "set": name, "n_models": 0, "missing": _names(missing),
                   "tau": float("nan"), "p": float("nan"), "top3": ""}
            if rankings is not None and len(rankings):
                block = entry_set(rankings, name)
                models = block["model"].tolist()
                composite = composite_point(block, "last")
                tau, p = point_tau(models, block[P0].to_numpy(dtype=np.float64), composite)
                row.update(n_models=len(models), tau=tau, p=p, top3=_names(top_k(models, composite)))
            rows.append(row)
    return pd.DataFrame(rows, columns=SEED_FOLD_COLUMNS)


def input_ablation_table(rankings: pd.DataFrame, samples: Mapping, sets: Sequence[str] = GATE_SETS) -> pd.DataFrame:
    """tau(P0, composite) when one of the four inputs of the composite is left out: the mean
    normalised rank of the three others (a missing value last), on every set of ``sets``, with the
    bootstrap statistics of :func:`tau_statistics`. The row ``left_out = none`` is the composite itself."""
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    rows = []
    for name in sets:
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        point = _inputs_of(block)
        replicates = {key: _matrix(samples, column, models) for key, column in an.COMPOSITE_COLUMNS.items()}
        p0, rep_p0 = block[P0].to_numpy(dtype=np.float64), _matrix(samples, P0, models)
        for left in ("none", *INPUTS):
            kept = tuple(key for key in INPUTS if key != left)
            composite = composite_matrix(point, kept)[0]
            have = all(replicates[key] is not None for key in kept)
            rep_c = composite_matrix({key: replicates[key] for key in kept}, kept) if have else None
            rows.append({"dataset": dataset, "set": name, "left_out": left, "inputs": ", ".join(kept), "n_models": len(models),
                         **_tau_row(models, p0, composite, rep_p0, rep_c), "top3": _names(top_k(models, composite)),
                         "note": "" if have and rep_p0 is not None else "no replicates"})
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


# ----------------------------------------------------------------------------- F, G: registered rules


def h53_table(rankings: pd.DataFrame, samples: Mapping, sets: Sequence[str] = GATE_SETS) -> pd.DataFrame:
    """H5.3: how many of the three best entries by a trajectory error are not among the three best by
    the composite (``last`` rule on the set), on the point values and over the replicates."""
    dataset = rankings["dataset"].iloc[0] if len(rankings) else ""
    rows = []
    for name in sets:
        block = entry_set(rankings, name)
        models = block["model"].tolist()
        composite = composite_point(block, "last")
        top_c = top_k(models, composite)
        rep_c = composite_replicates(samples, models, "last")
        rep_top_c = None if rep_c is None else replicate_top_k(models, rep_c)
        for trajectory, column in TRAJECTORY.items():
            top_t = top_k(models, block[column].to_numpy(dtype=np.float64))
            row = {"dataset": dataset, "set": name, "trajectory": trajectory, "n_models": len(models),
                   "top3_trajectory": _names(top_t), "top3_composite": _names(top_c),
                   "dropped": len(set(top_t) - set(top_c)), "p_boot_ge2": float("nan"), "p_boot_coincide": float("nan"),
                   "n_boot": 0, "note": ""}
            rep_t = _matrix(samples, column, models)
            if rep_top_c is None or rep_t is None:
                row["note"] = "no replicates"
            else:
                dropped = np.array([len(t - c) for t, c in zip(replicate_top_k(models, rep_t), rep_top_c)])
                row.update(p_boot_ge2=float(np.mean(dropped >= 2)), p_boot_coincide=float(np.mean(dropped == 0)),
                           n_boot=int(dropped.size))
            rows.append(row)
    return pd.DataFrame(rows, columns=H53_COLUMNS)


def h51_table(e1: pd.DataFrame, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """H5.1: the P0 ratios to persistence of E1 (``e1_ratios``) per dataset, over the entries that are
    not baselines and over the learned entries only (family not baseline or mechanistic)."""
    if labels is not None:
        e1 = e1[e1["model"].isin(list(labels))]
    rows = []
    for dataset, block in e1.groupby("dataset", sort=False):
        for group, part in (("non-baseline", block[block["family"] != "baseline"]),
                            ("learned", block[~block["family"].isin(["baseline", "mechanistic"])])):
            ratio, low, high = (part[c].to_numpy(dtype=np.float64) for c in ("ratio", "low", "high"))
            best = part.iloc[int(np.nanargmin(ratio))] if np.isfinite(ratio).any() else None
            rows.append({"dataset": dataset, "group": group, "n_models": len(part),
                         "n_within_10": int(((ratio >= 0.9) & (ratio <= 1.1)).sum()),
                         "n_ci_within_10": int(((low >= 0.9) & (high <= 1.1)).sum()),
                         "n_better_30": int((ratio < 0.7).sum()), "n_better_ci": int((high < 1.0).sum()),
                         "n_worse_30": int((ratio > 1.3).sum()),
                         "best_model": "" if best is None else best["model"],
                         "best_ratio": float("nan") if best is None else float(best["ratio"]),
                         "best_low": float("nan") if best is None else float(best["low"]),
                         "best_high": float("nan") if best is None else float(best["high"])})
    return pd.DataFrame(rows, columns=H51_COLUMNS)


def run_statistics_row(rankings: pd.DataFrame, dataset: str, split: str, fold: int, seed: int,
                       missing: Sequence[str] = (), reference: str = an.REFERENCE) -> dict:
    """One row of :func:`run_statistics_table` from the E3 rankings of one run (point values only).

    tau, p, top3: :func:`point_tau` and :func:`top_k` of P0 against :func:`composite_point` (``last``)
    over all entries, as :func:`seed_fold_table` (set ``all``). H5.3: :func:`h53_table` on the set
    ``all`` without replicates. H5.1: :func:`h51_table` (group ``non-baseline``) on the ratios
    ``p0_rmse_a(model) / p0_rmse_a(reference)`` of the run (``train.json`` of the same run).
    H5.4: :func:`an.e3_spacing_vs_speed` on the rankings without replicates.
    """
    row = {"dataset": dataset, "split": split, "fold": fold, "seed": seed, "n_models": 0, "missing": _names(missing),
           **{c: float("nan") for c in RUN_STAT_COLUMNS if c not in ("dataset", "split", "fold", "seed", "n_models", "missing")}}
    row.update(top3="", p0_top3="", whole_top3="", h51_best="")
    if rankings is None or not len(rankings):
        return row
    block = entry_set(rankings, "all")
    models = block["model"].tolist()
    composite = composite_point(block, "last")
    tau, p = point_tau(models, block[P0].to_numpy(dtype=np.float64), composite)
    row.update(n_models=len(models), tau=tau, p=p, top3=_names(top_k(models, composite)))
    h53 = h53_table(block, {}, sets=("all",)).set_index("trajectory")
    for key, trajectory in (("p0", "P0"), ("whole", "P1 whole")):
        row[f"{key}_top3"] = h53.at[trajectory, "top3_trajectory"]
        row[f"h53_dropped_{key}"] = int(h53.at[trajectory, "dropped"])
    p0 = dict(zip(models, block[P0].to_numpy(dtype=np.float64)))
    ref = p0.get(reference, float("nan"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.array([p0[m] / ref for m in models], dtype=np.float64)
    e1 = pd.DataFrame({"dataset": dataset, "model": models, "family": block["family"].tolist(), "ratio": ratios,
                       "low": np.nan, "high": np.nan})
    h51 = h51_table(e1)
    h51 = h51[h51["group"] == "non-baseline"]
    if len(h51):
        h = h51.iloc[0]
        row.update(h51_n=int(h["n_models"]), h51_within10=int(h["n_within_10"]), h51_better30=int(h["n_better_30"]),
                   h51_worse30=int(h["n_worse_30"]), h51_beyond30=int(h["n_better_30"]) + int(h["n_worse_30"]),
                   h51_best=h["best_model"], h51_best_ratio=float(h["best_ratio"]))
    speed = an.e3_spacing_vs_speed(None, dataset, rankings=rankings.reset_index(drop=True), replicates=({}, {}))
    for horizon, tau_h in zip(speed["horizon"], speed["tau"]):
        row[f"h54_tau_{horizon}"] = float(tau_h)
    return row


def run_statistics_table(root, dataset: str, labels: Sequence[str], runs: Sequence[tuple[int, int]] = SEED_FOLD_RUNS,
                         extra: Sequence[tuple[str, int, int]] = (("block", 0, 0),), split: str = "driver") -> pd.DataFrame:
    """Statistics of the registered tests (H5.1, H5.3, H5.4) and tau(P0, composite) in every run of
    ``runs`` (``(fold, seed)`` of ``split``) and of ``extra`` (``(split, fold, seed)``); exploratory
    (decision HD-5), point values only. An entry counts in a run when it has every file of
    :data:`RUN_FILES`; the others are listed in ``missing`` (a run without any such entry gives NaN)."""
    rows = []
    for run_split, fold, seed in [*((split, k, s) for k, s in runs), *extra]:
        present = [m for m in labels
                   if all((an.run_dir(root, dataset, m, run_split, fold=fold, seed=seed) / f).exists() for f in RUN_FILES)]
        missing = [m for m in labels if m not in present]
        rankings = an.e3_rankings(root, dataset, present, fold=fold, seed=seed, split=run_split) if present else None
        rows.append(run_statistics_row(rankings, dataset, run_split, fold, seed, missing))
    return pd.DataFrame(rows, columns=RUN_STAT_COLUMNS).astype({c: "Int64" for c in RUN_COUNT_COLUMNS})


# ----------------------------------------------------------------------------- E3h: further checks (REV5-1)

#: the entries of the group "one-step objective" of ``within_family.OBJECTIVES`` (check (a) of REV5-1)
ONE_STEP = wf.OBJECTIVES["one-step objective"]
H51_ONE_STEP_COLUMNS = ["dataset", "n_models", "n_within_10", "n_ci_within_10", "n_better_30", "n_worse_30", "n_beyond_30",
                        "best_model", "best_ratio", "best_low", "best_high", "missing", "ratios"]
H53_VIEW_COLUMNS = ["dataset", "p2_view", "n_models", "missing_p2", "tau", "p", "top3_composite", "p0_top3", "h53_dropped_p0",
                    "whole_top3", "h53_dropped_whole"]
P2_COLUMNS = ("p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100")


def h51_one_step_table(e1: pd.DataFrame, labels: Sequence[str] = ONE_STEP) -> pd.DataFrame:
    """REV5-1 (a): the counts of :func:`h51_table` (group ``non-baseline``, the same thresholds) over the
    entries of ``labels`` only, per dataset of ``e1`` (the P0 ratios to persistence of ``e1_ratios``), with
    the ratio of every entry (``ratios``, ascending) and the labels without a row (``missing``)."""
    labels = list(labels)
    part = e1[e1["model"].isin(labels)]
    table = h51_table(part, labels)
    table = table[table["group"] == "non-baseline"].set_index("dataset")
    rows = []
    for dataset, block in part.groupby("dataset", sort=False):
        h = table.loc[dataset]
        ordered = block.assign(_r=block["ratio"].to_numpy(dtype=np.float64)).sort_values(["_r", "model"], na_position="last")
        rows.append({"dataset": dataset, "n_models": int(h["n_models"]), "n_within_10": int(h["n_within_10"]),
                     "n_ci_within_10": int(h["n_ci_within_10"]), "n_better_30": int(h["n_better_30"]),
                     "n_worse_30": int(h["n_worse_30"]), "n_beyond_30": int(h["n_better_30"]) + int(h["n_worse_30"]),
                     "best_model": h["best_model"], "best_ratio": float(h["best_ratio"]), "best_low": float(h["best_low"]),
                     "best_high": float(h["best_high"]),
                     "missing": _names([m for m in labels if m not in set(block["model"])]),
                     "ratios": ", ".join(f"{m} {r:.3f}" for m, r in zip(ordered["model"], ordered["_r"]))})
    return pd.DataFrame(rows, columns=H51_ONE_STEP_COLUMNS)


def h53_view_row(rankings: pd.DataFrame, dataset: str, view: str) -> dict:
    """REV5-1 (b), one row: over all entries of ``rankings`` (E3 table of one P2 view), tau(P0, composite
    ``last``) and the composite top three as :func:`seed_fold_table` (set ``all``), and the H5.3 counts as
    :func:`run_statistics_row` (:func:`h53_table` on the set ``all`` without replicates); ``missing_p2``:
    entries without one of the three P2 inputs in this view."""
    block = entry_set(rankings, "all")
    models = block["model"].tolist()
    composite = composite_point(block, "last")
    tau, p = point_tau(models, block[P0].to_numpy(dtype=np.float64), composite)
    p2 = block[list(P2_COLUMNS)].to_numpy(dtype=np.float64)
    missing = [m for m, ok in zip(models, np.isfinite(p2).all(axis=1)) if not ok]
    h53 = h53_table(block, {}, sets=("all",)).set_index("trajectory")
    return {"dataset": dataset, "p2_view": view, "n_models": len(models), "missing_p2": _names(missing), "tau": tau, "p": p,
            "top3_composite": _names(top_k(models, composite)),
            "p0_top3": h53.at["P0", "top3_trajectory"], "h53_dropped_p0": int(h53.at["P0", "dropped"]),
            "whole_top3": h53.at["P1 whole", "top3_trajectory"], "h53_dropped_whole": int(h53.at["P1 whole", "dropped"])}


def h53_views_table(root, dataset: str, labels: Sequence[str] | None, views: Sequence[str] = ("in_range", "all"),
                    fold: int = 0, seed: int = 0) -> pd.DataFrame:
    """REV5-1 (b): :func:`h53_view_row` for every P2 view of ``views`` (``an.e3_rankings(..., p2_view=view)``;
    ``in_range`` is the registered composite, ``all`` takes the P2 inputs over every leader profile)."""
    rows = [h53_view_row(an.e3_rankings(root, dataset, labels, fold=fold, seed=seed, p2_view=view), dataset, view)
            for view in views]
    return pd.DataFrame(rows, columns=H53_VIEW_COLUMNS)


def e3f_dataset(root, dataset: str, labels: Sequence[str], n_boot: int = 1000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Blocks B-F of E3f for one dataset (fold 0, seed 0): one call of
    :func:`an.e3_replicates` on the full rankings of ``labels``, shared by every treatment."""
    rankings = an.e3_rankings(root, dataset, labels)
    samples, _ = an.e3_replicates(root, dataset, rankings, n_boot, seed, columns=REPLICATE_COLUMNS)
    loo, loo_entries = leave_one_out(rankings)
    collision_input, collision_entries = collision_input_table(
        rankings, samples, collision_inputs(root, dataset, rankings["model"].tolist()), n_boot, seed)
    n_profiles = in_range_profiles(root, dataset, rankings["model"].tolist())
    out = {
        "missing_rule": missing_rule_table(rankings, samples),
        "tau_bias": tau_bias_table(rankings, samples),
        "collision_gate": collision_gate_table(rankings, samples, n_profiles),
        "leave_one_out": loo,
        "leave_one_out_entries": loo_entries,
        "input_ablation": input_ablation_table(rankings, samples),
        "collision_input": collision_input,
        "collision_input_entries": collision_entries,
        "stability_rule": stability_rule_table(rankings, samples, stability_inputs(root, dataset, rankings["model"].tolist()),
                                               n_profiles, n_boot, seed),
        "h53_rule": h53_table(rankings, samples),
    }
    if dataset in an.MAIN_DATASETS:
        out["seed_fold"] = seed_fold_table(root, dataset, labels)
    return out
