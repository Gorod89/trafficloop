"""E3h, the exploratory further checks (decision REV5-1), on synthetic inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trafficloop.experiments import analysis as an
from trafficloop.experiments import robustness as rb
from trafficloop.experiments import within_family as wf
from trafficloop.stats.ranks import _tau_b

# ----------------------------------------------------------------------------- (a) H5.1 on the one-step entries


def test_h51_one_step_thresholds():
    one_step = list(wf.OBJECTIVES["one-step objective"])
    assert one_step == list(rb.ONE_STEP) and len(one_step) == 10
    # boundaries: 0.9 and 1.1 are within 10 %, 0.7 and 1.3 are not beyond 30 %
    ratios = dict(zip(one_step, [0.9, 1.1, 1.0, 0.7, 0.69, 1.3, 1.31, 2.0, 0.5, 1.05]))
    rows = [{"dataset": "alpha", "model": m, "family": "network", "ratio": r, "low": r - 0.01, "high": r + 0.01}
            for m, r in ratios.items()]
    # entries outside the group never count, even with extreme ratios
    rows += [{"dataset": "alpha", "model": m, "family": f, "ratio": 0.1, "low": 0.05, "high": 0.15}
             for m, f in (("idm", "mechanistic"), ("ddpg", "network"), ("persistence", "baseline"))]
    rows += [{"dataset": "beta", "model": m, "family": "network", "ratio": 1.0, "low": 0.95, "high": 1.05} for m in one_step[:4]]
    table = rb.h51_one_step_table(pd.DataFrame(rows))
    assert table.columns.tolist() == rb.H51_ONE_STEP_COLUMNS
    alpha = table.set_index("dataset").loc["alpha"]
    assert alpha["n_models"] == 10
    assert alpha["n_within_10"] == 4  # 0.9, 1.1, 1.0, 1.05
    assert alpha["n_ci_within_10"] == 2  # [0.99, 1.01], [1.04, 1.06]
    assert alpha["n_better_30"] == 2 and alpha["n_worse_30"] == 2 and alpha["n_beyond_30"] == 4  # 0.69, 0.5; 1.31, 2.0
    assert alpha["best_model"] == one_step[8] and alpha["best_ratio"] == pytest.approx(0.5)
    assert alpha["missing"] == "" and alpha["ratios"].startswith(f"{one_step[8]} 0.500, {one_step[4]} 0.690")
    beta = table.set_index("dataset").loc["beta"]
    assert beta["n_models"] == 4 and beta["n_within_10"] == 4 and beta["missing"] == ", ".join(one_step[4:])
    # the counts are those of the registered rule restricted to the ten labels
    reference = rb.h51_table(pd.DataFrame(rows), one_step)
    reference = reference[reference["group"] == "non-baseline"].set_index("dataset")
    for c in ("n_models", "n_within_10", "n_ci_within_10", "n_better_30", "n_worse_30", "best_model"):
        assert table.set_index("dataset")[c].tolist() == reference[c].tolist()


# ----------------------------------------------------------------------------- (b) the P2 view


#: (P0, whole-event error, in-range P2 inputs, all-profile P2 inputs, unstable fraction) per entry
ENTRIES = {
    "a": (0.10, 1.0, (0.1, 0.1, 0), (0.9, 0.9, 5), 0.0),
    "b": (0.20, 2.0, (0.2, 0.2, 0), (0.8, 0.8, 4), 0.0),
    "c": (0.30, 3.0, (0.3, 0.3, 0), (0.7, 0.7, 3), 0.0),
    "d": (0.40, 4.0, (0.9, 0.9, 5), (0.2, 0.2, 0), 0.0),
    "e": (0.50, 5.0, (0.8, 0.8, 4), (0.1, 0.1, 0), 0.0),
    "f": (0.60, 6.0, (0.7, 0.7, 3), (0.15, 0.3, 0), 0.0),
}
KEYS = ("concave_growth_error", "hysteresis_error", "collisions_100")


@pytest.fixture
def runs(tmp_path):
    for label, (p0, whole, inside, every, unstable) in ENTRIES.items():
        d = an.run_dir(tmp_path, "alpha", label)
        d.mkdir(parents=True)
        train = {"family": "network", "test": {"p0": {"rmse_a": p0}, "closed_loop_whole_event": {"rmse_s": whole, "rmse_v": whole}}}
        p2 = {**{f"{k}_in_range": v for k, v in zip(KEYS, inside)}, **dict(zip(KEYS, every))}
        if label == "f":
            p2.pop("hysteresis_error")  # one entry without an all-profile input
        (d / "train.json").write_text(json.dumps(train))
        (d / "p2.json").write_text(json.dumps({"summary": p2}))
        (d / "p3.json").write_text(json.dumps({"summary": {"unstable_fraction": unstable}}))
    return tmp_path


def test_p2_view_switch(runs):
    inside = an.e3_rankings(runs, "alpha", list(ENTRIES))
    assert inside.equals(an.e3_rankings(runs, "alpha", list(ENTRIES), p2_view="in_range"))
    every = an.e3_rankings(runs, "alpha", list(ENTRIES), p2_view="all")
    for i, column in enumerate(rb.P2_COLUMNS):
        assert inside[column].tolist() == [v[2][i] for v in ENTRIES.values()]
        expected = [v[3][i] for v in ENTRIES.values()]
        if column == "p2_hysteresis_error":
            expected[-1] = np.nan
        np.testing.assert_array_equal(every[column].to_numpy(dtype=float), np.array(expected, dtype=float))
    # nothing else depends on the view
    other = [c for c in an.E3_COLUMNS if c not in (*rb.P2_COLUMNS, "composite")]
    pd.testing.assert_frame_equal(inside[other], every[other])
    with pytest.raises(KeyError):
        an.e3_rankings(runs, "alpha", list(ENTRIES), p2_view="pulse")


def test_h53_views_table(runs):
    table = rb.h53_views_table(runs, "alpha", list(ENTRIES))
    assert table.columns.tolist() == rb.H53_VIEW_COLUMNS
    by = table.set_index("p2_view")
    assert by.at["in_range", "top3_composite"] == "a, b, c" and by.at["in_range", "h53_dropped_p0"] == 0
    assert by.at["in_range", "missing_p2"] == "" and by.at["in_range", "tau"] > 0
    assert by.at["all", "top3_composite"] == "e, d, f" and by.at["all", "h53_dropped_p0"] == 3
    assert by.at["all", "h53_dropped_whole"] == 3 and by.at["all", "missing_p2"] == "f" and by.at["all", "tau"] < 0
    assert by.at["all", "p0_top3"] == "a, b, c" and by.at["all", "whole_top3"] == "a, b, c"
    # the row of the in-range view is the run statistics of the same rankings
    row = rb.run_statistics_row(an.e3_rankings(runs, "alpha", list(ENTRIES)), "alpha", "driver", 0, 0)
    assert by.at["in_range", "tau"] == pytest.approx(row["tau"]) and by.at["in_range", "top3_composite"] == row["top3"]
    assert by.at["in_range", "h53_dropped_whole"] == row["h53_dropped_whole"]


# ----------------------------------------------------------------------------- (c) contrast between groups


def _contrast_frame(n: int = 12, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = [f"m{i:02d}" for i in range(n)]
    composite = rng.random(n)
    composite[2] = np.nan  # a missing composite ranks last
    return pd.DataFrame({"dataset": "alpha", "model": models, "p0_rmse_a": rng.random(n), "composite": composite})


GROUPS = {"g1": ("m00", "m01", "m02", "m03"), "g2": ("m04", "m05", "m06", "m07", "m08"), "g3": ("m09", "m10")}


def test_group_contrast_is_deterministic_under_the_seed():
    frame = _contrast_frame()
    first = wf.group_contrast_table(frame, n_perm=2000, seed=0, groups=GROUPS)
    assert first.columns.tolist() == wf.CONTRAST_COLUMNS
    pd.testing.assert_frame_equal(first, wf.group_contrast_table(frame, n_perm=2000, seed=0, groups=GROUPS))
    other = wf.group_contrast_table(frame, n_perm=2000, seed=1, groups=GROUPS)
    assert other["tau_perm_mean"].iloc[0] != first["tau_perm_mean"].iloc[0]
    row = first.iloc[0]
    # m11 belongs to no group: a group of its own
    assert row["groups"] == "g1 4; g2 5; g3 2; unassigned: m11 1" and row["n_perm"] == 2000
    tau, p = rb.point_tau(frame["model"].tolist(), frame["p0_rmse_a"].to_numpy(), frame["composite"].to_numpy())
    assert row["tau_observed"] == pytest.approx(tau) and row["p_observed"] == pytest.approx(p)


def test_one_group_is_a_plain_permutation_test():
    frame = _contrast_frame()
    n_perm = 3000
    row = wf.group_contrast_table(frame, n_perm=n_perm, seed=0, groups={"all": tuple(frame["model"])}).iloc[0]
    rng = np.random.default_rng(0)
    composite = np.where(np.isnan(frame["composite"]), np.inf, frame["composite"])
    permuted = rng.permuted(np.tile(composite, (n_perm, 1)), axis=1)
    taus = _tau_b(np.tile(frame["p0_rmse_a"].to_numpy(), (n_perm, 1)), permuted)
    assert row["tau_perm_mean"] == pytest.approx(taus.mean())
    assert row["tau_perm_low"] == pytest.approx(np.quantile(taus, 0.025))
    assert row["tau_perm_high"] == pytest.approx(np.quantile(taus, 0.975))
    k = int(np.sum(taus <= row["tau_observed"] + 1e-12))
    assert row["share_at_or_below"] == pytest.approx((k + 1) / (n_perm + 1))
    assert row["ratio_mean_observed"] == pytest.approx(taus.mean() / row["tau_observed"])
    # a plain permutation test centres near zero
    assert abs(row["tau_perm_mean"]) < 0.02


def test_groups_of_one_leave_tau_unchanged():
    frame = _contrast_frame()
    singletons = {m: (m,) for m in frame["model"]}
    row = wf.group_contrast_table(frame, n_perm=500, seed=0, groups=singletons).iloc[0]
    assert row["tau_perm_mean"] == pytest.approx(row["tau_observed"])
    assert row["tau_perm_low"] == pytest.approx(row["tau_observed"]) and row["tau_perm_high"] == pytest.approx(row["tau_observed"])
    assert row["share_at_or_below"] == 1.0 and row["ratio_mean_observed"] == pytest.approx(1.0)


def test_group_contrast_per_dataset():
    frame = pd.concat([_contrast_frame(), _contrast_frame(seed=4).assign(dataset="beta")], ignore_index=True)
    table = wf.group_contrast_table(frame, n_perm=200, seed=0, groups=GROUPS)
    assert table["dataset"].tolist() == ["alpha", "beta"]
    single = wf.group_contrast_table(frame[frame["dataset"] == "beta"], n_perm=200, seed=0, groups=GROUPS)
    pd.testing.assert_frame_equal(table.iloc[[1]].reset_index(drop=True), single)
