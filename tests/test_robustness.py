"""E3f robustness checks (trafficloop/experiments/robustness.py) on fabricated E3 frames and replicates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trafficloop.experiments import analysis as an
from trafficloop.experiments import robustness as rb

INPUT_COLUMNS = list(an.COMPOSITE_COLUMNS.values())


@pytest.fixture
def frames() -> tuple[pd.DataFrame, dict]:
    """Eight entries (two baselines without the stability input, one network without it) and 200
    replicates that perturb every value; the composite replicates are those of ``an._composite``."""
    rng = np.random.default_rng(3)
    models = ["persistence", "constant_speed", "idm", "gipps", "mlp", "lstm", "gru", "ddpg"]
    family = ["baseline", "baseline", "mechanistic", "mechanistic", "network", "network", "network", "network"]
    frame = pd.DataFrame({"dataset": "alpha", "model": models, "family": family,
                          "p0_rmse_a": [1.0, 1.4, 0.9, 0.8, 0.3, 0.25, 0.35, 0.5],
                          "p1_whole_rmse_s": [9.0, 12.0, 3.0, 2.5, 4.0, 5.0, 3.5, 6.0],
                          "p2_concave_growth_error": [0.9, 0.95, 0.2, 0.3, 0.6, 0.7, 0.5, 0.8],
                          "p2_hysteresis_error": [80.0, 90.0, 20.0, 25.0, 50.0, 60.0, 55.0, 70.0],
                          "p3_unstable_fraction": [np.nan, np.nan, 0.0, 0.0, 0.4, 0.6, np.nan, 0.5],
                          "p2_collisions_100": [0.0, 0.0, 0.0, 0.0, 5.0, 30.0, 0.0, 12.0]})
    frame["composite"] = an._composite(frame)
    n_boot = 200
    samples: dict[str, dict[str, np.ndarray]] = {}
    for column in ["p0_rmse_a", "p1_whole_rmse_s", *INPUT_COLUMNS]:
        values = frame[column].to_numpy(dtype=np.float64)
        noise = rng.normal(1.0, 0.15, size=(n_boot, len(models)))
        matrix = values[None, :] * noise
        if column == "p2_collisions_100":
            matrix = np.round(np.clip(matrix + rng.integers(-1, 2, size=matrix.shape), 0, None))
        samples[column] = {m: matrix[:, i] for i, m in enumerate(models)}
    composite = np.empty((n_boot, len(models)))
    for r in range(n_boot):
        block = pd.DataFrame({"model": models, **{c: [samples[c][m][r] for m in models] for c in INPUT_COLUMNS}})
        composite[r] = an._composite(block)
    samples["composite"] = {m: composite[:, i] for i, m in enumerate(models)}
    return frame, samples


def test_skip_composite_by_hand():
    values = {"concave_growth_error": np.array([[1.0, 2.0, 3.0]]), "hysteresis_error": np.array([[3.0, 2.0, 1.0]]),
              "unstable_fraction": np.array([[np.nan, 0.0, 1.0]]), "collisions_100": np.array([[np.nan, np.nan, 4.0]])}
    out = rb.skip_matrix(values)[0]
    # model 0: (0 + 1) / 2; model 1: (0.5 + 0.5 + 0) / 3; model 2: (1 + 0 + 1 + 0) / 4
    assert out == pytest.approx([0.5, 1.0 / 3.0, 0.5])
    none = rb.skip_matrix({k: np.full((1, 2), np.nan) for k in values})[0]
    assert np.isnan(none).all()


def test_last_rule_matches_the_published_composite(frames):
    frame, samples = frames
    assert np.array_equal(rb.composite_point(frame, "last"), frame["composite"].to_numpy())
    replicates = rb.composite_replicates(samples, frame["model"].tolist(), "last")
    expected = np.column_stack([samples["composite"][m] for m in frame["model"]])
    assert np.array_equal(replicates, expected)
    complete = rb.entry_set(frame, "complete")
    assert np.allclose(rb.composite_point(complete, "skip"), rb.composite_point(complete, "last"))


def test_sets(frames):
    frame, _ = frames
    assert len(rb.entry_set(frame, "all")) == 8
    assert rb.entry_set(frame, "no baselines")["model"].tolist() == ["idm", "gipps", "mlp", "lstm", "gru", "ddpg"]
    assert rb.entry_set(frame, "complete")["model"].tolist() == ["idm", "gipps", "mlp", "lstm", "ddpg"]
    with pytest.raises(KeyError):
        rb.entry_set(frame, "some")


def test_published_treatment_reproduces_the_e3_tau(frames):
    frame, samples = frames
    table = rb.missing_rule_table(frame, samples)
    published = table.set_index("treatment").loc["published"]
    reference = an._tau_row(frame, samples, {}, "p0_rmse_a", "composite")
    for key in ("tau", "low", "high", "p"):
        assert published[key] == pytest.approx(reference[key], abs=1e-12)
    assert published["tau_boot_mean"] == pytest.approx(reference["tau_boot"], abs=1e-12)
    assert published["n_missing_stability"] == 3 and published["missing_stability"] == "persistence, constant_speed, gru"
    assert table.set_index("treatment").loc["no baselines, last", "n_missing_stability"] == 1
    bias = rb.tau_bias_table(frame, samples).set_index("ranking_b")
    assert bias.loc["composite"].iloc[0]["tau"] == pytest.approx(published["tau"])
    check = rb.reproduction_check(table, pd.DataFrame([{"dataset": "alpha", "ranking_a": "P0", "ranking_b": "composite",
                                                       "tau": reference["tau"], "low": reference["low"], "high": reference["high"]}]))
    assert check[["tau", "low", "high"]].to_numpy().max() < 1e-12


def test_tau_statistics():
    taus = np.linspace(-1.0, 1.0, 2000)
    out = rb.tau_statistics(0.0, taus)
    assert out["share_below"] == 0.5
    assert out["bc_low"] == pytest.approx(out["low"], abs=1e-3) and out["bc_high"] == pytest.approx(out["high"], abs=1e-3)
    edge = rb.tau_statistics(-1.0, taus)  # nothing below: share clipped to 1 / (B + 1)
    assert edge["share_below"] == 0.0 and np.isfinite(edge["bc_low"]) and edge["bc_high"] < out["high"]
    assert np.isnan(rb.tau_statistics(0.0, None)["low"])


def test_collision_gate(frames):
    frame, samples = frames
    gate = rb.collision_gate_table(frame, samples, n_profiles=10)
    zero = gate[(gate["set"] == "all") & (gate["theta"] == 0.0)].iloc[0]
    assert zero["n_followers"] == 1000 and zero["n_admissible"] == 5
    assert zero["admissible"] == "persistence, constant_speed, idm, gipps, gru"
    first = set(zero["top3"].split(", "))
    assert first <= {"persistence", "constant_speed", "idm", "gipps", "gru"}
    loose = gate[(gate["set"] == "all") & (gate["theta"] == 0.01)].iloc[0]
    assert loose["n_admissible"] == 6  # 5 of 1000 followers pass, 12 and 30 do not
    assert rb.admissible(np.array([np.nan, 10.0, 11.0]), 1000, 0.01).tolist() == [False, True, False]
    assert np.isfinite(zero["low"]) and zero["n_boot"] == 200


def test_leave_one_out(frames):
    frame, _ = frames
    summary, entries = rb.leave_one_out(frame)
    assert len(entries) == 8 and (entries["n_models"] == 7).all()
    row = entries.set_index("left_out").loc["idm"]
    block = frame[frame["model"] != "idm"]
    tau, _ = rb.point_tau(block["model"].tolist(), block["p0_rmse_a"].to_numpy(), an._composite(block.reset_index(drop=True)))
    assert row["tau"] == pytest.approx(tau)
    assert summary.iloc[0]["loo_min"] == entries["tau"].min() and summary.iloc[0]["loo_max"] == entries["tau"].max()
    assert summary.iloc[0]["tau_all"] == pytest.approx(rb.point_tau(frame["model"].tolist(), frame["p0_rmse_a"].to_numpy(),
                                                                     frame["composite"].to_numpy())[0])


def test_top_k_breaks_ties_by_name():
    assert rb.top_k(["c", "b", "a", "d"], np.array([1.0, 1.0, np.nan, 0.0])) == ["d", "b", "c"]
    reps = rb.replicate_top_k(["c", "b", "a", "d"], np.array([[1.0, 1.0, np.nan, 0.0], [0.0, 0.0, 0.0, 0.0]]))
    assert reps == [{"d", "b", "c"}, {"a", "b", "c"}]


def test_h53_rule(frames):
    frame, samples = frames
    same = frame.assign(p0_rmse_a=frame["composite"], p1_whole_rmse_s=-frame["composite"])
    copy = {**samples, "p0_rmse_a": samples["composite"], "p1_whole_rmse_s": {m: -v for m, v in samples["composite"].items()}}
    table = rb.h53_table(same, copy).set_index(["set", "trajectory"])
    row = table.loc[("all", "P0")]
    assert row["dropped"] == 0 and row["p_boot_coincide"] == 1.0 and row["p_boot_ge2"] == 0.0
    assert table.loc[("all", "P1 whole"), "dropped"] == 3
    real = rb.h53_table(frame, samples)
    assert set(real["dropped"]) <= {0, 1, 2, 3} and (real["n_boot"] == 200).all()


def test_h51_rule():
    e1 = pd.DataFrame({"dataset": "alpha", "model": ["persistence", "constant_speed", "idm", "mlp", "lstm", "gru"],
                       "family": ["baseline", "baseline", "mechanistic", "network", "network", "network"],
                       "ratio": [1.0, 3.0, 1.5, 0.95, 0.6, 1.05], "low": [1.0, 2.9, 1.4, 0.92, 0.55, 0.85],
                       "high": [1.0, 3.1, 1.6, 0.99, 0.65, 1.2]})
    table = rb.h51_table(e1).set_index("group")
    row = table.loc["non-baseline"]
    assert (row["n_models"], row["n_within_10"], row["n_ci_within_10"], row["n_better_30"], row["n_better_ci"],
            row["n_worse_30"]) == (4, 2, 1, 1, 2, 1)
    assert row["best_model"] == "lstm" and row["best_ratio"] == 0.6
    assert table.loc["learned", "n_models"] == 3 and table.loc["learned", "n_worse_30"] == 0


def test_input_ablation(frames):
    frame, samples = frames
    table = rb.input_ablation_table(frame, samples)
    assert table.columns.tolist() == rb.INPUT_COLUMNS
    assert len(table) == 2 * 5 and set(table["left_out"]) == {"none", *an.COMPOSITE_COLUMNS}
    published = rb.missing_rule_table(frame, samples).set_index("treatment")
    for name, treatment in (("all", "published"), ("no baselines", "no baselines, last")):
        none = table[(table["set"] == name) & (table["left_out"] == "none")].iloc[0]
        for column in ("tau", "low", "high", "tau_boot_median"):
            assert none[column] == pytest.approx(published.loc[treatment, column], abs=1e-12)
        assert none["top3"] == published.loc[treatment, "top3"]
    # without the collision input the composite is the mean normalised rank of the three others (a missing value last)
    row = table[(table["set"] == "all") & (table["left_out"] == "collisions_100")].iloc[0]
    assert row["inputs"] == "concave_growth_error, hysteresis_error, unstable_fraction" and row["n_boot"] == 200
    kept = ["p2_concave_growth_error", "p2_hysteresis_error", "p3_unstable_fraction"]
    ranks = frame[kept].rank(method="average", na_option="bottom")
    by_hand = ((ranks - 1.0) / (len(frame) - 1)).mean(axis=1).to_numpy()
    tau, _ = rb.point_tau(frame["model"].tolist(), frame["p0_rmse_a"].to_numpy(), by_hand)
    assert row["tau"] == pytest.approx(tau, abs=1e-12)
    assert row["low"] <= row["tau_boot_median"] <= row["high"]


def test_collision_measures_by_hand():
    vehicles = pd.DataFrame({"profile": ["a"] * 6 + ["b"] * 6, "index": list(range(6)) * 2,
                             "collided": [False, False, True, False, True, False, False, True, False, False, False, False]})
    out = rb.collision_measures(vehicles, ["a", "b"]).set_index("profile")
    assert out.columns.tolist() == list(rb.COLLISION_INPUTS) == ["followers", "blocked", "profiles"]
    # a: followers 2 and 4 collide, so the followers 2 to 5 are at or behind the first collision
    assert out.loc["a"].tolist() == [2.0, 4.0, 1.0]
    # b: only the first follower collides (a platoon that does not react): one follower, the whole platoon blocked
    assert out.loc["b"].tolist() == [1.0, 5.0, 1.0]
    assert rb.collision_measures(vehicles.assign(collided=False), ["a"]).iloc[0, 1:].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(KeyError):
        rb.collision_measures(vehicles, ["c"])


def _measures(frame: pd.DataFrame, n_profiles: int = 4, followers: int = 40) -> dict[str, pd.DataFrame]:
    """Per-profile collision measures whose ``followers`` sum to the collision input of every entry:
    all collisions in the first profile; the two baselines collide with one follower in every profile."""
    out = {}
    for model, family, total in zip(frame["model"], frame["family"], frame["p2_collisions_100"]):
        count = np.zeros(n_profiles)
        count[0] = total
        if family == "baseline":
            count = np.full(n_profiles, 1.0)
        blocked = np.where(count > 0, followers, 0.0) if family == "baseline" else np.minimum(count * 2, followers)
        out[model] = pd.DataFrame({"profile": [f"p{i}" for i in range(n_profiles)], "followers": count, "blocked": blocked,
                                   "profiles": (count > 0).astype(float)})
    return out


def test_collision_input_table(frames):
    from trafficloop.protocols.composite import _aligned, _sum_over
    from trafficloop.stats.bootstrap import draw_counts

    frame, samples = frames
    frame = frame.copy()
    frame.loc[frame["family"] == "baseline", "p2_collisions_100"] = 4.0  # one follower in each of four profiles
    frame["composite"] = an._composite(frame)
    measures = _measures(frame)
    models = frame["model"].tolist()
    # without the replicates of the same profile draw: point values only
    table, entries = rb.collision_input_table(frame, samples, measures, n_boot=200, seed=0)
    assert table.columns.tolist() == rb.COLLISION_INPUT_COLUMNS and entries.columns.tolist() == rb.COLLISION_ENTRY_COLUMNS
    assert len(table) == 2 * 3 and (table["note"] == "no replicates").all() and (table["n_boot"] == 0).all()
    published, _ = rb.point_tau(models, frame["p0_rmse_a"].to_numpy(), frame["composite"].to_numpy())
    row = table[(table["set"] == "all") & (table["input"] == "followers")].iloc[0]
    assert row["tau"] == pytest.approx(published, abs=1e-12)
    assert np.array_equal(entries["followers"].to_numpy(), frame["p2_collisions_100"].to_numpy())
    assert np.allclose(entries["composite_followers"].to_numpy(), frame["composite"].to_numpy())
    # the baselines: 4 colliding followers, but every profile has a collision and every follower is blocked
    base = entries[entries["family"] == "baseline"]
    assert (base["blocked"] == 160.0).all() and (base["profiles"] == 4.0).all()
    assert (base["rank_blocked"] >= base["rank_followers"]).all()
    # with the collision replicates of the same draw the three rows have intervals
    aligned = _aligned({m: measures[m] for m in models}, "profile", rb.COLLISION_INPUTS, "test")
    counts = draw_counts(np.random.default_rng([0, 1]), 4, 200).astype(np.float64)
    drawn = _sum_over(counts, aligned["followers"])
    samples = {**samples, "p2_collisions_100": {m: drawn[:, i] for i, m in enumerate(models)}}
    table, _ = rb.collision_input_table(frame, samples, measures, n_boot=200, seed=0)
    assert (table["note"] == "").all() and (table["n_boot"] == 200).all()
    assert (table["low"] <= table["high"]).all()
    same = rb.input_ablation_table(frame, samples)
    none = same[(same["set"] == "all") & (same["left_out"] == "none")].iloc[0]
    row = table[(table["set"] == "all") & (table["input"] == "followers")].iloc[0]
    for column in ("tau", "low", "high", "tau_boot_median"):
        assert row[column] == pytest.approx(none[column], abs=1e-12)


def _speeds(frame: pd.DataFrame, n_speeds: int = 6) -> dict[str, pd.DataFrame]:
    """Per-speed stability tables whose published verdicts average to the unstable fraction of every entry
    (multiples of 1 / n_speeds); ``gipps`` passes a 0.1 Hz perturbation on unchanged (neutral)."""
    out = {}
    for model, fraction in zip(frame["model"], frame["p3_unstable_fraction"]):
        if not np.isfinite(fraction):
            continue
        k = int(round(fraction * n_speeds))
        unstable = np.array([True] * k + [False] * (n_speeds - k))
        gain = np.where(unstable, 1.5, 0.6)
        if model == "gipps":
            gain = np.full(n_speeds, 0.9999)
        out[model] = pd.DataFrame({"v_e": np.arange(5.0, 5.0 + n_speeds), "converged": True, "tracks": True,
                                   "unstable_numerical": unstable, "unstable_1e3": unstable, "gain_0.1": gain})
    return out


def test_stability_measures_by_hand():
    speeds = pd.DataFrame({"v_e": [5.0, 6.0, 7.0, 8.0], "converged": [True, True, True, False], "tracks": [True, True, True, False],
                           "unstable_numerical": [True, False, False, None], "unstable_1e3": [False, False, False, None],
                           "gain_0.1": [1.2, 0.9995, 0.7, np.nan]})
    out = rb.stability_measures(speeds)
    assert out.columns.tolist() == ["v_e", *rb.STABILITY_RULES]
    assert out["published"].tolist()[:3] == [1.0, 0.0, 0.0] and np.isnan(out["published"].iloc[3])
    assert out["tolerance 1e-3"].tolist()[:3] == [0.0, 0.0, 0.0]
    # the second speed passes the perturbation on with a gain of 0.9995: stable as published, unstable when neutral counts
    assert out["neutral counts as unstable"].tolist()[:3] == [1.0, 1.0, 0.0] and np.isnan(out["neutral counts as unstable"].iloc[3])


def test_stability_rule_table(frames):
    from trafficloop.protocols.composite import _aligned, _mean_over
    from trafficloop.stats.bootstrap import draw_counts

    frame, samples = frames
    frame = frame.copy()
    frame["p3_unstable_fraction"] = [np.nan, np.nan, 0.0, 0.0, 0.5, 2.0 / 3.0, np.nan, 0.5]
    frame["composite"] = an._composite(frame)
    measures = {model: rb.stability_measures(speeds) for model, speeds in _speeds(frame).items()}
    models = frame["model"].tolist()
    table = rb.stability_rule_table(frame, samples, measures, n_profiles=4, n_boot=200, seed=0)
    assert table.columns.tolist() == rb.STABILITY_RULE_COLUMNS and len(table) == 2 * 3
    assert (table["note"] == "no replicates").all()  # the replicates of the fixture are not those of this speed draw
    by = table[table["set"] == "all"].set_index("rule")
    published, _ = rb.point_tau(models, frame["p0_rmse_a"].to_numpy(), frame["composite"].to_numpy())
    assert by.loc["published", "tau"] == pytest.approx(published, abs=1e-12) and by.loc["published", "n_changed"] == 0
    assert by.loc["tolerance 1e-3", "n_changed"] == 0
    assert by.loc["neutral counts as unstable", "changed"] == "gipps" and by.loc["neutral counts as unstable", "n_changed"] == 1
    # with the replicates of the same draw (profile counts first, then speed counts) the rows have intervals
    have = [m for m in models if m in measures]
    aligned = _aligned({m: measures[m] for m in have}, "v_e", rb.STABILITY_RULES, "test")
    rng = np.random.default_rng([0, 1])
    draw_counts(rng, 4, 200)
    drawn = _mean_over(draw_counts(rng, 6, 200).astype(np.float64), aligned["published"])
    column = {m: drawn[:, have.index(m)] if m in have else np.full(200, np.nan) for m in models}
    table = rb.stability_rule_table(frame, {**samples, "p3_unstable_fraction": column}, measures, n_profiles=4, n_boot=200, seed=0)
    assert (table["note"] == "").all() and (table["n_boot"] == 200).all() and (table["low"] <= table["high"]).all()


# ----------------------------------------------------------------------------- run statistics (HD-5)


def _run_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The fixture with P1 spacing and speed errors: speed = spacing at 10 s and over the whole event,
    the reverse order at 30 s."""
    spacing = np.arange(1.0, len(frame) + 1.0)
    return frame.assign(p1_h10_rmse_s=spacing, p1_h10_rmse_v=2 * spacing, p1_h30_rmse_s=spacing, p1_h30_rmse_v=-spacing,
                        p1_whole_rmse_v=frame["p1_whole_rmse_s"])


def test_run_statistics_row(frames):
    frame, _ = frames
    frame = _run_frame(frame)
    row = rb.run_statistics_row(frame, "alpha", "driver", 1, 0)
    assert list(row) == rb.RUN_STAT_COLUMNS
    tau, p = rb.point_tau(frame["model"].tolist(), frame["p0_rmse_a"].to_numpy(), frame["composite"].to_numpy())
    assert row["tau"] == pytest.approx(tau) and row["p"] == pytest.approx(p) and row["n_models"] == 8 and row["missing"] == ""
    assert row["top3"] == ", ".join(rb.top_k(frame["model"].tolist(), frame["composite"].to_numpy()))
    # H5.1 by hand: P0 ratios to persistence (1.0) of the six entries that are not baselines:
    # idm 0.9 (within 10 %), gipps 0.8, mlp 0.3, lstm 0.25, gru 0.35, ddpg 0.5 (four more than 30 % better)
    assert (row["h51_n"], row["h51_within10"], row["h51_better30"], row["h51_worse30"], row["h51_beyond30"]) == (6, 1, 4, 0, 4)
    assert row["h51_best"] == "lstm" and row["h51_best_ratio"] == pytest.approx(0.25)
    assert (row["h54_tau_h10"], row["h54_tau_h30"], row["h54_tau_whole"]) == pytest.approx((1.0, -1.0, 1.0))
    # H5.3: a trajectory error equal to the composite keeps the top three, its reverse drops all three
    same = frame.assign(p0_rmse_a=frame["composite"], p1_whole_rmse_s=-frame["composite"])
    row = rb.run_statistics_row(same, "alpha", "driver", 1, 0)
    assert (row["h53_dropped_p0"], row["h53_dropped_whole"]) == (0, 3)
    assert row["p0_top3"] == row["top3"] and not set(row["whole_top3"].split(", ")) & set(row["top3"].split(", "))
    # without persistence the ratios are missing: nothing within or beyond, no best entry
    row = rb.run_statistics_row(frame[frame["model"] != "persistence"], "alpha", "driver", 1, 0)
    assert (row["h51_n"], row["h51_within10"], row["h51_beyond30"], row["h51_best"]) == (6, 0, 0, "")
    assert np.isnan(row["h51_best_ratio"])


def test_run_statistics_table(frames, tmp_path, monkeypatch):
    frame, _ = frames
    frame = _run_frame(frame)
    labels = frame["model"].tolist()
    complete = {(0, 0), (0, 1)}
    for fold, seed in rb.SEED_FOLD_RUNS:
        for label in labels:
            directory = an.run_dir(tmp_path, "alpha", label, fold=fold, seed=seed)
            directory.mkdir(parents=True)
            # complete runs have every file; in fold 1 only train.json and p1.json (a run still being evaluated)
            for name in rb.RUN_FILES if (fold, seed) in complete else (("train.json", "p1.json") if fold == 1 else ()):
                (directory / name).write_text("{}")
    calls = []

    def rankings(root, dataset, present, fold=0, seed=0, split="driver"):
        calls.append((split, fold, seed, tuple(present)))
        return frame[frame["model"].isin(present)].reset_index(drop=True)

    monkeypatch.setattr(an, "e3_rankings", rankings)
    table = rb.run_statistics_table(tmp_path, "alpha", labels)
    assert table.columns.tolist() == rb.RUN_STAT_COLUMNS
    assert list(zip(table["split"], table["fold"], table["seed"])) == [("driver", k, s) for k, s in rb.SEED_FOLD_RUNS] + [("block", 0, 0)]
    assert [c[:3] for c in calls] == [("driver", 0, 0), ("driver", 0, 1)]
    by = table.set_index(["split", "fold", "seed"])
    first = by.loc[("driver", 0, 0)]
    assert first["n_models"] == 8 and first["missing"] == "" and first["h51_beyond30"] == 4
    for key in (("driver", 1, 0), ("driver", 4, 0), ("block", 0, 0)):
        row = by.loc[key]
        assert row["n_models"] == 0 and row["missing"] == ", ".join(labels)
        assert np.isnan(row["tau"]) and pd.isna(row["h53_dropped_p0"]) and pd.isna(row["h51_n"]) and row["top3"] == ""
    assert rb.run_statistics_table(tmp_path, "alpha", labels, runs=((0, 0),), extra=()).shape[0] == 1
