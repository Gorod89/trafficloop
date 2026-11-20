"""E3e and E3g (trafficloop/experiments/within_family.py) on a fabricated E3 frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trafficloop.experiments import within_family as wf
from trafficloop.utils import REPO_ROOT


def _reference_labels() -> tuple[str, ...]:
    """The labels of the main campaign (``MODELS`` of scripts/m2.py), read from the source."""
    import ast

    tree = ast.parse((REPO_ROOT / "scripts" / "m2.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("scripts/m2.py defines no MODELS")


REFERENCE_LABELS = _reference_labels()


def _frame() -> pd.DataFrame:
    """Every reference entry on one dataset: the calibrated entries worst at one step and best on the
    composite, the entries with a one-step objective the other way round."""
    rng = np.random.default_rng(1)
    models = list(REFERENCE_LABELS)
    objective = [wf.objective_of(m) for m in models]
    fitted = np.array([o in ("one-step objective", "rollout or multi-step loss") for o in objective])
    p0 = np.where(fitted, 0.1, 1.0) + rng.uniform(0, 0.05, len(models))
    composite = np.where(fitted, 0.8, 0.2) + rng.uniform(0, 0.05, len(models))
    family = ["baseline" if o == "none" else "mechanistic" if o == "closed-loop calibration" else "network" for o in objective]
    return pd.DataFrame({"dataset": "alpha", "model": models, "family": family, "p0_rmse_a": p0, "composite": composite,
                         "p3_unstable_fraction": composite, "p1_whole_rmse_s": composite})


def test_every_reference_entry_has_an_objective():
    assigned = [m for members in wf.OBJECTIVES.values() for m in members]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(REFERENCE_LABELS)
    assert wf.objective_of("mlp") == "one-step objective" and wf.objective_of("no_such_entry") == wf.UNASSIGNED


def test_objective_tables():
    frame = _frame()
    ranks, taus = wf.objective_tables(frame)
    assert ranks.columns.tolist() == wf.OBJECTIVE_RANK_COLUMNS and taus.columns.tolist() == wf.OBJECTIVE_TAU_COLUMNS
    by = ranks.set_index("objective")
    assert by["n_models"].sum() == len(frame) and by.loc["one-step objective", "n_models"] == 10
    # 14 entries fitted to a one-step or rollout loss take the first 14 places at one step and the last 14 on the composite
    assert by.loc["one-step objective", "p0_rank_max"] <= 14 and by.loc["closed-loop calibration", "p0_rank_min"] >= 15
    assert by.loc["one-step objective", "composite_rank_min"] >= 12 and by.loc["closed-loop calibration", "composite_rank_max"] <= 11
    inside = taus.set_index("set")
    assert inside.loc["one-step objective", "n_models"] == 10 and inside.loc["one-step objective, rollout or multi-step loss", "n_models"] == 14
    keep = frame[[wf.objective_of(m) == "one-step objective" for m in frame["model"]]]
    from scipy import stats
    assert inside.loc["one-step objective", "tau"] == pytest.approx(stats.kendalltau(keep["p0_rmse_a"], keep["composite"])[0], abs=1e-12)
    over_all = wf.within_family_taus(frame)
    overall = over_all[(over_all["group"] == "all") & (over_all["against"] == "composite")].iloc[0]
    assert overall["tau"] < -0.3  # the correlation over all entries is the contrast between the objectives


def test_holm_columns():
    frame = pd.concat([_frame().assign(dataset=name) for name in ("alpha", "beta", "gamma")], ignore_index=True)
    family = wf.within_family_taus(frame)
    assert family.columns.tolist() == wf.COLUMNS
    _, taus = wf.objective_tables(frame)
    assert taus.columns.tolist() == wf.OBJECTIVE_TAU_COLUMNS
    for table, keys in ((family, ["group", "against"]), (taus, ["set"])):
        for _, block in table.groupby(keys):
            p, adjusted = block["p"].to_numpy(), block["p_holm"].to_numpy()
            assert (adjusted >= p - 1e-15).all() and (adjusted <= 1.0).all()
            # three identical datasets: the smallest p-value is multiplied by three
            assert adjusted.min() == pytest.approx(min(1.0, 3 * p.min()))
