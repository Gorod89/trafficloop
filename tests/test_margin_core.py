"""The margin core of the certified hybrid as the exploratory entry ``idm_margin`` (decision REV4-1): the core is
read bit-exactly from the hybrid's run, the IDM built from it is the hybrid without its residual, broken sources
are refused, the entry stays out of the reference campaign, and the what-if composite reproduces the stored one."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from trafficloop.data.batch import A_PREV, DV, S, V
from trafficloop.experiments import margin_core as mc
from trafficloop.models.mechanistic import IDM
from trafficloop.models.registry import create, load_model, save_model
from trafficloop.utils import REPO_ROOT, write_json

CORE = (35.2, 0.98, 3.19, 1.57, 0.5)
PROVENANCE = {"dataset": "synthetic", "data_version": "reconstructed",
              "split": {"kind": "driver", "fold": 0, "seed": 0}, "seed": 0, "config_hash": "abc123def456",
              "git_revision": None}


def _hybrid_run(directory: Path, core=CORE, kind: str = mc.CORE_KIND, seed: int = 0):
    """A fake run of the certified hybrid: random residual weights, the core in ``theta`` and in ``train.json``."""
    torch.manual_seed(seed)
    model = create(mc.HYBRID)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    rng = np.random.default_rng(seed)
    params = [c * float(rng.uniform(0.9, 1.1)) for c in core]  # values that are not round in float64
    model.set_core(params)
    save_model(model, directory / "model.pt", PROVENANCE)
    record = {"kind": kind, "stability_margin": 0.2, "params": model.core_params(), "margin_min": 0.21}
    write_json(directory / "train.json", {"model": mc.HYBRID, "family": "network", **PROVENANCE,
                                          "certificate": {"core": record, "attempts": []}})
    return model.eval()


def _states(n: int = 500, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    hist = np.zeros((n, 1, 4))
    hist[:, 0, S] = rng.uniform(1.0, 80.0, n)
    hist[:, 0, DV] = rng.uniform(-4.0, 4.0, n)
    hist[:, 0, V] = rng.uniform(0.0, 35.0, n)
    hist[:, 0, A_PREV] = rng.uniform(-2.0, 2.0, n)
    return torch.tensor(hist, dtype=torch.float64)


def test_core_is_read_bit_exactly_and_equals_the_hybrid_without_residual(tmp_path):
    hybrid = _hybrid_run(tmp_path / "hybrid")
    core = mc.read_core(tmp_path / "hybrid")
    assert core["params"] == hybrid.theta.tolist()
    summary = mc.write_entry(tmp_path / "out", core, PROVENANCE)
    entry, provenance = load_model(tmp_path / "out" / "model.pt")
    assert isinstance(entry, IDM) and provenance == PROVENANCE
    assert torch.equal(entry.params, hybrid.theta)
    assert summary["family"] == "mechanistic" and summary["model"] == mc.LABEL and summary["class"] == "idm"
    assert summary["training"]["params"] == dict(zip(IDM.param_names, core["params"]))
    assert "residual_idm_certified" in summary["training"]["kind"] and summary["training"]["calibrated"] is False
    hist = _states()
    with torch.no_grad():
        assert not torch.equal(hybrid(hist), entry(hist))  # the residual is not zero ...
        hybrid.r_max = 0.0  # ... until it is switched off
        assert torch.equal(hybrid(hist), entry(hist))
        assert torch.equal(hybrid.core_acc(hist), entry(hist))


def test_broken_sources_are_refused(tmp_path):
    with pytest.raises(mc.MarginCoreError, match="no finished"):
        mc.read_core(tmp_path / "missing")
    _hybrid_run(tmp_path / "pooled", kind="pooled")
    with pytest.raises(mc.MarginCoreError, match="not 'pooled with stability margin'"):
        mc.read_core(tmp_path / "pooled")
    _hybrid_run(tmp_path / "shifted")
    path = tmp_path / "shifted" / "train.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["certificate"]["core"]["params"]["T"] = float(np.nextafter(data["certificate"]["core"]["params"]["T"], 10.0))
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(mc.MarginCoreError, match="differs"):
        mc.read_core(tmp_path / "shifted")
    (tmp_path / "shifted" / "model.pt").unlink()
    with pytest.raises(mc.MarginCoreError, match="no finished"):
        mc.read_core(tmp_path / "shifted")


def test_entry_stays_out_of_the_reference_campaign():
    labels = mc.reference_labels()
    assert labels is not None and len(labels) == 25 and mc.HYBRID in labels and mc.CONTROL in labels
    assert mc.LABEL not in labels
    assert Path(mc.ROOT).as_posix() != "results/models" and not Path(mc.ROOT).as_posix().startswith("results/models/")
    assert f"paths.results={mc.ROOT}" in mc.overrides("ngsim") and "model=idm_margin" in mc.overrides("ngsim")
    assert "model=idm" in mc.overrides("ngsim", mc.CONTROL)
    conf = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / "idm_margin.yaml"))
    assert conf["name"] == "idm" and conf["label"] == mc.LABEL
    mc.check_root(REPO_ROOT / mc.ROOT / "models" / "x")
    with pytest.raises(mc.MarginCoreError):
        mc.check_root(REPO_ROOT / "results" / "models" / "x", "results")
    with pytest.raises(mc.MarginCoreError):
        mc.check_root(REPO_ROOT / "results" / "models" / "x")


def test_entry_key_follows_the_source():
    core = {"params": list(CORE), "config_hash": "aaa"}
    key = mc.entry_key({"a": 1}, core)
    assert key == mc.entry_key({"a": 1}, dict(core))
    assert key != mc.entry_key({"a": 1}, {**core, "config_hash": "bbb"})
    assert key != mc.entry_key({"a": 1}, {**core, "params": [*CORE[:4], 0.6]})


def test_control_difference_of_identical_runs_is_zero(tmp_path):
    for name in ("a", "b"):
        write_json(tmp_path / name / "p3.json", {"summary": {"unstable_fraction": 0.25, "grid": {"n": 4}, "ok": True}})
    assert mc.control_difference(tmp_path / "a", tmp_path / "b") == (0.0, 2)
    write_json(tmp_path / "b" / "p3.json", {"summary": {"unstable_fraction": 0.5, "grid": {"n": 4}}})
    assert mc.control_difference(tmp_path / "a", tmp_path / "b") == (0.25, 2)
    assert np.isnan(mc.control_difference(tmp_path / "a", tmp_path / "missing")[0])


def test_what_if_with_the_hybrid_reinserted_reproduces_the_stored_composite():
    path = REPO_ROOT / "results" / "m4" / "e3_rankings.csv"
    if not path.exists():
        pytest.skip("results/m4/e3_rankings.csv is absent")
    stored = pd.read_csv(path)
    checked = 0
    for dataset, rows in stored.groupby("dataset"):
        if mc.HYBRID not in set(rows["model"]):
            continue
        hybrid = rows[rows["model"] == mc.HYBRID].iloc[0].to_dict()
        composite, rank = mc.what_if(rows, hybrid)
        np.testing.assert_allclose(composite, rows["composite"].to_numpy(dtype=float), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(rank, rows["composite"].rank(method="average").to_numpy())
        checked += 1
    assert checked >= 1
