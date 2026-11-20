"""The margin core of the certified hybrid as an exploratory entry ``idm_margin`` (decision REV4-1).

The core of ``residual_idm_certified`` is the pooled IDM calibrated with the string-stability margin;
its parameters are stored in the hybrid's run (``train.json["certificate"]["core"]["params"]``, equal to
the buffer ``theta`` of ``model.pt``). This module reads them (:func:`read_core`), writes an IDM entry with
exactly these parameters, no calibration and no training (:func:`build_entry`), and a copy of the stored
``idm`` entry as the control of the evaluation (:func:`copy_control`), both under a result root of their own
(:data:`ROOT`, never ``results/models``). After ``scripts/evaluate.py`` has run there,
:func:`comparison` gives the three-way table (pooled IDM, margin core, hybrid) with the composite and the
rank the margin core would take in place of the hybrid among the reference entries. Nothing here enters a
ranking, a registered test or the leaderboard. Entry point: ``scripts/margin_core.py``.
"""

from __future__ import annotations

import filecmp
import importlib.util
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trafficloop.experiments import analysis as an
from trafficloop.models.mechanistic import IDM
from trafficloop.models.registry import create, save_model
from trafficloop.utils import REPO_ROOT, config_hash, git_revision, read_json, resolve_path, write_json

#: result root of the exploratory entry and of its control (``paths.results`` of the scripts)
ROOT = "results/exploratory/margin_core"
LABEL = "idm_margin"
CONTROL = "idm"
HYBRID = "residual_idm_certified"
#: ``certificate.core.kind`` of the hybrid's core with the stability margin (``CertifiedResidualIDM.before_training``)
CORE_KIND = "pooled with stability margin"
DATASETS = ("ngsim", "follownet_highd", "openacc", "waymo_hu", "follownet_ngsim", "tgsim", "highd")
PROTOCOLS = ("p1", "p1s", "p2", "p2h", "p3", "p4")
VERSION, SPLIT, FOLD, SEED = an.VERSION, "driver", 0, 0
#: metrics of the comparison: one step, whole-event closed loop, the composite inputs
METRICS = ("p0_rmse_a", "p1_whole_rmse_s", "p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100",
           "p3_unstable_fraction")
COMPARISON_COLUMNS = ["dataset", "model", "family", *METRICS, "composite", "rank", "n_models",
                      "composite_in_place_of_hybrid", "rank_in_place_of_hybrid", "control_max_abs_diff",
                      "control_n_compared", "note"]


class MarginCoreError(ValueError):
    """The hybrid's run is missing, its core is not the margin core, or ``theta`` differs from the record."""


# ----------------------------------------------------------------------------- the entry


def run_dir(root, dataset: str, label: str) -> Path:
    """Run directory of fold 0, seed 0 of the driver split under a ``models`` root."""
    return an.run_dir(root, dataset, label, SPLIT, FOLD, SEED, VERSION)


def read_core(source: Path) -> dict:
    """The margin core of the hybrid's run ``source``: parameters in the order of ``IDM.param_names``.

    Raises :class:`MarginCoreError` when ``train.json`` or ``model.pt`` is missing, the core is not the
    margin core, or ``theta`` of the checkpoint is not bit-exactly ``certificate.core.params``.
    """
    source = Path(source)
    train_path, model_path = source / "train.json", source / "model.pt"
    if not (train_path.exists() and model_path.exists()):
        raise MarginCoreError(f"no finished {HYBRID} run at {source} (train.json and model.pt needed)")
    train = read_json(train_path)
    core = (train.get("certificate") or {}).get("core") or {}
    if core.get("kind") != CORE_KIND or core.get("stability_margin") is None:
        raise MarginCoreError(f"{source}: the core is {core.get('kind')!r} with margin {core.get('stability_margin')}, "
                              f"not {CORE_KIND!r}")
    names = list(IDM.param_names)
    if set(core.get("params", {})) != set(names):
        raise MarginCoreError(f"{source}: core parameters {sorted(core.get('params', {}))}, expected {names}")
    params = [float(core["params"][n]) for n in names]
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    theta = state.get("state", {}).get("theta")
    if theta is None or not torch.equal(theta.to(torch.float64), torch.tensor(params, dtype=torch.float64)):
        raise MarginCoreError(f"{source}: theta of model.pt {None if theta is None else theta.tolist()} differs from "
                              f"certificate.core.params {params}")
    return {
        "params": params,
        "kind": core["kind"],
        "stability_margin": float(core["stability_margin"]),
        "margin_min": core.get("margin_min"),
        "config_hash": train.get("config_hash"),
        "checkpoint_config_hash": state.get("provenance", {}).get("config_hash"),
        "git_revision": train.get("git_revision"),
        "split": train.get("split"),
        "run": _relative(source),
    }


def _relative(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it (no machine path in a result file)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def core_model(core: Mapping) -> IDM:
    """The IDM entry with the parameters of the core (float64, as the hybrid's ``theta``)."""
    return create("idm", params=list(core["params"]))


def entry_key(run_config: Mapping, core: Mapping) -> str:
    """Configuration hash of the entry: the run configuration and the source (its hash and the parameters).

    A retrained hybrid gives another key, so ``scripts/evaluate.py`` recomputes the protocol files.
    """
    return config_hash({"entry": dict(run_config), "source_config_hash": core["config_hash"],
                        "params": list(core["params"])})


def training_log(core: Mapping, device: str | None = None) -> dict:
    """``train.json["training"]`` of the entry: where the parameters come from; nothing was calibrated."""
    return {
        "kind": f"{core['kind']} {core['stability_margin']:g}, read from {HYBRID} (config hash {core['config_hash']})",
        "source": {key: core[key] for key in ("run", "config_hash", "kind", "stability_margin", "margin_min",
                                               "git_revision")},
        "params": dict(zip(IDM.param_names, core["params"])),
        "calibrated": False,
        "seconds": 0.0,
        **({} if device is None else {"scored_on": device}),
    }


def write_entry(out: Path, core: Mapping, provenance: Mapping, scores: Mapping | None = None,
                tables: Mapping[str, pd.DataFrame] | None = None, device: str | None = None) -> dict:
    """Write ``model.pt`` and ``train.json`` of ``idm_margin`` into ``out`` (and the test tables, if given)."""
    out = Path(out)
    model = core_model(core)
    save_model(model, out / "model.pt", dict(provenance))
    summary = {
        "model": LABEL,
        "class": "idm",
        "family": model.family,
        **provenance,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training": training_log(core, device),
        "exploratory": "decision REV4-1: not a reference entry, enters no ranking, test or leaderboard",
        "test": dict(scores or {}),
    }
    write_json(out / "train.json", summary)
    for name, table in (tables or {}).items():
        table.to_parquet(out / f"{name}.events.parquet", index=False)
    return summary


def overrides(dataset: str, label: str = LABEL, root: str = ROOT) -> list[str]:
    """Hydra overrides of ``scripts/evaluate.py`` (and of the configuration the entry is written with)."""
    return [f"dataset={dataset}", f"model={'idm_margin' if label == LABEL else label}",
            f"paths.results={Path(root).as_posix()}", f"train.version={VERSION}", f"train.split={SPLIT}",
            f"train.fold={FOLD}", f"train.seed={SEED}"]


def check_root(out: Path, root: str = ROOT) -> None:
    """``out`` must lie under ``root`` and not under ``results/models`` (the reference runs)."""
    parents = Path(out).resolve().parents
    if resolve_path(root).resolve() not in parents or resolve_path("results/models").resolve() in parents:
        raise MarginCoreError(f"{out} is not under {root} or lies under results/models")


def reference_labels() -> tuple[str, ...] | None:
    """The 25 reference entries (``MODELS`` of ``scripts/m2.py``), ``None`` when that file cannot be read."""
    spec = importlib.util.spec_from_file_location("_m2_models", REPO_ROOT / "scripts" / "m2.py")
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - the labels only filter the runs of a dataset missing from the stored table
        return None
    return tuple(module.MODELS)


def evaluate_args(dataset: str, label: str = LABEL, root: str = ROOT) -> list[str]:
    return [*overrides(dataset, label, root), f"protocols=[{','.join(PROTOCOLS)}]"]


def build_entry(dataset: str, models_root=Path("results/models"), root: str = ROOT, force: bool = False) -> dict:
    """Read the core of the hybrid of ``dataset``, score the IDM with it on the test part as the trainer does
    (:func:`trafficloop.train.run.score_model`) and write the run; a finished run with the same key is kept."""
    from trafficloop import pipeline
    from trafficloop.data.splits import Split
    from trafficloop.experiments.sweep import compose_config
    from trafficloop.train.run import model_dir, score_model

    core = read_core(run_dir(resolve_path(models_root), dataset, HYBRID))
    cfg = compose_config("train", overrides(dataset, LABEL, root))
    out = model_dir(cfg)
    check_root(out, root)
    key = entry_key(pipeline.run_config(cfg), core)
    done = out / "train.json"
    if done.exists() and read_json(done).get("config_hash") == key and not force:
        return {"dataset": dataset, "directory": out.as_posix(), "status": "kept", "config_hash": key}
    started = time.time()
    t = cfg.train
    split = Split.load(pipeline.split_path(cfg, t.split))
    if core["split"] and int(core["split"].get("seed", split.seed)) != int(split.seed):
        raise MarginCoreError(f"{dataset}: split seed {split.seed} differs from the hybrid's {core['split']}")
    events = pipeline.load_events(cfg, t.version)
    test = pipeline._usable(pipeline.part_events(events, split, t.fold, "test"), cfg.protocol.warmup)
    model = core_model(core)
    scores, tables = score_model(cfg, model, test)
    provenance = {
        "dataset": dataset,
        "data_version": t.version,
        "split": {"kind": t.split, "fold": int(t.fold), "seed": int(split.seed)},
        "seed": int(t.seed),
        "config_hash": key,
        "git_revision": git_revision(),
    }
    write_entry(out, core, provenance, scores, tables, device=pipeline._device(t.device))
    return {"dataset": dataset, "directory": out.as_posix(), "status": "written", "config_hash": key,
            "seconds": round(time.time() - started, 1), **scores}


def copy_control(dataset: str, models_root=Path("results/models"), root: str = ROOT, force: bool = False) -> dict:
    """Copy ``model.pt`` and ``train.json`` of the stored ``idm`` run into ``<root>/models`` (label ``idm``)."""
    source = run_dir(resolve_path(models_root), dataset, CONTROL)
    target = run_dir(resolve_path(root) / "models", dataset, CONTROL)
    names = ("model.pt", "train.json")
    if not all((source / n).exists() for n in names):
        raise MarginCoreError(f"no stored {CONTROL} run at {source}")
    if source.resolve() == target.resolve():
        raise MarginCoreError(f"the control would overwrite the stored run {source}")
    same = all((target / n).exists() and filecmp.cmp(source / n, target / n, shallow=False) for n in names)
    if same:
        return {"dataset": dataset, "directory": target.as_posix(), "status": "kept"}
    if any((target / n).exists() for n in names) and not force:
        raise MarginCoreError(f"{target} holds another {CONTROL} run than {source}; pass force=true to replace it")
    target.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copyfile(source / n, target / n)
    return {"dataset": dataset, "directory": target.as_posix(), "status": "copied"}


# ----------------------------------------------------------------------------- comparison


def what_if(rows: pd.DataFrame, replacement: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Composite and rank (average ranks, 1 = best) of the E3 rows with the hybrid's row replaced by ``replacement``."""
    frame = rows.reset_index(drop=True).copy()
    at = frame.index[frame["model"] == HYBRID]
    if len(at) != 1:
        raise MarginCoreError(f"expected one {HYBRID} row, found {len(at)}")
    for column in an.E3_METRICS:
        if column != "composite" and column in replacement and column in frame.columns:
            frame.loc[at[0], column] = replacement[column]
    composite = an._composite(frame)
    rank = pd.Series(composite).rank(method="average").to_numpy(dtype=np.float64)
    return composite, rank


def _leaves(value, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a nested summary, keyed by their path."""
    if isinstance(value, Mapping):
        out: dict[str, float] = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{prefix}{key}."))
        return out
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    return {prefix.rstrip("."): float(value)}


def control_difference(stored: Path, control: Path, protocols: Sequence[str] = PROTOCOLS) -> tuple[float, int]:
    """Largest absolute difference over the numeric summary values of the protocol files of the control and the
    stored ``idm`` run, and the number of values compared (NaN, 0 when the control has not been evaluated)."""
    diffs = []
    for name in protocols:
        a, b = an._summary(stored, name), an._summary(control, name)
        if a is None or b is None:
            continue
        la, lb = _leaves(a), _leaves(b)
        diffs += [abs(la[k] - lb[k]) for k in la.keys() & lb.keys() if np.isfinite(la[k]) or np.isfinite(lb[k])]
    if not diffs:
        return float("nan"), 0
    return float(np.nanmax(np.where(np.isnan(diffs), np.inf, diffs))), len(diffs)


def _missing(directory: Path, protocols: Sequence[str] = PROTOCOLS) -> list[str]:
    return [p for p in protocols if not (directory / f"{p}.json").exists()]


def comparison(root_models, root_exploratory=ROOT, datasets: Sequence[str] = DATASETS,
               labels: Sequence[str] | None = None, stored: pd.DataFrame | None = None) -> pd.DataFrame:
    """Three rows per dataset (``idm``, ``idm_margin``, ``residual_idm_certified``), aggregate numbers only.

    ``stored``: the E3 table of the reference entries (``results/m4/e3_rankings.csv``); a dataset it lacks is
    computed with :func:`~trafficloop.experiments.analysis.e3_rankings` on ``root_models`` (``labels``: the
    reference entries). ``composite`` and ``rank`` are those of the reference table; for ``idm_margin``
    ``composite_in_place_of_hybrid`` and ``rank_in_place_of_hybrid`` come from the same table with the hybrid's
    row replaced by the margin core's (:func:`what_if`). ``control_max_abs_diff`` (row ``idm``): the control
    evaluated under ``root_exploratory`` against the stored ``idm`` run (:func:`control_difference`).
    """
    root_models, exploratory = Path(resolve_path(root_models)), Path(resolve_path(root_exploratory)) / "models"
    rows = []
    for dataset in datasets:
        source = run_dir(root_models, dataset, HYBRID)
        margin_dir = run_dir(exploratory, dataset, LABEL)
        if not (margin_dir / "train.json").exists():
            continue
        if stored is not None and dataset in set(stored["dataset"]):
            table = stored[stored["dataset"] == dataset].reset_index(drop=True)
        else:
            table = an.e3_rankings(root_models, dataset, labels)
        if HYBRID not in set(table["model"]) or not (source / "train.json").exists():
            continue
        rank = table["composite"].rank(method="average").to_numpy(dtype=np.float64)
        margin = an.e3_rankings(exploratory, dataset, [LABEL]).iloc[0].to_dict()
        missing = _missing(margin_dir)
        composite, ranks = what_if(table, margin)
        at = int(np.flatnonzero(table["model"].to_numpy() == HYBRID)[0])
        diff, n = control_difference(run_dir(root_models, dataset, CONTROL), run_dir(exploratory, dataset, CONTROL))
        for model in (CONTROL, LABEL, HYBRID):
            if model == LABEL:
                row = dict(margin)
                row.update(composite=np.nan, rank=np.nan,
                           composite_in_place_of_hybrid=np.nan if missing else float(composite[at]),
                           rank_in_place_of_hybrid=np.nan if missing else float(ranks[at]),
                           note=f"not evaluated: {','.join(missing)}" if missing else "")
            else:
                i = int(np.flatnonzero(table["model"].to_numpy() == model)[0]) if model in set(table["model"]) else None
                row = {} if i is None else table.iloc[i].to_dict()
                row.update(model=model, composite=np.nan if i is None else float(table["composite"].iloc[i]),
                           rank=np.nan if i is None else float(rank[i]), note="" if i is not None else "no run")
                if model == CONTROL:
                    row.update(control_max_abs_diff=diff, control_n_compared=n)
            row.update(dataset=dataset, n_models=len(table))
            rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def write_comparison(root_models="results/models", root_exploratory=ROOT, datasets: Sequence[str] = DATASETS,
                     labels: Sequence[str] | None = None, stored_csv="results/m4/e3_rankings.csv") -> Path:
    """``<root_exploratory>/comparison.csv``."""
    path = resolve_path(stored_csv)
    stored = pd.read_csv(path) if path.exists() else None
    labels = reference_labels() if labels is None else labels
    frame = comparison(root_models, root_exploratory, datasets, labels, stored)
    out = resolve_path(root_exploratory) / "comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    print(write_comparison())
