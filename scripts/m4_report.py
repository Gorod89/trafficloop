"""Collect the experiments of milestone 4 into ``results/m4/REPORT.md`` and CSV tables.

    python scripts/m4_report.py [root=results/models] [out=results/m4] [n_boot=1000] [seed=0]
                                [splits=data/splits] [events=data/events] [models=main]

E1-E4 of docs/experiments.md, section 3 (``trafficloop/experiments/analysis.py``), and E3b (composite
variants with realism inputs, their sensitivity and the Pareto view), from the run directories
``<root>/<dataset>/reconstructed/<split>-fold<k>/<label>/seed<s>/``. A campaign that has not run
(or not finished) is stated in the report; its tables stay empty. ``models=main`` (default)
restricts E1 and E3 to the reference models of the main campaign (``MODELS`` of
``scripts/m2.py``; the ablation runs of ``scripts/m4.py ablation`` share the directory of fold 0),
``models=all`` takes every finished run. Writes ``REPORT.md``, ``e1_ratios.csv``,
``e2_paired.csv``, ``e2_unpaired.csv``, ``e3_rankings.csv``, ``e3_kendall.csv`` (with the
spacing-against-speed rows), ``e3_composite_variants.csv``, ``e3_composite_sensitivity.csv``,
``e3_pareto.csv``, ``e4_transfer.csv``, ``e1_horizons.csv`` and ``e1_horizon_taus.csv`` (E1b), ``e3_human_reference.csv``, ``e3_human_taus.csv`` and ``e3_human_top3.csv`` (E3d), ``e3_within_family.csv`` (E3e), ``e3f_missing_rule.csv``, ``e3f_tau_bias.csv``,
``e3f_collision_gate.csv``, ``e3f_leave_one_out.csv``, ``e3f_leave_one_out_entries.csv``,
``e3f_seed_fold.csv``, ``e3f_h53_rule.csv``, ``e3f_h51_rule.csv`` and ``e3f_run_statistics.csv`` (E3f,
exploratory robustness checks of tau(P0, composite) and the registered rules H5.1 and H5.3, and the
statistics of H5.1, H5.3 and H5.4 in the other training runs and the block run of ``highd``), and
``e3h_h51_one_step.csv``, ``e3h_h53_all_profiles.csv`` and ``e3h_group_contrast.csv`` (E3h, the exploratory
further checks, decision REV5-1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trafficloop.experiments import analysis as an
from trafficloop.experiments import horizons as hz
from trafficloop.experiments import human_reference as hr
from trafficloop.experiments import robustness as rb
from trafficloop.experiments import within_family as wf
from trafficloop.utils import REPO_ROOT, resolve_path


def _shown(path: Path) -> str:
    """A path relative to the repository root when it lies inside it (no local absolute paths in the report)."""
    path = Path(path)
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()

FAMILY_ORDER = {"baseline": 0, "mechanistic": 1, "network": 2}
P_COLUMNS = {"p", "p_bootstrap", "p_wilcoxon", "p_holm", "p_holm_wilcoxon"}
#: E3 columns shown in the report (all of them are in e3_rankings.csv)
E3_SHOWN = {
    "p0_rmse_a": "P0", "p1_h10_rmse_s": "P1 h10", "p1_h30_rmse_s": "P1 h30", "p1_whole_rmse_s": "P1 whole",
    "p2_concave_growth_error": "P2 concave", "p2_hysteresis_error": "P2 hyst.", "p2_collisions_100": "P2 coll.",
    "p3_unstable_fraction": "P3 unst.", "p4_collision_rate_synthetic": "P4 syn.", "p4_collision_rate_empirical": "P4 emp.",
    "composite": "composite",
}
DEFAULTS = {"root": "results/models", "out": "results/m4", "n_boot": "1000", "seed": "0", "splits": "data/splits",
            "events": "data/events", "models": "main"}


def arguments(argv: list[str] | None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise SystemExit(f"unknown argument {item!r}; usage: m4_report.py " + " ".join(f"[{k}=...]" for k in DEFAULTS))
        args[key] = value
    if args["models"] not in ("main", "all"):
        raise SystemExit(f"models={args['models']!r}: use models=main or models=all")
    return {"root": resolve_path(args["root"]), "out": resolve_path(args["out"]), "n_boot": int(args["n_boot"]),
            "seed": int(args["seed"]), "splits": resolve_path(args["splits"]), "events": resolve_path(args["events"]),
            "models": args["models"]}


# ----------------------------------------------------------------------------- formatting


def fmt(value, column: str = "") -> str:
    """A cell: 3 decimals, counts as integers, p-values below 0.001 as ``<0.001``, missing ``n/a``."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        if column in P_COLUMNS and value < 0.001:
            return "<0.001"
        if column.startswith("rank"):  # average ranks and their percentiles
            return f"{value:.4g}"
        if value.is_integer() and (column.startswith("n_") or "collisions" in column or column in {"runs", "P2 coll."}):
            return str(int(value))
        return f"{value:.3f}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(frame: pd.DataFrame, columns: list[str] | None = None, names: dict | None = None) -> str:
    """Markdown table of ``frame`` (``columns`` in that order, headers renamed by ``names``)."""
    frame = frame if columns is None else frame[columns]
    if frame.empty:
        return "(no rows)\n"
    headers = [str((names or {}).get(c, c)) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v, c) for v, c in zip(row, headers)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def by_family(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    order = frame["family"].map(FAMILY_ORDER).fillna(len(FAMILY_ORDER))
    return frame.assign(_order=order).sort_values(["_order", value, "model"], kind="stable").drop(columns="_order")


def interval(low: float, high: float) -> str:
    return "n/a" if not (np.isfinite(low) and np.isfinite(high)) else f"[{low:.3f}, {high:.3f}]"


# ----------------------------------------------------------------------------- campaigns


def main_labels() -> tuple[str, ...] | None:
    """Model labels of the main campaign (scripts/m2.py), when importable."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from m2 import MODELS
    except Exception:  # noqa: BLE001 - the counts of the campaign table and the label filter depend on it
        return None
    finally:
        sys.path.pop(0)
    return tuple(MODELS)


def main_models() -> int | None:
    """Number of model labels of the main campaign (scripts/m2.py), when importable."""
    labels = main_labels()
    return None if labels is None else len(labels)


def label_filter(which: str):
    """``labels_of(root, dataset)``: the labels of E1 and E3, ``None`` for every finished run
    (``models=all``, or when scripts/m2.py cannot be imported)."""
    known = main_labels() if which == "main" else None

    def labels_of(root: Path, dataset: str) -> list[str] | None:
        return None if known is None else [m for m in an.run_labels(root, dataset) if m in known]

    return labels_of


def completeness(root: Path) -> pd.DataFrame:
    """Expected and finished runs of every campaign of docs/experiments.md, section 1 (the main
    campaign counts the labels of scripts/m2.py only, when known: the ablations share its directory)."""
    known = main_labels()
    n_models = None if known is None else len(known)
    labels = (*an.SUBSET, an.REFERENCE)
    found = {
        "main": sum(len([m for m in an.run_labels(root, d) if known is None or m in known]) for d in an.MAIN_DATASETS),
        "seeds": sum(len(an.run_labels(root, d, seed=s)) for d in an.MAIN_DATASETS for s in (1, 2)),
        "paired": sum(len(set(an.run_labels(root, d, arm, k)) & set(an.SUBSET)) for d in an.LEAK_DATASETS
                      for arm in (an.CLEAN, an.LEAKY) for k in an.FOLDS),
        "event": sum(len(set(an.run_labels(root, d, "event", 0)) & set(an.SUBSET)) for d in an.LEAK_DATASETS),
        "transfer (training)": sum(len(set(an.run_labels(root, d)) & set(labels)) for d in an.TRANSFER_DATASETS
                                   if d not in an.MAIN_DATASETS),
        "transfer (P1 files)": sum(
            (an.run_dir(root, d, m) / f"p1_{o}.json").exists()
            for d in an.TRANSFER_DATASETS for m in labels for o in an.TRANSFER_DATASETS if o != d
        ),
    }
    expected = {
        "main": None if n_models is None else n_models * len(an.MAIN_DATASETS),
        "seeds": None if n_models is None else 2 * n_models * len(an.MAIN_DATASETS),
        "paired": 2 * len(an.FOLDS) * len(an.SUBSET) * len(an.LEAK_DATASETS),
        "event": len(an.SUBSET) * len(an.LEAK_DATASETS),
        "transfer (training)": len(labels) * (len(an.TRANSFER_DATASETS) - len(an.MAIN_DATASETS)),
        "transfer (P1 files)": len(labels) * len(an.TRANSFER_DATASETS) * (len(an.TRANSFER_DATASETS) - 1),
    }
    rows = []
    for name, n in found.items():
        total = expected[name]
        status = "not run" if n == 0 else ("complete" if total is not None and n >= total else "partial")
        rows.append({"campaign": name, "expected": total if total is not None else float("nan"), "found": n, "status": status})
    return pd.DataFrame(rows)


def section_campaigns(root: Path) -> tuple[str, pd.DataFrame]:
    done = completeness(root)
    text = "\n## Campaigns\n\n" + table(done)
    missing = done.loc[done["status"] != "complete", "campaign"].tolist()
    if missing:
        text += ("\nNot complete: " + ", ".join(f"`{m}` ({s})" for m, s in zip(done["campaign"], done["status"]) if m in missing)
                 + ". Tables that need them are marked *not available* below. `transfer (P1 files)` counts the "
                 "`p1_<other>.json` of every label over the six datasets.\n")
    status = an.campaign_status(root)
    if not status.empty:
        text += "\nRuns found per dataset:\n\n" + table(status)
    stale = []
    for dataset in an.run_datasets(root):
        for label in an.run_labels(root, dataset):
            files = an.stale_protocols(an.run_dir(root, dataset, label))
            if files:
                stale.append(f"{dataset}/{label} ({', '.join(files)})")
    if stale:
        text += ("\nProtocol files computed from an earlier checkpoint than the current `train.json` (stale, decision "
                 "M4-5; their values are used and flagged until the protocols are recomputed): " + "; ".join(stale) + ".\n")
    return text, done


# ----------------------------------------------------------------------------- E1-E4


def section_e1(root: Path, n_boot: int, seed: int, labels_of=None) -> tuple[str, pd.DataFrame]:
    text = ("\n## E1: one-step error relative to persistence (H5.1)\n\n"
            f"`ratio` = RMSE_a(model) / RMSE_a(persistence) of P0 on fold 0 of the driver split (seed 0); `low`, `high`: 95 % "
            f"interval from {n_boot} driver-bootstrap replicates with common random numbers (the follower counts are drawn once "
            "per dataset and applied to every table). `seed_min`/`seed_max`: range over training seeds 0-2; `folds_min`/"
            "`folds_max`: range over the clean arms of folds 0-4 (probe events, SUBSET only); n/a with fewer than two runs.\n")
    datasets = [d for d in an.run_datasets(root) if (an.run_dir(root, d, an.REFERENCE) / "test_p0.events.parquet").exists()]
    if not datasets:
        return text + "\nNot available: no driver-split fold-0 run of persistence.\n", pd.DataFrame(columns=an.E1_COLUMNS)
    frames = []
    for dataset in datasets:
        frame = an.e1_ratios(root, dataset, labels_of(root, dataset) if labels_of else None, n_boot, seed)
        frames.append(frame)
        text += f"\n### {dataset}\n\n"
        below = frame[(frame["high"] < 1.0)]["model"].tolist()
        text += table(by_family(frame, "ratio"), ["model", "family", "rmse_a", "ratio", "low", "high", "n_drivers", "seed_min",
                                                  "seed_max", "n_seeds", "folds_min", "folds_max", "n_folds"])
        text += f"\nInterval entirely below 1 (better than persistence): {', '.join(below) if below else 'none'}.\n"
        if frame["n_seeds"].max() < 2:
            text += "Seed spread: not available (campaign `seeds` has not run).\n"
        if frame["n_folds"].max() < 2:
            text += "Fold spread: not available (campaign `paired` has not produced two clean folds with persistence on the probe events).\n"
    return text, pd.concat(frames, ignore_index=True)


def section_e1b(root: Path, labels_of=None) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E1b: closed-loop spacing error from 1 s windows to the whole event\n\n"
            "`<h>_rmse_s`: pooled spacing RMSE of P1 at the horizon (1 s windows from protocol `p1s`, 10 s and 30 s from `p1`, "
            "the whole event); `<h>_median_rmse_s`: median over the windows (over the events for the whole event); "
            "`whole_top5_share`: share of the whole-event summed squared spacing error carried by the five followers with the "
            "largest sums. The taus between the P0 ranking and the horizon rankings are point estimates (intervals of the P1 "
            "horizons: E3).\n")
    frames = []
    for dataset in an.run_datasets(root):
        frame = hz.e1_horizons(root, dataset, labels_of(root, dataset) if labels_of else None)
        if frame.empty:
            continue
        frames.append(frame)
        text += f"\n### {dataset}\n\n"
        text += table(by_family(frame, "whole_rmse_s"), ["model", "family", "h1_rmse_s", "h1_median_rmse_s", "h10_rmse_s",
                                                         "h10_median_rmse_s", "h30_rmse_s", "h30_median_rmse_s", "whole_rmse_s",
                                                         "whole_median_rmse_s", "whole_top5_share", "n_followers"])
        missing = frame[frame["h1_n_windows"] == 0]["model"].tolist()
        if missing:
            text += f"\n1 s windows not available for: {', '.join(missing)} (protocol `p1s` not run).\n"
    if not frames:
        return text + "\nNot available: no finished run.\n", pd.DataFrame(columns=hz.COLUMNS), pd.DataFrame(columns=hz.TAU_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    taus = hz.horizon_taus(frame)
    text += "\n### Kendall's tau-b between the P0 ranking and the horizon rankings\n\n" + table(taus)
    return text, frame, taus


def section_e2(root: Path, n_boot: int, seed: int, splits: Path, events: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E2: leakage inflation (H5.2)\n\n"
            "Paired design (decision R2): both arms scored on the same probe events of folds 0-4, pooled; `inflation` = (clean - "
            "leaky) / clean (positive: the leak lowers the error); `mean_difference` (clean - leaky), its interval and `p_bootstrap` "
            f"from `paired_difference` ({n_boot} driver replicates, common random numbers), `p_wilcoxon` the signed-rank test on the "
            "per-driver differences, `p_holm` the Holm adjustment of `p_bootstrap` over the models of a dataset, subset and metric, "
            "`p_holm_wilcoxon` that of `p_wilcoxon` over the same family. "
            "Subset `all`: every probe event; `affected`: the probe events of drivers with at least one event in the leaky "
            "training part of the same fold.\n")
    datasets = sorted(set(an.LEAK_DATASETS) | {d for k in an.FOLDS for d in an.run_datasets(root, an.CLEAN, k)})
    paired_frames, unpaired_frames = [], []
    for dataset in datasets:
        text += f"\n### {dataset}\n\n"
        affected, reason = None, ""
        try:
            affected = an.affected_followers(dataset, splits, events)
        except (OSError, KeyError, ValueError) as error:
            reason = f"{type(error).__name__}: {error}"
        frame = an.e2_paired(root, dataset, an.SUBSET, n_boot, seed, affected=affected)
        paired_frames.append(frame)
        if frame.empty:
            text += f"Paired comparison: not available (no fold with both arms of a SUBSET model under `{an.CLEAN}-fold<k>`; campaign `paired`).\n"
        else:
            folds = sorted({k for k in an.FOLDS for m in an.SUBSET if (an.run_dir(root, dataset, m, an.CLEAN, k) / "train.json").exists()})
            text += f"Folds with runs: {', '.join(map(str, folds))}; models: {frame['model'].nunique()} of {len(an.SUBSET)}.\n"
            if affected is None:
                text += f"Affected subset: not available ({reason}).\n"
            else:
                share = frame[(frame["metric"] == "p0_rmse_a")].pivot_table(index="model", columns="subset", values="n_events", aggfunc="first")
                if "affected" in share and "all" in share and len(share):
                    row = share.iloc[0]
                    text += (f"Affected probe events: {int(row['affected'])} of {int(row['all'])} ({row['affected'] / row['all']:.1%}) "
                             f"for {share.index[0]}, drivers {int(frame['n_affected_drivers'].dropna().iloc[0])}.\n")
            for subset in an.E2_SUBSETS:
                part = frame[frame["subset"] == subset]
                if part.empty:
                    continue
                text += f"\nSubset `{subset}`:\n\n" + table(part, ["model", "family", "metric", "clean", "leaky", "inflation", "mean_difference",
                                                                   "low", "high", "p_bootstrap", "p_wilcoxon", "p_holm", "p_holm_wilcoxon", "n_drivers", "n_events", "n_folds"])
            text += "\nClass means:\n\n" + table(an.e2_class_means(frame), ["subset", "metric", "family", "mean_inflation", "min_inflation",
                                                                         "max_inflation", "mean_difference", "n_models"])
        unpaired = an.e2_unpaired(root, dataset, an.SUBSET)
        unpaired_frames.append(unpaired)
        if unpaired.empty:
            text += "\nUnpaired comparison (event split against driver split, fold 0): not available (no SUBSET model with runs of both splits; campaign `event`).\n"
        else:
            text += ("\nUnpaired comparison (event split against driver split on fold 0, each on its own test part; `inflation` = "
                     "(driver - event) / driver, no test):\n\n" + table(unpaired, ["model", "family", "metric", "driver", "event", "inflation",
                                                                                   "difference", "n_events_driver", "n_events_event"]))
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame(columns=an.E2_COLUMNS)
    unpaired = pd.concat(unpaired_frames, ignore_index=True) if unpaired_frames else pd.DataFrame(columns=an.E2_UNPAIRED_COLUMNS)
    return text, paired, unpaired


E3_PROTOCOLS = ("p2", "p3", "p4")


def e3_datasets(root: Path, labels_of=None, incomplete: dict | None = None) -> list[str]:
    """Datasets whose fold-0 runs carry the protocols P2-P4 (the main campaign): at least two runs with
    ``p2.json`` and every run of the labels (``labels_of``, default every finished run) with P2-P4. A
    dataset whose campaign is still evaluating (some runs without P2-P4) is left out and, when
    ``incomplete`` is given, entered there with the number of complete runs and the first missing ones."""
    out = []
    for dataset in an.run_datasets(root):
        labels = labels_of(root, dataset) if labels_of else None
        labels = an.run_labels(root, dataset) if labels is None else labels
        if sum((an.run_dir(root, dataset, m) / "p2.json").exists() for m in labels) < 2:
            continue
        missing = [m for m in labels if not all((an.run_dir(root, dataset, m) / f"{p}.json").exists() for p in E3_PROTOCOLS)]
        if missing:
            if incomplete is not None:
                incomplete[dataset] = f"{len(labels) - len(missing)} of {len(labels)} runs with P2-P4 (missing e.g. {', '.join(missing[:3])})"
            continue
        out.append(dataset)
    return out


def section_e3(root: Path, n_boot: int, seed: int, labels_of=None) -> tuple[str, pd.DataFrame, pd.DataFrame, dict]:
    """The E3 text, the rankings and Kendall tables, and ``{dataset: (rankings, replicates)}`` for E3b."""
    text = ("\n## E3: rankings across protocols (H5.3, H5.4)\n\n"
            "Point values of fold 0, seed 0 (lower is better; P2 and the synthetic P4 over the profiles and scenarios inside the "
            "speed range of the training data; P1 whole and P0 from `train.json`). Kendall's tau-b between every pair of rankings: "
            f"`tau` and `p` on the point values, `low`/`high` the 95 % interval and `tau_boot` the mean over {n_boot} replicates "
            "(drivers for P0, P1 and the empirical P4; in-range profiles for P2 and the composite, speeds with an equilibrium for P3, "
            "in-range scenarios for the synthetic P4). A model without a value ranks last.\n")
    incomplete: dict[str, str] = {}
    datasets = e3_datasets(root, labels_of, incomplete)
    missing = [d for d in an.MAIN_DATASETS if d not in datasets and d not in incomplete]
    if missing:
        text += f"\nNot available for {', '.join(missing)}: no fold-0 runs with P2-P4 (campaign `main`).\n"
    if incomplete:
        text += ("\nLeft out while their campaign is still evaluating (a ranking needs every model): "
                 + "; ".join(f"{d}: {why}" for d, why in incomplete.items()) + ".\n")
    rankings_frames, kendall_frames, cache = [], [], {}
    for dataset in datasets:
        started = time.time()
        rankings = an.e3_rankings(root, dataset, labels_of(root, dataset) if labels_of else None)
        replicates = an.e3_replicates(root, dataset, rankings, n_boot, seed)
        cache[dataset] = (rankings, replicates)
        kendall = an.e3_kendall(root, dataset, n_boot, seed, rankings=rankings, replicates=replicates)
        speed = an.e3_spacing_vs_speed(root, dataset, n_boot, seed, rankings=rankings, replicates=replicates)
        rankings_frames.append(rankings)
        kendall_frames += [kendall, speed.assign(ranking_a="P1 " + speed["horizon"] + " rmse_s",
                                                 ranking_b="P1 " + speed["horizon"] + " rmse_v").reindex(columns=an.KENDALL_COLUMNS)]
        text += f"\n### {dataset}\n\n" + table(by_family(rankings, "p1_whole_rmse_s"), ["model", "family", *E3_SHOWN, "stale"], E3_SHOWN)
        reasons = replicates[1]
        if reasons:
            text += "\nNo bootstrap replicates (point values and point tau only):\n\n" + "".join(f"* `{c}`: {r}\n" for c, r in reasons.items())
        text += "\nKendall tau between rankings:\n\n" + table(kendall, ["ranking_a", "ranking_b", "tau", "low", "high", "p", "tau_boot", "n_models"])
        text += "\nSpacing against speed (H5.4): tau between the `rmse_s` and the `rmse_v` ranking of P1:\n\n"
        text += table(speed, ["horizon", "tau", "low", "high", "p", "tau_boot", "n_models"])
        text += "\nTop three (`tied`: further models tied with the third):\n\n" + table(an.e3_top3(rankings), ["ranking", "top1", "top2", "top3", "values", "tied"])
        text += "\nEntries into and exits from the top three:\n\n" + "".join(f"* {line.split(': ', 1)[1]}\n" for line in an.e3_top3_changes(rankings))
        print(f"E3 {dataset}: {time.time() - started:.1f} s", flush=True)
    rankings = pd.concat(rankings_frames, ignore_index=True) if rankings_frames else pd.DataFrame(columns=an.E3_COLUMNS)
    kendall = pd.concat([k for k in kendall_frames if not k.empty], ignore_index=True) if any(not k.empty for k in kendall_frames) \
        else pd.DataFrame(columns=an.KENDALL_COLUMNS)
    return text, rankings, kendall, cache


# ----------------------------------------------------------------------------- E3b


def _rank_interval(low: float, high: float) -> str:
    """A rank interval at one decimal, as on the leaderboard."""
    return "n/a" if not (np.isfinite(low) and np.isfinite(high)) else f"[{low:.1f}, {high:.1f}]"


def _tau_text(row) -> str:
    """``tau [low, high]`` of tau(P0, composite)."""
    if not np.isfinite(row["tau_p0"]):
        return "n/a"
    return f"{row['tau_p0']:.3f}" + (f" {interval(row['tau_p0_low'], row['tau_p0_high'])}" if np.isfinite(row["tau_p0_low"]) else "")


def p0_sign_sentence(sensitivity: pd.DataFrame) -> tuple[str, bool | None]:
    """Whether tau(P0, composite) keeps the sign of v1 (the headline of E3) under every variant,
    and the range and the intervals over every composite of the sensitivity table; the second
    value is True when the sign is kept under every variant (None without a v1 value)."""
    rows = sensitivity.set_index("variant_or_leave_one_out")
    if "v1" not in rows.index or not np.isfinite(rows.at["v1", "tau_p0"]):
        return "Kendall tau of P0 against the composite: not available (no P0 or composite ranking).", None
    sign = np.sign(rows.at["v1", "tau_p0"])
    variants = [v for v in an.VARIANTS if v in rows.index]
    flipped = [v for v in variants if np.isfinite(rows.at[v, "tau_p0"]) and np.sign(rows.at[v, "tau_p0"]) != sign]
    listed = ", ".join(f"{v} {_tau_text(rows.loc[v])}" for v in variants)
    kept = not flipped
    text = (f"Kendall tau of P0 against the composite: {listed}. The sign of the headline (v1, "
            f"{'negative' if sign < 0 else 'positive' if sign > 0 else 'zero'}) is "
            + ("kept under every variant" if kept else f"reversed under {', '.join(flipped)}"))
    taus = rows["tau_p0"].to_numpy(dtype=np.float64)
    finite = np.isfinite(taus)
    if finite.any():
        same = int((np.sign(taus[finite]) == sign).sum())
        low, high = rows["tau_p0_low"].to_numpy(dtype=np.float64), rows["tau_p0_high"].to_numpy(dtype=np.float64)
        has = np.isfinite(low) & np.isfinite(high)
        excludes = has & ((high < 0) | (low > 0))
        open_rows = [label for label, h, e in zip(rows.index, has, excludes) if h and not e]
        text += (f"; over all {len(rows)} composites of the sensitivity table it lies between {np.nanmin(taus):.3f} and "
                 f"{np.nanmax(taus):.3f} with the sign of v1 in {same} of {int(finite.sum())}, and its 95 % interval excludes 0 "
                 f"in {int(excludes.sum())} of {int(has.sum())}" + (f" (not for {', '.join(open_rows)})" if open_rows else ""))
    return text + ".", kept


def section_e3b(root: Path, cache: dict, n_boot: int, seed: int) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E3b: composite variants and sensitivity\n\n"
            "The composite of E3 (`v1`: mean normalised rank of the concave-growth error, the hysteresis error, the unstable "
            "fraction and the 100-vehicle collisions) has no realism input: a follower that keeps a large gap and never reacts "
            "is stable and never collides. The variants add the whole-event P1 spacing RMSE (`v1+p1`), the relative "
            "fundamental-diagram error of P2 (`fd_error_rel`; `v1+fd`) or both (`v2`, six inputs, equal weights). `borda v2` "
            "is the Borda count of v2 (sum of n - rank, normalised to [0, 1] with 0 = best); with average ranks on ties it is an "
            "affine function of the rank sum, so it equals v2 by construction. Leave one out: v2 without each of its inputs; "
            "weights: v2 with weight 2 on the unstable fraction, on the collisions and on both (the stability and collision "
            "inputs). `fd_error_rel` is pooled over the samples of every P2 profile (the 20 OpenACC windows and the pulse: P2 "
            "has no in-range version), compares the density bins with at least 200 samples on both sides (`n_FD_bins`, which "
            "differs between models), and has no per-profile version to resample: it is held at its point value in the "
            f"replicates. Intervals ({n_boot} replicates, seed {seed}): the rank interval of v2 and the interval of tau(P0, "
            "composite) resample the in-range profiles and the speeds with an equilibrium, as in E3, and the drivers of the "
            "whole-event P1 input with the draw of P0 (common random numbers); the v1 row reproduces the P0-composite row of "
            "E3. Ranks: 1 = best, average ranks on ties. `top3 changes`: entries into and exits from the top three (ties at the "
            "third position counted in) against v1 for the variants and against v2 for Borda, leave one out and weights. Pareto "
            "view: `on front` marks the models that no other model matches or beats in both the whole-event P1 error and "
            "composite v2 (both lower is better; the P1 error is one of the six inputs of v2).\n")
    if not cache:
        empty = (pd.DataFrame(columns=an.VARIANTS_COLUMNS), pd.DataFrame(columns=an.SENSITIVITY_COLUMNS),
                 pd.DataFrame(columns=an.PARETO_COLUMNS))
        return text + "\nNot available: no dataset with E3 rankings.\n", *empty
    variants_frames, sensitivity_frames, pareto_frames, kept = [], [], [], {}
    for dataset, (rankings, replicates) in cache.items():
        variants = an.e3_composite_variants(root, dataset, rankings=rankings, replicates=replicates)
        sensitivity = an.e3_composite_sensitivity(root, dataset, rankings=rankings, replicates=replicates)
        pareto = an.pareto_table(root, dataset, rankings=rankings)
        variants_frames.append(variants)
        sensitivity_frames.append(sensitivity)
        pareto_frames.append(pareto)
        shown = variants.assign(rank_v2_ci=[_rank_interval(lo, hi) for lo, hi in zip(variants["rank_v2_low"], variants["rank_v2_high"])])
        text += f"\n### {dataset}\n\nComposite variants (best v2 first):\n\n"
        text += table(shown, ["model", "family", an.P1_INPUT, an.FD_INPUT, "fd_bins_compared", *(f"composite_{v}" for v in an.VARIANTS),
                              "composite_borda_v2", *(f"rank_{v}" for v in an.VARIANTS), "rank_v2_ci"],
                      {an.P1_INPUT: "P1 whole (m)", an.FD_INPUT: "FD rel.", "fd_bins_compared": "n_FD_bins",
                       **{f"composite_{v}": v for v in an.VARIANTS}, "composite_borda_v2": "borda v2",
                       **{f"rank_{v}": f"rank {v}" for v in an.VARIANTS}, "rank_v2_ci": "rank v2 [95%]"})
        top5 = ", ".join(variants["model"].head(5))
        text += f"\nTop five under v2: {top5}.\n"
        shown = sensitivity.assign(tau_p0_ci=[interval(lo, hi) for lo, hi in zip(sensitivity["tau_p0_low"], sensitivity["tau_p0_high"])])
        columns = ["variant_or_leave_one_out", "inputs", "kendall_tau_vs_v1", "kendall_tau_vs_v2", "top3", "top3_changes", "tau_p0", "tau_p0_ci"]
        if shown["note"].astype(str).str.len().gt(0).any():
            columns.append("note")
        text += "\nSensitivity (Kendall tau-b on the point values, a missing value last):\n\n"
        text += table(shown, columns, {"variant_or_leave_one_out": "composite", "kendall_tau_vs_v1": "tau vs v1",
                                       "kendall_tau_vs_v2": "tau vs v2", "top3_changes": "top3 changes", "tau_p0": "tau(P0)",
                                       "tau_p0_ci": "tau(P0) [95%]"})
        sentence, kept[dataset] = p0_sign_sentence(sensitivity)
        text += "\n" + sentence + "\n"
        shown = pareto.assign(front=np.where(pareto["on_front"].astype(bool), "yes", ""))
        text += "\nPareto view (by the whole-event P1 error):\n\n"
        text += table(shown, ["model", "family", an.P1_INPUT, "composite_v2", "unstable_fraction", "collisions_100", "front"],
                      {an.P1_INPUT: "P1 whole (m)", "composite_v2": "v2", "unstable_fraction": "P3 unst.", "collisions_100": "P2 coll.",
                       "front": "on front"})
        text += f"\nOn the front: {', '.join(pareto.loc[pareto['on_front'].astype(bool), 'model']) or 'none'}.\n"
    known = {d: k for d, k in kept.items() if k is not None}
    if known:
        reversed_ = [d for d, k in known.items() if not k]
        text += ("\nAcross datasets: the sign of tau(P0, composite) " + ("is kept under every variant on " + ", ".join(known)
                 if not reversed_ else "changes under at least one variant on " + ", ".join(reversed_)) + ".\n")
    return (text, pd.concat(variants_frames, ignore_index=True), pd.concat(sensitivity_frames, ignore_index=True),
            pd.concat(pareto_frames, ignore_index=True))


def section_e3c(root: Path) -> tuple[str, pd.DataFrame]:
    """E3c: the training ablations of the networks next to their reference entries (decision M4-7)."""
    text = ("\n## E3c: training ablations of the networks\n\n"
            "Campaign `ablation` (decision M4-7): rollout-loss horizons of 15 s and 30 s (`*_loop_h15`, `*_loop_h30`), "
            "a 3 s input history (`*_hist30`), a four-fold width (`*_wide`) and the stability penalty evaluated in a "
            "neighbourhood of every equilibrium (`stability_mlp_nbhd`); fold 0, seed 0 of the driver split, same training "
            "budget as the reference entries. `unstable` = numerical P3 fraction at 0.05 m/s, `large` at 1 m/s; collisions "
            "of P2 over the in-range profiles, P4 synthetic in range. The 15 s and 30 s horizons do not exist on "
            "FollowNet-highD (15 s events).\n")
    frames = []
    for dataset in an.run_datasets(root):
        frame = an.ablation_table(root, dataset)
        if frame.empty:
            continue
        frames.append(frame)
        text += f"\n### {dataset}\n\n" + table(frame, ["group", "model", "p0_rmse_a", "p1_h10_rmse_s", "p1_whole_rmse_s",
                                                      "p1_whole_collision_rate", "p2_collisions_100", "p3_n_equilibria",
                                                      "p3_unstable_fraction", "p3_unstable_fraction_large", "p3_max_gain",
                                                      "p4_collision_rate_synthetic", "p4_collision_rate_empirical"])
    if not frames:
        text += "\nNot available (campaign `ablation` has not run).\n"
        return text, pd.DataFrame()
    return text, pd.concat(frames, ignore_index=True)


def section_e3d(root: Path, labels_of=None) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E3d: the composite with the human platoon reference (protocol p2h)\n\n"
            "P2 scores the platoons against recorded followers behind the OpenACC leaders (17 of 19 windows with ACC "
            "followers); p2h repeats the protocol behind human-driven chains of NGSIM and TGSIM and the two manual AstaZero "
            "runs (docs/protocols.md). `acc_*`: in-range P2 inputs, `human_*`: the same inputs of p2h; `composite_v1h` "
            "and `composite_v2h`: the variants v1 and v2 built from the human reference (P1 whole-event error and P3 "
            "unstable fraction unchanged). Taus are point estimates between the rankings of the models of the main "
            "campaign.\n")
    frames = []
    for dataset in an.run_datasets(root):
        frame = hr.e3_human_reference(root, dataset, labels_of(root, dataset) if labels_of else None)
        if frame.empty or frame["human_concave_growth_error"].notna().sum() == 0:
            continue
        frames.append(frame)
        text += f"\n### {dataset}\n\n"
        text += table(frame.sort_values("composite_v2h"), ["model", "family", "acc_concave_growth_error",
                                                            "human_concave_growth_error", "acc_hysteresis_error",
                                                            "human_hysteresis_error", "acc_collisions_100", "human_collisions_100",
                                                            "human_n_profiles_in_range", "composite_v2", "composite_v2h",
                                                            "rank_v2", "rank_v2h"])
    if not frames:
        return (text + "\nNot available: no p2h result.\n", pd.DataFrame(columns=hr.COLUMNS),
                pd.DataFrame(columns=hr.TAU_COLUMNS), pd.DataFrame(columns=hr.TOP_COLUMNS))
    frame = pd.concat(frames, ignore_index=True)
    taus, top = hr.human_reference_taus(frame), hr.human_reference_top3(frame)
    text += "\n### Kendall's tau-b between the rankings\n\n" + table(taus)
    text += "\n### First three models of every composite\n\n" + table(top)
    return text, frame, taus, top


def section_e3e(e3: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    text = ("\n## E3e: rank agreement inside the model families\n\n"
            "Kendall's tau-b between the P0 ranking and the composite, the P3 unstable fraction and the whole-event P1 "
            "spacing error over all entries, inside the learned entries (`network`) and inside the mechanistic entries, per "
            "dataset; point estimates with the two-sided p-value of scipy's kendalltau (the mechanistic family has six "
            "entries, so no bootstrap interval is given). The anti-correlation of E3 is partly between the two families, "
            "whose one-step and traffic-level merits are opposite, and partly inside the learned family.\n")
    frame = wf.within_family_taus(e3) if len(e3) else pd.DataFrame(columns=wf.COLUMNS)
    if frame.empty:
        return text + "\nNot available.\n", frame
    return text + "\n" + table(frame), frame


def section_e3g(e3: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """E3g: the entries by what they are fitted to (exploratory, added after the results)."""
    text = ("\n## E3g: one-step accuracy and the composite by training objective (exploratory)\n\n"
            "Added after the results (2026-11-16). The families of E3e differ in what their entries "
            "are fitted to: the mechanistic entries are calibrated in closed loop on spacing and speed and never on the "
            "one-step acceleration, most learned entries are trained on the one-step error. The entries are grouped by "
            "objective (`trafficloop/experiments/within_family.py`, `OBJECTIVES`; the column \"training or calibration\" of the "
            "entries table). Ranks are average ranks among the reference entries of the dataset (1 = best).\n")
    if not len(e3):
        return (text + "\nNot available.\n", pd.DataFrame(columns=wf.OBJECTIVE_RANK_COLUMNS),
                pd.DataFrame(columns=wf.OBJECTIVE_TAU_COLUMNS))
    ranks, taus = wf.objective_tables(e3)
    text += "\n### Places by objective\n\nMedian (minimum, maximum) rank of the entries of each objective under P0 and under the composite.\n\n"
    shown = ranks.assign(p0=[f"{m:g} ({lo:g}-{hi:g})" for m, lo, hi in zip(ranks["p0_rank_median"], ranks["p0_rank_min"], ranks["p0_rank_max"])],
                         composite=[f"{m:g} ({lo:g}-{hi:g})" for m, lo, hi in
                                    zip(ranks["composite_rank_median"], ranks["composite_rank_min"], ranks["composite_rank_max"])])
    text += table(shown, ["dataset", "objective", "n_models", "p0", "composite"], {"p0": "P0 rank", "composite": "composite rank"})
    text += ("\n### Rank agreement with the objective held fixed\n\n"
             "Kendall's tau-b between the P0 ranking and the composite (over all entries, as in E3e) inside sets of entries "
             "with one kind of objective; two-sided p-value of scipy's kendalltau, no interval (9 to 14 entries).\n\n")
    text += table(taus)
    return text, ranks, taus


E3H_TABLES = {"h51_one_step": rb.H51_ONE_STEP_COLUMNS, "h53_all_profiles": rb.H53_VIEW_COLUMNS,
              "group_contrast": wf.CONTRAST_COLUMNS}


def section_e3h(root: Path, e1: pd.DataFrame, e3: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """E3h: the three further checks (exploratory, decision REV5-1), the frames of their CSVs."""
    text = ("\n## E3h: further checks (exploratory, decision REV5-1)\n\n"
            "Added after the results (2026-11-17), from stored results only; no verdict depends on them. "
            "Fold 0, seed 0 of the driver split.\n")
    out = {name: pd.DataFrame(columns=columns) for name, columns in E3H_TABLES.items()}
    labels = main_labels()
    in_e3 = list(dict.fromkeys(e3["dataset"])) if len(e3) else []
    main = [d for d in an.MAIN_DATASETS if d in in_e3]
    if labels is None or not in_e3:
        return text + "\nNot available: no E3 rankings of the main campaign.\n", out
    if len(e1):
        out["h51_one_step"] = rb.h51_one_step_table(e1[e1["dataset"].isin(main)])
    out["h53_all_profiles"] = pd.concat([rb.h53_views_table(root, d, list(labels)) for d in main], ignore_index=True) \
        if main else out["h53_all_profiles"]
    out["group_contrast"] = wf.group_contrast_table(e3[e3["dataset"].isin(an.TRANSFER_DATASETS)])
    h51 = out["h51_one_step"]
    text += ("\n### (a) H5.1 on the entries with a one-step objective\n\n"
             "The counts of the registered rule H5.1 (`h51_table`) over the ten entries of the group \"one-step objective\" of "
             "`OBJECTIVES` only, from the P0 ratios of E1 (model / persistence): `n_within_10` (0.9 <= ratio <= 1.1), "
             "`n_ci_within_10` (interval inside [0.9, 1.1]), `n_better_30` (ratio < 0.7), `n_worse_30` (ratio > 1.3), "
             "`n_beyond_30` their sum; `ratios` lists every entry.\n\n")
    text += table(h51.assign(ci=_ci(h51, "best_low", "best_high")),
                  ["dataset", "n_models", "n_within_10", "n_ci_within_10", "n_better_30", "n_worse_30", "n_beyond_30", "best_model",
                   "best_ratio", "ci", "ratios"], {"ci": "best 95% CI"})
    h53 = out["h53_all_profiles"]
    text += ("\n### (b) H5.3 with the P2 inputs over all leader profiles\n\n"
             "The composite (`last` rule over all entries) with the concave-growth error, the hysteresis error and the "
             "100-vehicle collisions of P2 taken over every leader profile (the keys of `p2.json` without the suffix "
             "`_in_range`; `p2_view = all`) next to the registered in-range composite (`in_range`); the unstable fraction is "
             "unchanged. `tau`, `p`: Kendall's tau-b between the P0 ranking and the composite; `h53_dropped_*`: entries of the "
             "top three by the one-step and by the whole-event error not in the composite top three (point values); "
             "`missing_p2`: entries without one of the three P2 inputs.\n\n")
    text += table(h53, ["dataset", "p2_view", "n_models", "tau", "p", "top3_composite", "p0_top3", "h53_dropped_p0", "whole_top3",
                        "h53_dropped_whole", "missing_p2"])
    contrast = out["group_contrast"]
    text += ("\n### (c) Contrast between the groups of training objective\n\n"
             f"tau(P0, composite) over all entries (`tau_observed`, as in E3) and over {wf.N_PERMUTATIONS} permutations of the "
             "composite values inside each group of `OBJECTIVES` (one generator seeded with 0 per dataset; a missing value "
             "ranks last and moves with its value): the mean (`tau_perm_mean`), the 2.5 % and 97.5 % quantiles, the share of "
             "permutations with tau at or below the observed one, counting the observed arrangement once ((k + 1) / (n + 1)), "
             "and mean / observed; `groups` gives the group sizes.\n\n")
    text += table(contrast.assign(ci=_ci(contrast, "tau_perm_low", "tau_perm_high")),
                  ["dataset", "n_models", "tau_observed", "tau_perm_mean", "ci", "share_at_or_below", "ratio_mean_observed", "groups"],
                  {"ci": "permutation 95% range"})
    return text, out


E3F_TABLES = ("missing_rule", "tau_bias", "collision_gate", "collision_input", "collision_input_entries", "stability_rule", "leave_one_out",
              "leave_one_out_entries", "input_ablation", "seed_fold", "h53_rule", "h51_rule", "run_statistics")
E3F_EMPTY = {"missing_rule": rb.MISSING_COLUMNS, "tau_bias": rb.BIAS_COLUMNS, "collision_gate": rb.GATE_COLUMNS,
             "collision_input": rb.COLLISION_INPUT_COLUMNS, "collision_input_entries": rb.COLLISION_ENTRY_COLUMNS,
             "stability_rule": rb.STABILITY_RULE_COLUMNS,
             "leave_one_out": rb.LOO_COLUMNS, "leave_one_out_entries": rb.LOO_ENTRY_COLUMNS,
             "input_ablation": rb.INPUT_COLUMNS, "seed_fold": rb.SEED_FOLD_COLUMNS,
             "h53_rule": rb.H53_COLUMNS, "h51_rule": rb.H51_COLUMNS, "run_statistics": rb.RUN_STAT_COLUMNS}


def _ci(frame: pd.DataFrame, low: str, high: str) -> list[str]:
    return [interval(lo, hi) for lo, hi in zip(frame[low], frame[high])]


def section_e3f(root: Path, n_boot: int, seed: int, e1: pd.DataFrame, kendall: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    """E3f: robustness of tau(P0, composite) (exploratory, added after the results), the frames of its CSVs."""
    text = ("\n## E3f: robustness of tau(P0, composite) (exploratory)\n\n"
            "Added after the results (2026-11-16); none of these checks was registered in advance. All "
            "taus are Kendall's tau-b over a fixed set of entries of the main campaign (the 25 reference entries, or the "
            "subset named), fold 0 and seed 0 of the driver split unless stated otherwise. One call of `e3_replicates` per "
            f"dataset ({n_boot} replicates, seed {seed}) gives the replicates of P0, the whole-event P1 error and the four "
            "composite inputs, shared by every treatment below.\n")
    labels = main_labels()
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in E3F_TABLES}
    in_e3 = set(kendall["dataset"]) if len(kendall) else set()
    datasets = [d for d in an.TRANSFER_DATASETS if labels is not None and d in in_e3]
    for dataset in datasets:
        started = time.time()
        for name, frame in rb.e3f_dataset(root, dataset, list(labels), n_boot, seed).items():
            frames[name].append(frame)
        if dataset in an.MAIN_DATASETS:
            frames["run_statistics"].append(rb.run_statistics_table(root, dataset, list(labels), extra=rb.EXTRA_RUNS.get(dataset, ())))
        print(f"E3f {dataset}: {time.time() - started:.1f} s", flush=True)
    if labels is not None and len(e1):
        frames["h51_rule"].append(rb.h51_table(e1[e1["dataset"].isin(datasets)], labels))
    out = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=E3F_EMPTY[name])
           for name, parts in frames.items()}
    if not datasets:
        return text + "\nNot available: no dataset of the transfer set with E3 rankings of the main campaign.\n", out
    missing, bias = out["missing_rule"], out["tau_bias"]
    check = rb.reproduction_check(missing, kendall)
    worst = float(np.nanmax(check[["tau", "low", "high"]].to_numpy(dtype=np.float64))) if len(check) else float("nan")
    text += ("\n### Missing-input rule and set of entries\n\n"
             "`last`: the rule of the composite (a missing input ranks last among the set); `skip`: every input is ranked "
             "among the entries of the set that have it and an entry is averaged over the inputs it has. Sets: all 25 "
             "entries, without the two baselines (23), and the entries with all four inputs (`complete`; the inputs missing "
             "are P3 unstable fractions, listed per dataset). Replicate r recomputes the composite with the treatment from "
             "the r-th replicates of the inputs of the set. `low`/`high`: 95 % percentile interval; `share_below`: share of "
             "replicate taus strictly below the point tau; `BC`: bias-corrected percentile interval (z0 from that share). "
             f"The `published` rows reproduce tau, low and high of the P0-composite rows of E3 (largest absolute difference "
             f"{worst:.1e}).\n\n")
    text += table(missing.assign(ci=_ci(missing, "low", "high"), bc=_ci(missing, "bc_low", "bc_high")),
                  ["dataset", "treatment", "n_models", "n_missing_stability", "tau", "p", "ci", "tau_boot_median", "share_below",
                   "bc", "top3"], {"ci": "95% CI", "bc": "BC 95% CI", "tau_boot_median": "median"})
    lacking = missing[missing["treatment"] == "published"]
    text += "\nEntries without the stability input: " + "; ".join(
        f"{d}: {m}" for d, m in zip(lacking["dataset"], lacking["missing_stability"])) + ".\n"
    text += ("\n### Point estimate and bootstrap distribution of the published taus\n\n"
             "The same statistics for tau(P0, composite), tau(P0, P3 unstable) and tau(P1 whole, composite) of E3 (all "
             "entries, a missing value last): where the point tau lies at the edge of the percentile interval, the share "
             "of replicates below it and the bias-corrected interval show the asymmetry of the bootstrap distribution.\n\n")
    text += table(bias.assign(ci=_ci(bias, "low", "high"), bc=_ci(bias, "bc_low", "bc_high")),
                  ["dataset", "ranking_a", "ranking_b", "tau", "ci", "tau_boot_mean", "tau_boot_median", "share_below", "bc"],
                  {"ci": "95% CI", "bc": "BC 95% CI", "tau_boot_mean": "mean", "tau_boot_median": "median"})
    gate = out["collision_gate"]
    text += ("\n### Collision gate\n\n"
             "An entry is admissible when its share of colliding followers in the in-range P2 profiles, "
             "`collisions_100 / (100 n)` with n = `n_profiles_in_range` of `p2.json` (the synthetic braking pulse included, as "
             "in the collision input), is at most theta (no collision value: inadmissible). Gated score = composite (`last` "
             "rule on the set) + 1 for an inadmissible entry: every admissible entry ranks before every inadmissible one and "
             "the composite orders each group. The gate is re-evaluated in every replicate from the replicate collisions "
             "(the denominator stays 100 n).\n\n")
    text += table(gate.assign(ci=_ci(gate, "low", "high")),
                  ["dataset", "set", "theta", "n_models", "n_followers", "n_admissible", "top3", "tau", "p", "ci"], {"ci": "95% CI"})
    variants, per_entry = out["collision_input"], out["collision_input_entries"]
    text += ("\n### The collision input\n\n"
             "The collision input counts the followers whose own gap closes (`followers`); a collided vehicle keeps driving. "
             "Followers that do not react to their leader (persistence, constant speed) therefore count one collision per "
             "profile, that of the first follower, and the input rewards the absence of a reaction. Two replacements from "
             "`p2.vehicles.parquet` (in-range profiles, 100 followers): `blocked`, the followers at or behind the first "
             "colliding one, and `profiles`, the profiles with at least one collision. tau(P0, composite) with the input "
             "replaced (`last` rule), over all entries and without the baselines; the profile draws of the replicates are "
             "those of E3 (the `followers` rows reproduce the published composite).\n\n")
    text += table(variants.assign(ci=_ci(variants, "low", "high")),
                  ["dataset", "set", "input", "n_models", "tau", "p", "ci", "tau_boot_median", "top3"],
                  {"ci": "95% CI", "tau_boot_median": "median"})
    baselines = per_entry[per_entry["family"] == "baseline"]
    text += ("\nThe baselines under the three measures (value summed over the in-range profiles; composite rank among all "
             "entries); every entry: `e3f_collision_input_entries.csv`.\n\n")
    text += table(baselines, ["dataset", "model", *rb.COLLISION_INPUTS, *(f"rank_{k}" for k in rb.COLLISION_INPUTS)])
    rules = out["stability_rule"]
    text += ("\n### The stability verdict\n\n"
             "The protocol calls an equilibrium unstable when the maximal gain exceeds 1 + 1e-5 (`published`). A model that "
             "passes a perturbation on unchanged (a pure time shift such as Newell's: gain one at every frequency, 0.9995 at "
             "0.1 Hz in the discrete simulation) is stable under this rule. Two other rules from the saved gains of "
             "`p3.speeds.parquet`: `tolerance 1e-3` (maximal gain above 1 + 1e-3) and `neutral counts as unstable` (unstable "
             f"under the published rule, or a gain at 0.1 Hz above 1 - {rb.NEUTRAL_EPS:g}). tau(P0, composite) with the unstable "
             "fraction recomputed under each rule; `changed`: entries whose unstable fraction differs from the published one. "
             "The speed draws of the replicates are those of E3.\n\n")
    text += table(rules.assign(ci=_ci(rules, "low", "high")),
                  ["dataset", "set", "rule", "n_models", "n_changed", "changed", "tau", "p", "ci", "top3"], {"ci": "95% CI"})
    text += ("\n### Leave one entry out\n\n"
             "tau(P0, composite `last`) recomputed on the 24 other entries for each entry (point values; the composite is "
             "re-ranked on the 24). Per entry: `e3f_leave_one_out_entries.csv`.\n\n")
    text += table(out["leave_one_out"])
    inputs = out["input_ablation"]
    text += ("\n### Leave one input out\n\n"
             "tau(P0, composite) when one of the four inputs is left out of the composite: the mean normalised rank of the "
             "three others (a missing value last), over all entries and without the baselines; `none` is the composite "
             "itself. Replicate r recomputes the three-input composite from the r-th replicates of its inputs.\n\n")
    text += table(inputs.assign(ci=_ci(inputs, "low", "high")),
                  ["dataset", "set", "left_out", "n_models", "tau", "p", "ci", "tau_boot_median", "top3"],
                  {"ci": "95% CI", "tau_boot_median": "median"})
    text += ("\n### Training seeds and folds\n\n"
             "Point tau(P0, composite `last`) on the main datasets for the runs of other seeds (fold 0) and other folds "
             "(seed 0) of the driver split, over the entries with `train.json`, `p2.json` and `p3.json` of that run (the "
             "composite recomputed on them; `missing` lists the others).\n\n")
    text += table(out["seed_fold"])
    text += ("\n### Statistics of the registered tests in the other training runs (exploratory; the verdicts are those of fold 0, "
             "seed 0)\n\n"
             "Point values, no intervals, for every run of the seed and fold check on the main datasets and, on `highd`, the run "
             "under the split by blocks of recordings (fold 0, seed 0; decision HD-5). These runs cannot change the verdicts. "
             "Entries: those with `train.json`, `p1.json`, `p2.json` and `p3.json` of that run (`missing`: the others). `tau`, "
             "`top3`: tau(P0, composite `last`) and the composite top three over all of them. H5.3: `p0_top3` and `whole_top3`, "
             "the three most accurate entries by the one-step error and by the whole-event closed-loop spacing error, and "
             "`h53_dropped_*`, how many of them are not in the composite top three. H5.1: over the `h51_n` entries that are not "
             "baselines, the ratio of the P0 error (`train.json`) to that of persistence in the same run: `h51_within10` "
             "(0.9 <= ratio <= 1.1), `h51_beyond30` (ratio < 0.7 or > 1.3; split into `h51_better30` and `h51_worse30`), and "
             "the smallest ratio. H5.4: tau between the rankings by spacing and by speed error of P1 at 10 s, 30 s and over the "
             "whole event (a missing value last). Every column: `e3f_run_statistics.csv`.\n\n")
    text += table(out["run_statistics"], ["dataset", "split", "fold", "seed", "n_models", "tau", "top3", "h53_dropped_p0",
                                          "h53_dropped_whole", "h51_n", "h51_within10", "h51_beyond30", "h51_best", "h51_best_ratio",
                                          "h54_tau_h10", "h54_tau_h30", "h54_tau_whole"])
    text += ("\n### Registered rule H5.3\n\n"
             "\"By the composite at least 2 of the 3 best models by trajectory RMSE drop out of the top three; refuted when the "
             "top threes coincide.\" `dropped`: entries of the trajectory top three (P0 or whole-event P1) not in the "
             "composite top three (`last` rule on the set; ties broken by model name); `p_boot_ge2` and `p_boot_coincide`: "
             "shares of replicates with dropped >= 2 and dropped = 0.\n\n")
    text += table(out["h53_rule"], ["dataset", "set", "trajectory", "n_models", "top3_trajectory", "top3_composite", "dropped",
                                    "p_boot_ge2", "p_boot_coincide"])
    h51 = out["h51_rule"]
    text += ("\n### Registered rule H5.1\n\n"
             "\"Persistence within 10 % of the one-step models; refuted by a gap above 30 %.\" From the P0 ratios of E1 "
             "(model / persistence, 95 % driver-bootstrap interval) over the 23 entries that are not baselines and over the "
             "learned entries only: `n_within_10` (0.9 <= ratio <= 1.1), `n_ci_within_10` (interval inside [0.9, 1.1]), "
             "`n_better_30` (ratio < 0.7), `n_better_ci` (interval below 1), `n_worse_30` (ratio > 1.3).\n\n")
    text += table(h51.assign(ci=_ci(h51, "best_low", "best_high")),
                  ["dataset", "group", "n_models", "n_within_10", "n_ci_within_10", "n_better_30", "n_better_ci", "n_worse_30",
                   "best_model", "best_ratio", "ci"], {"ci": "best 95% CI"})
    return text, out


def section_e4(root: Path, n_boot: int, seed: int) -> tuple[str, pd.DataFrame]:
    text = ("\n## E4: transfer (P5)\n\n"
            "Relative degradation `rmse_s(A -> B) / rmse_s(B -> B)` of P1 for every label (rows: model and training "
            f"dataset A, columns: test dataset B; 1 = no loss); intervals from {n_boot} driver replicates over the test drivers of B "
            "(one draw per target, common to both tables), shown as `[low, high]` where the per-window tables of both evaluations exist.\n")
    labels = main_labels() or (*an.SUBSET, an.REFERENCE)  # every reference entry once the full matrix exists
    frame = an.e4_transfer(root, list(labels), n_boot, seed)
    if frame.empty:
        return text + "\nNot available: no `p1.json` of a SUBSET run.\n", frame
    sources = sorted(frame["trained_on"].unique())
    targets = sorted(frame["tested_on"].unique())
    if len(targets) < 2:
        text += "\nOnly one dataset with runs: no transfer pair.\n"
    missing = [d for d in an.TRANSFER_DATASETS if d not in sources]
    if missing:
        text += f"\nNo trained SUBSET runs on {', '.join(missing)} (campaign `transfer` has not run): the matrix covers {', '.join(sources)}.\n"
    foreign = frame[frame["trained_on"] != frame["tested_on"]]
    without = foreign["low"].isna().sum()
    if without:
        notes = foreign.loc[foreign["low"].isna(), "note"].iloc[0]
        text += (f"\n{without} of {len(foreign)} foreign entries have point values only: the per-window tables of the transfer "
                 f"evaluation (`p1_<B>.<h>.parquet`, written since decision M4-5) are missing or do not match their summary "
                 f"(e.g. {notes}).\n")
    for horizon in sorted(frame["horizon"].unique(), key=lambda h: an.HORIZONS.index(h) if h in an.HORIZONS else 9):
        part = frame[frame["horizon"] == horizon]
        cells = part.assign(cell=[
            f"{r:.3f} {interval(lo, hi)}" if np.isfinite(lo) and a != b else (f"{r:.3f}" if np.isfinite(r) else "n/a")
            for r, lo, hi, a, b in zip(part["ratio"], part["low"], part["high"], part["trained_on"], part["tested_on"])
        ])
        matrix = cells.pivot_table(index=["model", "trained_on"], columns="tested_on", values="cell", aggfunc="first").reset_index()
        matrix.columns.name = None
        text += f"\n### Horizon {horizon}\n\n" + table(matrix.fillna("n/a"))
    summary, targets_frame = an.e4_summary(frame), an.e4_targets(frame)
    if not summary.empty:
        text += "\nMean degradation per model over its foreign pairs:\n\n" + table(summary)
        text += "\nMean degradation per target dataset:\n\n" + table(targets_frame)
    return text, frame


def main(argv: list[str] | None = None) -> Path:
    args = arguments(argv)
    root, out, n_boot, seed = args["root"], args["out"], args["n_boot"], args["seed"]
    started = time.time()
    labels_of = label_filter(args["models"])
    scope = ("the reference models of the main campaign (`scripts/m2.py`) in E1 and E3" if main_labels() is not None and
             args["models"] == "main" else "every finished run in E1 and E3")
    text = ("# Milestone 4: experiments E1-E4\n\n"
            f"Generated by `scripts/m4_report.py` from `{_shown(root)}` ({n_boot} bootstrap replicates, seed {seed}; {scope}); definitions in "
            "`docs/experiments.md`, section 3. Tables: 3 decimals, `n/a` = not available. Per-window tables are used only when they "
            "reproduce their JSON summary (decision M4-5: in milestone 3 the transfer evaluation overwrote the own `p1.<h>.parquet`).\n")
    campaigns, _ = section_campaigns(root)
    e1_text, e1 = section_e1(root, n_boot, seed, labels_of)
    e1b_text, e1b, e1bt = section_e1b(root, labels_of)
    e2_text, e2, e2u = section_e2(root, n_boot, seed, args["splits"], args["events"])
    e3_text, e3, e3k, cache = section_e3(root, n_boot, seed, labels_of)
    e3b_text, e3v, e3s, e3p = section_e3b(root, cache, n_boot, seed)
    e3c_text, e3c = section_e3c(root)
    e3d_text, e3d, e3dt, e3dtop = section_e3d(root, labels_of)
    e3e_text, e3e = section_e3e(e3)
    e3f_text, e3f = section_e3f(root, n_boot, seed, e1, e3k)
    e3g_text, e3g_ranks, e3g_taus = section_e3g(e3)
    e3h_text, e3h = section_e3h(root, e1, e3)
    e4_text, e4 = section_e4(root, n_boot, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(text + campaigns + e1_text + e1b_text + e2_text + e3_text + e3b_text + e3c_text + e3d_text + e3e_text
                                   + e3f_text + e3g_text + e3h_text + e4_text, encoding="utf-8")
    for name, frame in (("e1_ratios", e1), ("e1_horizons", e1b), ("e1_horizon_taus", e1bt), ("e2_paired", e2), ("e2_unpaired", e2u), ("e3_rankings", e3), ("e3_kendall", e3k),
                        ("e3_composite_variants", e3v), ("e3_composite_sensitivity", e3s), ("e3_pareto", e3p),
                        ("e3_ablations", e3c), ("e3_human_reference", e3d), ("e3_human_taus", e3dt),
                        ("e3_human_top3", e3dtop), ("e3_within_family", e3e), ("e3_objective_ranks", e3g_ranks),
                        ("e3_objective_taus", e3g_taus), ("e4_transfer", e4),
                        *((f"e3f_{name}", frame) for name, frame in e3f.items()),
                        *((f"e3h_{name}", frame) for name, frame in e3h.items())):
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"report written to {out / 'REPORT.md'} ({time.time() - started:.0f} s): E1 {len(e1)} rows, E2 {len(e2)} paired and "
          f"{len(e2u)} unpaired rows, E3 {len(e3)} models and {len(e3k)} taus, E3b {len(e3s)} sensitivity rows, E4 {len(e4)} entries",
          flush=True)
    return out / "REPORT.md"


if __name__ == "__main__":
    main()
