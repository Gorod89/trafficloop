"""The margin core of the certified hybrid as the exploratory entry ``idm_margin`` (decision REV4-1).

    python scripts/margin_core.py [datasets=ngsim,openacc] [build=true] [evaluate=false] [report=false] [force=false]

``build`` (per dataset; a dataset whose hybrid run does not exist yet is skipped): reads the margin core of
``residual_idm_certified`` (fold 0, seed 0, driver split), checks its kind and that ``theta`` of ``model.pt``
equals ``certificate.core.params``, writes the IDM with these parameters (no calibration, no training; test
scores as the trainer computes them) to ``results/exploratory/margin_core/models/<dataset>/reconstructed/
driver-fold0/idm_margin/seed0/`` and copies the stored ``idm`` run next to it as the control. Then prints the
``scripts/evaluate.py`` commands of both, or runs them with ``evaluate=true``. ``report=true`` writes
``results/exploratory/margin_core/comparison.csv`` (aggregate numbers only). ``force=true`` writes the entry
and the control again although they exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

from trafficloop.experiments import margin_core as mc
from trafficloop.utils import REPO_ROOT

DEFAULTS = {"datasets": ",".join(mc.DATASETS), "build": "true", "evaluate": "false", "report": "false",
            "force": "false", "root": mc.ROOT, "models_root": "results/models"}


def _flag(value: str) -> bool:
    if value.lower() not in ("true", "false"):
        raise SystemExit(f"expected true or false, got {value!r}")
    return value.lower() == "true"


def arguments(argv: list[str] | None = None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            usage = " ".join(f"[{k}=...]" for k in DEFAULTS)
            raise SystemExit(f"unknown argument {item!r}; usage: margin_core.py {usage}")
        args[key] = value
    args["datasets"] = tuple(d.strip() for d in args["datasets"].strip("[]").split(",") if d.strip())
    for key in ("build", "evaluate", "report", "force"):
        args[key] = _flag(args[key])
    return args


def command(dataset: str, label: str, root: str) -> str:
    """The ``scripts/evaluate.py`` command of one run (the list argument quoted for the shell)."""
    args = mc.evaluate_args(dataset, label, root)
    return "python scripts/evaluate.py " + " ".join(f'"{a}"' if "[" in a else a for a in args)


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    root, ready = args["root"], []
    for dataset in args["datasets"]:
        source = mc.run_dir(Path(REPO_ROOT) / args["models_root"], dataset, mc.HYBRID)
        if not (source / "train.json").exists():
            print(f"{dataset}: skipped, no {mc.HYBRID} run at {source} yet")
            continue
        if args["build"]:
            entry = mc.build_entry(dataset, args["models_root"], root, args["force"])
            control = mc.copy_control(dataset, args["models_root"], root, args["force"])
            print(f"{dataset}: {mc.LABEL} {entry['status']} ({entry['directory']}, "
                  f"config hash {entry['config_hash']}); control {control['status']} ({control['directory']})")
        ready.append(dataset)
    failed = []
    for dataset in ready:
        for label in (mc.LABEL, mc.CONTROL):
            if args["evaluate"]:
                from trafficloop.experiments.sweep import call_script

                status = call_script("evaluate.py", mc.evaluate_args(dataset, label, root))
                if status is not None:
                    failed.append(f"{dataset}/{label} ({status})")
            else:
                print(command(dataset, label, root))
    if failed:
        print("evaluation failed: " + ", ".join(failed))
    if args["report"]:
        print(f"comparison: {mc.write_comparison(args['models_root'], root, args['datasets'])}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
