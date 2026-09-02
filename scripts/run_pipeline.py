"""Run the whole pipeline end to end, in order, with clear failure reporting.

Each stage is a separate process so a crash in one does not take the others'
state with it, and every stage is idempotent - re-running the pipeline resumes
rather than restarting.

    python scripts/run_pipeline.py                 # everything
    python scripts/run_pipeline.py --from train    # resume from the training stage
    python scripts/run_pipeline.py --only export docs
    python scripts/run_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    command: list[str]
    produces: str
    optional: bool = False


STAGES: list[Stage] = [
    Stage("probe", "Measure the hardware (AMP viability, throughput, VRAM)",
          ["scripts/probe_hardware.py"], "artifacts/results/hardware_probe.json"),
    Stage("download", "Resolve the dataset (downloads only if absent)",
          ["scripts/download_dataset.py"], "data/dataset_path.json"),
    Stage("inspect", "Inspect the dataset",
          ["training/dataset_inspect.py"], "artifacts/dataset_report.json"),
    Stage("splits", "Build the splits",
          ["training/prepare_splits.py"], "data/splits.json"),
    Stage("knowledge", "Build the disease knowledge base",
          ["scripts/build_disease_info.py"], "data/disease_info.json"),
    # Three workers, not the script's default of five: each one holds a copy of
    # the feature matrix, and this machine's commit limit is the binding
    # constraint. The RBF SVM is skipped because `probability=True` fits five
    # internal CV folds of an O(n^2) solver over 11,400 samples, which costs
    # hours; the brief's "SVM" requirement is met by the linear SVM, and the
    # omission is recorded in the results rather than hidden. Run it explicitly
    # with `python training/train_classical.py --models rbf_svm` if wanted.
    Stage("classical", "Train the classical baselines",
          ["training/train_classical.py", "--workers", "3", "--skip-rbf",
           "--xgb-device", "cpu"],
          "artifacts/results/classical_results.json"),
    Stage("research", "Train the benchmark, ablation and placement suites",
          ["training/run_experiments.py", "--suite", "research"],
          "artifacts/results/experiments.json"),
    Stage("tuning", "Coarse hyper-parameter search on the proposed model",
          ["training/tune_hyperparameters.py"], "artifacts/results/tuning_results.json",
          optional=True),
    Stage("production", "Train the production models on the full split",
          ["training/run_experiments.py", "--suite", "production"],
          "artifacts/results/experiments.json", optional=True),
    Stage("analyse", "Chart the results and summarise the ablation",
          ["training/analyse_results.py"], "artifacts/results/summary.json"),
    Stage("export", "Select and export the production model",
          ["training/export_model.py", "--criterion", "f1_macro"],
          "models/exported/production.json"),
    Stage("ood", "Calibrate the out-of-distribution thresholds",
          ["training/calibrate_ood.py"], "models/metadata/ood_thresholds.json"),
    Stage("serving", "Measure end-to-end serving performance",
          ["scripts/benchmark_serving.py"], "artifacts/results/serving_benchmark.json"),
    Stage("docs", "Regenerate the results documentation",
          ["scripts/generate_docs.py"], "docs/results.md"),
    Stage("notebooks", "Regenerate the notebooks",
          ["scripts/build_notebooks.py"], "notebooks/01_dataset_analysis.ipynb"),
    # Building a notebook only writes its source. Executing it is what turns the
    # file from a promise into a result, and makes a broken cell fail loudly
    # instead of sitting unnoticed in a checked-in file with no outputs.
    Stage("notebooks-run", "Execute the notebooks so their outputs are real",
          ["scripts/execute_notebooks.py"],
          "artifacts/results/notebook_execution.json"),
]


def run(stage: Stage, dry_run: bool, extra_env: dict,
        extra_args: list[str] | None = None) -> tuple[bool, float]:
    extra_args = extra_args or []
    printed = " ".join([*stage.command, *extra_args])
    print(f"\n{'=' * 78}\n>> {stage.title}\n   python {printed}\n{'=' * 78}")
    if dry_run:
        return True, 0.0

    started = time.perf_counter()
    env = {**os.environ, **extra_env}
    result = subprocess.run(
        [sys.executable, *stage.command, *extra_args], cwd=PROJECT_ROOT, env=env, check=False
    )
    elapsed = time.perf_counter() - started

    produced = PROJECT_ROOT / stage.produces
    if result.returncode != 0:
        print(f"\n[FAIL] {stage.title} exited with code {result.returncode} after {elapsed:.0f}s")
        return False, elapsed
    if not produced.exists():
        print(f"\n[FAIL] {stage.title} finished but did not produce {stage.produces}")
        return False, elapsed
    print(f"\n[ OK ] {stage.title} - {elapsed:.0f}s -> {stage.produces}")
    return True, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=[s.key for s in STAGES],
                        help="Resume from this stage.")
    parser.add_argument("--only", nargs="+", choices=[s.key for s in STAGES],
                        help="Run only these stages.")
    parser.add_argument("--dataset-path", default=None,
                        help="Use this dataset directory instead of downloading. "
                             "Passed to the resolve stage; no Kaggle credentials "
                             "or network access are used.")
    parser.add_argument("--skip", nargs="+", choices=[s.key for s in STAGES], default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going after a failing stage instead of stopping.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Passed through to the training stages.")
    args = parser.parse_args()

    selected = list(STAGES)
    if args.only:
        selected = [s for s in STAGES if s.key in args.only]
    elif args.start:
        index = next(i for i, s in enumerate(STAGES) if s.key == args.start)
        selected = STAGES[index:]
    selected = [s for s in selected if s.key not in args.skip]

    # The OpenMP workaround must be in the environment of every child process.
    extra_env = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1"}

    if args.num_workers is not None:
        selected = [
            Stage(s.key, s.title,
                  [*s.command, "--num-workers", str(args.num_workers)]
                  if "run_experiments.py" in s.command[0] else s.command,
                  s.produces, s.optional)
            for s in selected
        ]

    print(f"Pipeline: {len(selected)} stage(s)")
    for stage in selected:
        print(f"  - {stage.key:<11} {stage.title}")

    results = []
    total = 0.0
    for stage in selected:
        # Only the resolve stage understands --dataset-path.
        stage_args = (["--path", args.dataset_path]
                      if stage.key == "download" and args.dataset_path else [])
        ok, elapsed = run(stage, args.dry_run, extra_env, stage_args)
        results.append((stage, ok, elapsed))
        total += elapsed
        if not ok and not stage.optional and not args.continue_on_error:
            print("\nStopping. Fix the failure above, then resume with:")
            print(f"  python scripts/run_pipeline.py --from {stage.key}")
            break

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for stage, ok, elapsed in results:
        print(f"  {('PASS' if ok else 'FAIL'):<5} {stage.key:<11} {elapsed:>7.0f}s  {stage.title}")
    print(f"{'=' * 78}\nTotal: {total / 60:.1f} min\n")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
