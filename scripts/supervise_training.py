"""Run a training suite to completion, restarting it if the process dies.

Why this exists
---------------
The development machine has 7.35 GB of RAM against a ~31 GB commit limit, and a
multi-hour training sweep occasionally has its process terminated by the OS with
no traceback - the tell-tale sign being an empty stderr and a log that simply
stops. Rather than babysit the run, this supervisor exploits a property the
pipeline already has: ``run_experiments.py`` is **resumable**. Every completed
configuration is in the ledger and is skipped on the next start, so relaunching
after a death loses at most the model that was in flight.

The supervisor stops when every expected experiment is in the ledger, or when
the attempt budget is exhausted, and reports exactly which models are still
missing either way.

    python scripts/supervise_training.py --suite research
    python scripts/supervise_training.py --suite production --max-attempts 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import app.core.runtime  # noqa: F401,E402  # isort:skip

from app.core.config import settings  # noqa: E402
from app.ml.architectures import MODEL_ZOO  # noqa: E402

SUITE_GROUPS = {
    "benchmark": ("benchmark",),
    "ablation": ("ablation",),
    "placement": ("placement",),
    "research": ("benchmark", "ablation", "placement"),
}


def expected_models(suite: str, explicit: list[str] | None) -> list[str]:
    if explicit:
        return explicit
    if suite == "production":
        # Read the list from run_experiments rather than keeping a second copy
        # here. The two drifted once already: the production suite grew from two
        # models to four and this function still reported two, so the supervisor
        # would have declared the sweep finished with half of it untrained.
        from run_experiments import SUITES
        return list(SUITES["production"]["models"])
    groups = SUITE_GROUPS[suite]
    return [name for name, spec in MODEL_ZOO.items() if spec.group in groups]


def completed_models(protocol_filter: set[str]) -> set[str]:
    """Model names that already have a test-scored record in the ledger."""
    ledger_path = settings.results_dir / "experiments.json"
    if not ledger_path.exists():
        return set()
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {
        record["model_name"]
        for record in ledger
        if record.get("protocol") in protocol_filter
        and (record.get("test") or {}).get("metrics", {}).get("accuracy") is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="research",
                        choices=[*SUITE_GROUPS, "production"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cooldown", type=int, default=20,
                        help="Seconds to wait between attempts, letting memory settle.")
    args = parser.parse_args()

    settings.ensure_dirs()
    wanted = set(expected_models(args.suite, args.models))
    protocols = ({"benchmark", "ablation", "placement"} if args.suite == "research"
                 else {args.suite})

    log_path = settings.logs_dir / "supervisor.log"

    def report(message: str) -> None:
        stamped = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {message}"
        print(stamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    report(f"Supervising suite '{args.suite}' - {len(wanted)} model(s) expected")

    command = [sys.executable, "training/run_experiments.py", "--suite", args.suite]
    if args.models:
        command += ["--models", *args.models]
    if args.num_workers is not None:
        command += ["--num-workers", str(args.num_workers)]

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1",
           "PYTHONWARNINGS": "ignore"}

    for attempt in range(1, args.max_attempts + 1):
        done = completed_models(protocols)
        missing = sorted(wanted - done)
        if not missing:
            report(f"All {len(wanted)} model(s) complete after {attempt - 1} attempt(s).")
            return 0

        report(f"Attempt {attempt}/{args.max_attempts}: {len(done & wanted)}/{len(wanted)} done, "
               f"missing {missing}")
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        elapsed = time.perf_counter() - started

        after = completed_models(protocols)
        gained = sorted(after - done)
        report(f"Attempt {attempt} exited with code {result.returncode} after "
               f"{elapsed / 60:.1f} min; completed this attempt: {gained or 'none'}")

        if not gained and result.returncode != 0:
            # No forward progress and a failure: another identical attempt is
            # unlikely to help, so surface it rather than looping pointlessly.
            report("No progress on this attempt. Check artifacts/logs/suite_stderr.log "
                   "and artifacts/logs/experiments.log before retrying.")
            if attempt >= 2:
                break
        time.sleep(args.cooldown)

    done = completed_models(protocols)
    missing = sorted(wanted - done)
    if missing:
        report(f"Stopped with {len(missing)} model(s) still missing: {missing}")
        return 1
    report("All expected models are complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
