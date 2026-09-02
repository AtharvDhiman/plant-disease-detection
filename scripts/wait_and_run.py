"""Wait for a process to finish, then run a command. For sequencing GPU jobs.

This machine cannot run two training jobs at once - the commit limit, not the
GPU, is the binding constraint - so long pipelines have to be sequenced rather
than parallelised. Launching this detached lets a multi-hour chain survive the
session that started it.

The waiting is deliberately conservative. It does not spawn a helper process to
check whether the target is alive, because that check would fail under exactly
the memory pressure it exists to wait out, and a failed check that reads as
"finished" starts the next job on top of the running one. An inconclusive answer
means keep waiting. After the target exits there is a cooldown, because worker
processes hold roughly a gigabyte of commit charge each and do not release it the
moment their parent returns.

    python scripts/wait_and_run.py --pid 8396 -- python scripts/run_pipeline.py --from classical
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_seed_replicates import process_alive, wait_for  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, action="append", default=[],
                        help="Wait for this PID to exit. May be repeated.")
    parser.add_argument("--cooldown", type=int, default=240,
                        help="Seconds to pause after the last PID exits.")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run once the wait is over, after `--`.")
    args = parser.parse_args()

    command = [part for part in args.command if part != "--"]
    if not command:
        raise SystemExit("No command given. Put it after `--`.")

    for pid in args.pid:
        if process_alive(pid) is False:
            print(f"PID {pid} is already gone.", flush=True)
            continue
        wait_for(pid, poll_seconds=args.poll, cooldown_seconds=args.cooldown,
                 what=command[0] if len(command) < 3 else " ".join(command[:3]) + " ...")

    print(f"\nRunning: {' '.join(command)}\n", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    print(f"\nFinished in {(time.perf_counter() - started) / 60:.1f} min "
          f"with exit code {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
