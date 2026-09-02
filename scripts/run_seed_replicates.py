"""Train seed replicates of one architecture to measure the real noise floor.

Why this exists
---------------
The ablation compares architectures that differ only in their attention block,
and asks whether the resulting deltas are real. Answering that needs a threshold,
and the obvious threshold is wrong.

Repeating an *identical* configuration bounds how much the result moves when
nothing about the model changes. That is a genuine measurement, but it holds the
initialisation fixed, and the arms being compared do not: a CNN and a CNN+CBAM
have different parameter shapes, so they draw different starting weights from the
same seed. Every delta between two arms therefore carries initialisation and
data-order variance that an identical-configuration repeat excludes by
construction.

Retraining one architecture at several seeds measures exactly that missing
component. Everything that differs between two architectures now varies - initial
weights, augmentation draws, batch composition - except the architecture itself,
so the spread across seeds is the smallest difference that can be attributed to
architecture rather than luck.

The runs are recorded in the normal experiment ledger. ``analyse_results.py``
recognises them as replicates (same model name, non-canonical seed), keeps them
out of the comparison tables so one model does not appear several times in every
chart, and promotes the seed spread to the significance threshold.

Usage
-----
    python scripts/run_seed_replicates.py                     # cnn_baseline, seeds 43 44
    python scripts/run_seed_replicates.py --model cnn_se --seeds 43 44 45
    python scripts/run_seed_replicates.py --wait-for-pid 17304
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def process_alive(pid: int) -> bool | None:
    """Report whether `pid` is running. Returns None if that cannot be determined.

    This deliberately avoids spawning a helper process. The earlier version
    shelled out to PowerShell, which failed under exactly the condition this
    function exists to wait out - memory pressure from the job being waited on -
    and an empty result was read as "the process has exited". The replicates then
    started on top of a live training run and died importing torch with
    WinError 1455.

    ``OpenProcess`` with SYNCHRONIZE is a bare kernel call: no allocation, no
    child process, nothing to fail under load. A process that has exited but
    still has an open handle somewhere reports WAIT_OBJECT_0 on its handle, which
    distinguishes a zombie from a live process.
    """
    import ctypes
    from ctypes import wintypes

    SYNCHRONIZE = 0x00100000
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    WAIT_OBJECT_0 = 0x0
    ERROR_INVALID_PARAMETER = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)

    handle = kernel32.OpenProcess(
        SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        # Invalid parameter means no such process id. Anything else - typically
        # access denied - means the process exists but is not inspectable, so
        # report it as alive rather than guessing.
        return ctypes.get_last_error() != ERROR_INVALID_PARAMETER

    try:
        # Zero timeout: signalled means the process object has terminated.
        return kernel32.WaitForSingleObject(handle, 0) != WAIT_OBJECT_0
    finally:
        kernel32.CloseHandle(handle)


def wait_for(pid: int, poll_seconds: int = 60, cooldown_seconds: int = 180,
             what: str = "the next job") -> None:
    """Block until `pid` exits, then pause so its memory is actually released.

    The cooldown matters as much as the wait. Training workers hold roughly a
    gigabyte of commit charge each and do not release it the instant the parent
    returns; starting a fresh torch import into that window is what produced
    WinError 1455 the first time this ran.
    """
    print(f"Waiting for PID {pid} to finish before starting {what} ...", flush=True)
    unknown_polls = 0
    while True:
        alive = process_alive(pid)
        if alive is None:
            # Never treat an inconclusive check as permission to proceed.
            unknown_polls += 1
            print(f"  could not determine state of PID {pid} "
                  f"({unknown_polls} consecutive); continuing to wait", flush=True)
        elif not alive:
            break
        else:
            unknown_polls = 0
        time.sleep(poll_seconds)

    print(f"PID {pid} has exited; cooling down {cooldown_seconds}s so its memory "
          f"is released before importing torch.", flush=True)
    time.sleep(cooldown_seconds)


def run_with_retry(command: list[str], label: str, attempts: int = 3,
                   backoff_seconds: int = 240) -> bool:
    """Run `command`, retrying when it dies from commit-limit exhaustion.

    WinError 1455 happens at torch import, before any of the trainer's own
    worker-startup recovery can engage, so it has to be handled out here. It is
    also transient: the fix is to wait for whatever else is holding memory to
    release it, not to change the run.
    """
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False,
                                capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if result.returncode == 0:
            print(f"[ OK ] {label} finished in {elapsed:.0f}s", flush=True)
            return True

        stderr = result.stderr or ""
        sys.stderr.write(stderr)
        memory_failure = "1455" in stderr or "paging file is too small" in stderr.lower()
        print(f"[FAIL] {label} exited with {result.returncode} after {elapsed:.0f}s"
              f"{' (commit limit)' if memory_failure else ''}", flush=True)

        if not memory_failure or attempt == attempts:
            return False
        print(f"  memory exhaustion is transient; waiting {backoff_seconds}s before "
              f"attempt {attempt + 1}/{attempts}", flush=True)
        time.sleep(backoff_seconds)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cnn_baseline",
                        help="Architecture to replicate. Should be the ablation control, "
                             "since that is what every arm is compared against.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44],
                        help="Seeds to train in addition to the canonical run.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--wait-for-pid", type=int, default=None,
                        help="Wait for this PID to exit first (e.g. the training supervisor).")
    parser.add_argument("--cooldown", type=int, default=180,
                        help="Seconds to pause after the waited-for PID exits, so its "
                             "memory is released before torch is imported.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.wait_for_pid:
        wait_for(args.wait_for_pid, cooldown_seconds=args.cooldown,
                 what="the seed replicates")

    failures = []
    for seed in args.seeds:
        command = [
            sys.executable, "training/run_experiments.py",
            "--suite", "benchmark",
            "--models", args.model,
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
        ]
        print(f"\n{'=' * 78}\n>> {args.model} @ seed {seed}\n   {' '.join(command[1:])}\n{'=' * 78}",
              flush=True)
        if args.dry_run:
            continue
        if not run_with_retry(command, f"{args.model} @ seed {seed}"):
            failures.append(seed)

    if not args.dry_run:
        print("\nRegenerating analysis with the replicates included ...", flush=True)
        subprocess.run([sys.executable, "training/analyse_results.py"],
                       cwd=PROJECT_ROOT, check=False)
        subprocess.run([sys.executable, "scripts/generate_docs.py"],
                       cwd=PROJECT_ROOT, check=False)

    if failures:
        print(f"\nSeeds that failed: {failures}", flush=True)
        return 1
    print("\nREPLICATES_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
