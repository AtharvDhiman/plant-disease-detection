"""Execute the notebooks in place so their outputs are real, not empty shells.

``build_notebooks.py`` writes the notebook *sources*. A notebook checked in with
no outputs is a promise rather than a result: the reader cannot tell whether the
code runs, and a broken cell is invisible until someone opens it. Executing them
here turns each one into evidence, and makes a failure loud - the exit code is
non-zero and the offending cell is named.

Execution is sequential and single-kernel-at-a-time on purpose. Each notebook
imports torch and instantiates models; two kernels at once is enough to exhaust
the commit limit on this machine.

    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py --only 02 05
    python scripts/execute_notebooks.py --timeout 1200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def execute(path: Path, timeout: int) -> tuple[bool, str]:
    """Run one notebook in place; return success and a short message."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        # Notebooks live in notebooks/ and resolve paths relative to the project
        # root via a sys.path insert in their first cell, so the working
        # directory has to be the notebook's own folder for that to line up.
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        # Keep the notebook's partial outputs: seeing how far it got is the
        # fastest way to find why it stopped.
        nbformat.write(notebook, path)
        first_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else "cell failed"
        return False, first_line[:200]
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return False, f"{type(exc).__name__}: {exc}"[:200]

    nbformat.write(notebook, path)
    executed = sum(1 for c in notebook.cells
                   if c.cell_type == "code" and c.get("execution_count"))
    total = sum(1 for c in notebook.cells if c.cell_type == "code")
    return True, f"{executed}/{total} code cells executed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Notebook number prefixes to run, e.g. 02 05.")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-cell timeout in seconds.")
    args = parser.parse_args()

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if args.only:
        notebooks = [n for n in notebooks
                     if any(n.name.startswith(prefix) for prefix in args.only)]
    if not notebooks:
        raise SystemExit("No notebooks matched.")

    report = []
    failures = []
    for notebook in notebooks:
        print(f"\n{'=' * 78}\n>> {notebook.name}\n{'=' * 78}", flush=True)
        started = time.perf_counter()
        ok, message = execute(notebook, args.timeout)
        elapsed = time.perf_counter() - started
        status = "[ OK ]" if ok else "[FAIL]"
        print(f"{status} {notebook.name} - {message} ({elapsed:.0f}s)", flush=True)
        report.append({
            "notebook": notebook.name,
            "ok": ok,
            "detail": message,
            "seconds": round(elapsed, 1),
        })
        if not ok:
            failures.append(notebook.name)

    # A machine-readable record so the pipeline can verify this stage ran, and so
    # a reader can tell executed notebooks from unexecuted ones without opening
    # each file.
    report_path = PROJECT_ROOT / "artifacts" / "results" / "notebook_execution.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notebooks": report,
        "all_succeeded": not failures,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}", flush=True)

    if failures:
        print(f"\nFailed: {', '.join(failures)}", flush=True)
        return 1
    print(f"\nAll {len(notebooks)} notebook(s) executed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
