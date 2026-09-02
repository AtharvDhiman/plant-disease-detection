"""Start both servers, capture the screenshots, then stop the servers.

``capture_screenshots.py`` assumes the application is already running, which
makes it awkward to chain behind a long training job. This wrapper owns the
whole operation: it starts the API and the dev server, waits until each actually
answers rather than sleeping a fixed interval, captures, and shuts both down in
a finally block so a failed capture does not leave orphaned servers holding a
port and a gigabyte of memory.

Nothing here is started if a server is already listening on its port - that
usually means the developer has one running deliberately, and killing it would
be rude.

    python scripts/capture_with_servers.py
    python scripts/capture_with_servers.py --dark
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PORT = 8000
FRONTEND_PORT = 5173


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening; cheaper than an HTTP probe."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_until_ready(url: str, timeout: int, label: str) -> bool:
    """Poll `url` until it answers or `timeout` elapses.

    Polling the real endpoint rather than sleeping a guessed interval matters
    here: the backend loads a model at startup, which takes far longer on a cold
    filesystem cache than on a warm one.
    """
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310
                if response.status < 500:
                    print(f"  {label} ready ({response.status})", flush=True)
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    print(f"  {label} did not become ready in {timeout}s: {last_error}", flush=True)
    return False


def start_backend() -> subprocess.Popen | None:
    if port_open(BACKEND_PORT):
        print(f"Backend already listening on {BACKEND_PORT}; leaving it alone.", flush=True)
        return None
    print("Starting the API ...", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--app-dir", "backend", "--port", str(BACKEND_PORT), "--log-level", "warning"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )


def start_frontend() -> subprocess.Popen | None:
    if port_open(FRONTEND_PORT):
        print(f"Frontend already listening on {FRONTEND_PORT}; leaving it alone.", flush=True)
        return None
    print("Starting the dev server ...", flush=True)
    # npm is a .cmd shim on Windows and cannot be executed directly.
    # --host 127.0.0.1 is not cosmetic: Vite otherwise binds "localhost", which
    # Windows resolves to ::1, and an IPv4 readiness probe is then refused even
    # though the server is up and printing its banner.
    return subprocess.Popen(
        ["cmd.exe", "/c", "npm", "run", "dev", "--",
         "--port", str(FRONTEND_PORT), "--host", "127.0.0.1", "--strictPort"],
        cwd=PROJECT_ROOT / "frontend",
    )


def stop(process: subprocess.Popen | None, label: str) -> None:
    """Stop a server and everything it spawned.

    `terminate()` alone is not enough on Windows: npm runs behind a cmd.exe
    shim, so terminating the shim orphans the node process, which keeps holding
    the port. `taskkill /T` walks the process tree, which is what actually frees
    it. Uvicorn gets the same treatment because its reloader can spawn children
    too.
    """
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {label} ...", flush=True)
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   capture_output=True, check=False)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    if port_open(BACKEND_PORT if label.endswith("API") else FRONTEND_PORT):
        print(f"  warning: a port is still in use after stopping {label}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dark", action="store_true",
                        help="Capture in the dark theme as well.")
    parser.add_argument("--startup-timeout", type=int, default=180)
    args = parser.parse_args()

    backend = frontend = None
    try:
        backend = start_backend()
        frontend = start_frontend()

        ok = wait_until_ready(f"http://127.0.0.1:{BACKEND_PORT}/api/health",
                              args.startup_timeout, "API")
        ok = wait_until_ready(f"http://127.0.0.1:{FRONTEND_PORT}/",
                              args.startup_timeout, "dev server") and ok
        if not ok:
            print("Servers did not come up; not capturing.", flush=True)
            return 1

        command = [sys.executable, "scripts/capture_screenshots.py",
                   "--base-url", f"http://127.0.0.1:{FRONTEND_PORT}"]
        if args.dark:
            command.append("--dark")
        print(f"\nCapturing: {' '.join(command[1:])}\n", flush=True)
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    finally:
        stop(frontend, "the dev server")
        stop(backend, "the API")


if __name__ == "__main__":
    sys.exit(main())
