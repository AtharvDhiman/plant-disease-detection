#!/usr/bin/env python3
"""Launcher for the PlantGuard AI FastAPI backend server.

Usage:
    python run_server.py
    python run_server.py --port 8000 --reload
    python run_server.py --host 0.0.0.0 --port 8080 --workers 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
MODELS_DIR = ROOT_DIR / "models" / "exported"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def check_environment() -> dict[str, str | bool]:
    """Check availability of models, frontend bundle, and acceleration libraries."""
    has_onnx = (MODELS_DIR / "serving.onnx").is_file()
    has_pt = list(MODELS_DIR.glob("*.pt")) or list((ROOT_DIR / "artifacts" / "checkpoints").glob("*.pt"))
    has_frontend = (FRONTEND_DIST / "index.html").is_file()

    torch_available = False
    try:
        import torch
        torch_available = True
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_name = "cpu (torch not installed)"

    onnx_available = False
    try:
        import onnxruntime  # noqa: F401
        onnx_available = True
    except ImportError:
        pass

    return {
        "has_onnx_model": has_onnx,
        "has_pt_model": bool(has_pt),
        "has_frontend": has_frontend,
        "torch_available": torch_available,
        "onnx_available": onnx_available,
        "device": device_name,
    }


def print_banner(host: str, port: int, reload: bool, env_info: dict[str, str | bool]) -> None:
    """Print an informative startup banner with quick links."""
    base_url = f"http://{('127.0.0.1' if host in ('0.0.0.0', '127.0.0.1') else host)}:{port}"
    border = "=" * 68

    print(f"\n{border}")
    print("  🌿  PlantGuard AI — Plant Disease Detection Backend Server")
    print(border)
    print(f"  • API Endpoint:     {base_url}/api")
    print(f"  • Interactive Docs: {base_url}/docs (Swagger UI)")
    print(f"  • ReDoc Schema:     {base_url}/redoc")
    print(f"  • Health Probe:     {base_url}/api/health")
    if env_info["has_frontend"]:
        print(f"  • Web Dashboard:    {base_url}/ (Single-Page App)")
    else:
        print("  • Web Dashboard:    [Not built yet - run 'npm run build' in frontend/]")
    print("--------------------------------------------------------------------")
    print(f"  • Active Device:    {env_info['device']}")
    print(f"  • PyTorch:          {'Available' if env_info['torch_available'] else 'Not installed'}")
    print(f"  • ONNX Runtime:     {'Available' if env_info['onnx_available'] else 'Not installed'}")
    print(f"  • Auto-reload:      {'Enabled' if reload else 'Disabled'}")
    print(f"{border}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection Backend Server Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Network interface to bind to (use 0.0.0.0 for public / LAN access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on code changes (recommended for development)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (keep 1 when memory is constrained)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "onnx"],
        default=None,
        help="Override model serving backend (default: automatic selection)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Server logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend:
        os.environ["PDD_SERVING_BACKEND"] = args.backend

    env_info = check_environment()
    print_banner(args.host, args.port, args.reload, env_info)

    try:
        import uvicorn
    except ImportError:
        print("[ERROR] 'uvicorn' is not installed in the current Python environment.", file=sys.stderr)
        print("Install it via: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            app_dir=str(BACKEND_DIR),
        )
    except KeyboardInterrupt:
        print("\n[INFO] Backend server stopped by user.")


if __name__ == "__main__":
    main()
