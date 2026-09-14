"""Tests for run_server.py and server.py entrypoints."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import run_server


def test_check_environment_structure():
    info = run_server.check_environment()
    assert isinstance(info, dict)
    assert "has_onnx_model" in info
    assert "has_pt_model" in info
    assert "has_frontend" in info
    assert "device" in info


def test_parse_args_defaults():
    with patch.object(sys, "argv", ["run_server.py"]):
        args = run_server.parse_args()
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.reload is False
        assert args.workers == 1


def test_parse_args_custom():
    test_argv = [
        "run_server.py",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--reload",
        "--workers", "2",
        "--backend", "onnx",
        "--log-level", "debug",
    ]
    with patch.object(sys, "argv", test_argv):
        args = run_server.parse_args()
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.reload is True
        assert args.workers == 2
        assert args.backend == "onnx"
        assert args.log_level == "debug"


def test_print_banner_runs(capsys):
    info = run_server.check_environment()
    run_server.print_banner("127.0.0.1", 8000, True, info)
    captured = capsys.readouterr()
    assert "PlantGuard AI" in captured.out
    assert "http://127.0.0.1:8000/api" in captured.out
    assert "http://127.0.0.1:8000/docs" in captured.out
