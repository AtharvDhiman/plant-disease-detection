#!/usr/bin/env bash
# Plant Disease Detection - Backend Server Launcher for Unix/Linux/macOS
echo "======================================================================"
echo "  Starting PlantGuard AI Backend Server..."
echo "======================================================================"
echo ""
python3 run_server.py --host 0.0.0.0 --port 8000 --reload
