#!/usr/bin/env bash
# ========================================================
# CICIoT2023 AI-Powered IoT IDS — Linux & macOS Launcher
# ========================================================
set -e

echo "========================================================"
echo "  CICIoT2023 AI-Powered IoT IDS — SOC Dashboard Launcher"
echo "========================================================"
echo ""

if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not running."
    echo "Please install or start Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "[*] Launching SOC Dashboard container..."
echo "[*] Web URL: http://localhost:8501"
echo ""

if docker compose version &> /dev/null; then
    docker compose up -d dashboard
else
    docker build -t ciciot2023-ids-demo -f Dockerfile.demo .
    docker run --rm -d -p 8501:8501 --name ciciot2023-soc-dashboard ciciot2023-ids-demo
fi

echo ""
echo "[OK] Dashboard container is running at http://localhost:8501"

# Attempt to open browser if running in GUI environment
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8501" || true
elif command -v open &> /dev/null; then
    open "http://localhost:8501" || true
fi

echo "To stop: docker compose down"
