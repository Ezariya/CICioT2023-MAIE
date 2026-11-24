@echo off
setlocal
echo ========================================================
echo   CICIoT2023 AI-Powered IoT IDS — Portable Launcher
echo ========================================================
echo.

where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [*] Starting IDS SOC Dashboard via Docker Compose...
echo [*] The dashboard will be accessible at: http://localhost:8501
echo.

docker compose up -d dashboard
if %ERRORLEVEL% neq 0 (
    echo [*] Docker Compose fallback: building and running container directly...
    docker build -t ciciot2023-ids-demo -f Dockerfile.demo .
    docker run --rm -d -p 8501:8501 --name ciciot2023-soc-dashboard ciciot2023-ids-demo
)

echo.
echo [OK] Dashboard container is running!
echo Opening http://localhost:8501 in your default browser...
start http://localhost:8501

echo.
echo To stop the dashboard later, run: docker compose down
echo.
pause
