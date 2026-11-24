# ========================================================
# CICIoT2023 AI-Powered IoT IDS — PowerShell Launcher
# ========================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  CICIoT2023 AI-Powered IoT IDS — SOC Dashboard Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please start or install Docker Desktop: https://www.docker.com/products/docker-desktop"
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host "[*] Launching SOC Dashboard container..." -ForegroundColor Yellow
Write-Host "[*] Web URL: http://localhost:8501" -ForegroundColor Gray

docker compose up -d dashboard
if ($LASTEXITCODE -ne 0) {
    Write-Host "[*] Falling back to direct docker run..." -ForegroundColor Yellow
    docker build -t ciciot2023-ids-demo -f Dockerfile.demo .
    docker run --rm -d -p 8501:8501 --name ciciot2023-soc-dashboard ciciot2023-ids-demo
}

Write-Host "`n[OK] Dashboard is live! Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:8501"

Write-Host "`nTo view logs: docker logs -f ciciot2023-soc-dashboard" -ForegroundColor Gray
Write-Host "To stop:      docker compose down`n" -ForegroundColor Gray
