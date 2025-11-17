#!/usr/bin/env pwsh
# Cleanup Script - Remove old data, logs, and temporary files
# Usage: .\cleanup.ps1

Write-Host "=" * 60 -ForegroundColor Yellow
Write-Host "DNS Shield - Cleanup Script" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Yellow

$projectRoot = "C:\dns-shield"

# 1. Cleanup logs
Write-Host "`n[1] Cleaning logs..." -ForegroundColor Green
$logsDir = Join-Path $projectRoot "logs"
if (Test-Path $logsDir) {
    Get-ChildItem $logsDir -Include *.log -Recurse | Remove-Item -Force
    Write-Host "✓ Logs cleaned"
} else {
    Write-Host "- No logs directory"
}

# 2. Cleanup __pycache__
Write-Host "`n[2] Cleaning __pycache__..." -ForegroundColor Green
Get-ChildItem $projectRoot -Include __pycache__ -Recurse -Directory | 
    Remove-Item -Recurse -Force
Write-Host "✓ __pycache__ cleaned"

# 3. Cleanup .pytest_cache
Write-Host "`n[3] Cleaning .pytest_cache..." -ForegroundColor Green
Get-ChildItem $projectRoot -Include .pytest_cache -Recurse -Directory | 
    Remove-Item -Recurse -Force
Write-Host "✓ .pytest_cache cleaned"

# 4. Cleanup .pyc files
Write-Host "`n[4] Cleaning .pyc files..." -ForegroundColor Green
Get-ChildItem $projectRoot -Include *.pyc -Recurse | Remove-Item -Force
Write-Host "✓ .pyc files cleaned"

# 5. Cleanup Redis data (optional)
Write-Host "`n[5] Redis cleanup (optional)" -ForegroundColor Yellow
$response = Read-Host "Flush Redis cache? (y/n)"
if ($response -eq 'y') {
    redis-cli FLUSHALL
    Write-Host "✓ Redis flushed"
} else {
    Write-Host "- Skipped"
}

# 6. Cleanup database logs (optional)
Write-Host "`n[6] Database cleanup (optional)" -ForegroundColor Yellow
$response = Read-Host "Clear old database records (>30 days)? (y/n)"
if ($response -eq 'y') {
    $env:PGPASSWORD = 'dns_shield_password'
    $query = "DELETE FROM dns_queries WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days';"
    psql -h localhost -U dns_shield -d dns_shield -c $query
    Write-Host "✓ Old records deleted"
} else {
    Write-Host "- Skipped"
}

# 7. Show disk usage
Write-Host "`n[7] Disk usage:" -ForegroundColor Green
$projectSize = (Get-ChildItem $projectRoot -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Project size: $([Math]::Round($projectSize, 2)) MB"

# 8. Show summary
Write-Host "`n" + "=" * 60 -ForegroundColor Yellow
Write-Host "✓ Cleanup complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Yellow