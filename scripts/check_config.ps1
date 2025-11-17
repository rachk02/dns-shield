#!/usr/bin/env pwsh
# Configuration Checker & Retriever Script
# Vérifie et affiche les configurations du système DNS Shield

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "DNS SHIELD - CONFIGURATION CHECKER" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan

# =============================================
# 1. PostgreSQL Configuration
# =============================================
Write-Host "`n[1] POSTGRESQL CONFIGURATION" -ForegroundColor Green
Write-Host "-" * 70

$pgHost = "localhost"
$pgPort = 5432
$pgDB = "dns_shield"
$pgUser = "postgres"

# Test PostgreSQL connection
try {
    # Vérifier service PostgreSQL
    $pgService = Get-Service PostgreSQL-x64-* -ErrorAction SilentlyContinue
    if ($pgService) {
        Write-Host "✓ PostgreSQL Service: $(if ($pgService.Status -eq 'Running') { 'RUNNING' } else { 'STOPPED' })" -ForegroundColor $(if ($pgService.Status -eq 'Running') { 'Green' } else { 'Red' })
    } else {
        Write-Host "⚠ PostgreSQL Service: NOT FOUND (check installation)" -ForegroundColor Yellow
    }
    
    # Test connection with psql
    $env:PGPASSWORD = 'postgres'
    $result = psql -h $pgHost -U $pgUser -t -c "SELECT NOW();" 2>$null
    
    if ($result) {
        Write-Host "✓ PostgreSQL Connection: SUCCESS" -ForegroundColor Green
        Write-Host "  Host: $pgHost" -ForegroundColor Gray
        Write-Host "  Port: $pgPort" -ForegroundColor Gray
        Write-Host "  Database: $pgDB" -ForegroundColor Gray
        Write-Host "  User: $pgUser" -ForegroundColor Gray
        
        # Vérifier si db dns_shield existe
        $dbExists = psql -h $pgHost -U $pgUser -l 2>$null | grep -c "dns_shield"
        if ($dbExists) {
            Write-Host "✓ Database 'dns_shield': EXISTS" -ForegroundColor Green
        } else {
            Write-Host "⚠ Database 'dns_shield': NOT CREATED" -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ PostgreSQL Connection: FAILED" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ PostgreSQL check failed: $_" -ForegroundColor Red
}

# =============================================
# 2. Redis Configuration
# =============================================
Write-Host "`n[2] REDIS CONFIGURATION" -ForegroundColor Green
Write-Host "-" * 70

$redisHost = "localhost"
$redisPort = 6379

try {
    # Test Redis connection
    $result = redis-cli -h $redisHost -p $redisPort ping 2>$null
    
    if ($result -eq "PONG") {
        Write-Host "✓ Redis Connection: SUCCESS" -ForegroundColor Green
        Write-Host "  Host: $redisHost" -ForegroundColor Gray
        Write-Host "  Port: $redisPort" -ForegroundColor Gray
        
        # Get Redis info
        $info = redis-cli -h $redisHost -p $redisPort INFO memory 2>$null
        $used_memory = $info | grep "used_memory_human" | Cut -d ':' -f2
        
        Write-Host "  Memory Usage: $used_memory" -ForegroundColor Gray
        
        # Get connected clients
        $clients = redis-cli -h $redisHost -p $redisPort INFO clients 2>$null | grep "connected_clients" | Cut -d ':' -f2
        Write-Host "  Connected Clients: $clients" -ForegroundColor Gray
    } else {
        Write-Host "✗ Redis Connection: FAILED" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Redis check failed: $_" -ForegroundColor Red
}

# =============================================
# 3. Docker Configuration
# =============================================
Write-Host "`n[3] DOCKER CONFIGURATION" -ForegroundColor Green
Write-Host "-" * 70

try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "✓ Docker: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "✗ Docker: NOT INSTALLED" -ForegroundColor Red
    }
    
    # Check Docker daemon
    $dockerPs = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
    
    if ($dockerPs) {
        Write-Host "✓ Docker Daemon: RUNNING" -ForegroundColor Green
        Write-Host "`n  Active Containers:" -ForegroundColor Gray
        
        $containers = @("dns-shield-redis", "dns-shield-prometheus", "dns-shield-grafana")
        foreach ($container in $containers) {
            $status = docker ps --filter "name=$container" --format "{{.Status}}" 2>$null
            if ($status) {
                Write-Host "  ✓ $container : $status" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $container : NOT RUNNING" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "✗ Docker Daemon: NOT RUNNING" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Docker check failed: $_" -ForegroundColor Red
}

# =============================================
# 4. Python Configuration
# =============================================
Write-Host "`n[4] PYTHON CONFIGURATION" -ForegroundColor Green
Write-Host "-" * 70

try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    
    # Vérifier venv
    if (Test-Path "C:\dns-shield\venv\Scripts\python.exe") {
        Write-Host "✓ Virtual Environment: EXISTS" -ForegroundColor Green
        
        # List installed packages
        $packages = @("flask", "transformers", "torch", "redis", "psycopg2-binary", "tensorflow")
        Write-Host "`n  Key Packages:" -ForegroundColor Gray
        
        foreach ($pkg in $packages) {
            $result = .\venv\Scripts\python -c "import $pkg; print('$pkg')" 2>$null
            if ($result) {
                Write-Host "  ✓ $pkg" -ForegroundColor Green
            } else {
                Write-Host "  ✗ $pkg : NOT INSTALLED" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "⚠ Virtual Environment: NOT FOUND" -ForegroundColor Yellow
    }
} catch {
    Write-Host "✗ Python check failed: $_" -ForegroundColor Red
}

# =============================================
# 5. Web Services Status
# =============================================
Write-Host "`n[5] WEB SERVICES STATUS" -ForegroundColor Green
Write-Host "-" * 70

$services = @(
    @{ Name = "Grafana"; URL = "http://localhost:3000"; Port = 3000 },
    @{ Name = "Prometheus"; URL = "http://localhost:9090"; Port = 9090 },
    @{ Name = "DGA Service"; URL = "http://localhost:8001"; Port = 8001 },
    @{ Name = "BERT Service"; URL = "http://localhost:8002"; Port = 8002 },
    @{ Name = "Ensemble Service"; URL = "http://localhost:8003"; Port = 8003 },
    @{ Name = "API Gateway"; URL = "http://localhost:9000"; Port = 9000 }
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 2 -ErrorAction Stop
        Write-Host "✓ $($service.Name): ONLINE" -ForegroundColor Green
    } catch {
        Write-Host "✗ $($service.Name): OFFLINE" -ForegroundColor Red
    }
}

# =============================================
# 6. Environment Variables
# =============================================
Write-Host "`n[6] ENVIRONMENT VARIABLES" -ForegroundColor Green
Write-Host "-" * 70

# Check if .env exists
if (Test-Path "C:\dns-shield\.env") {
    Write-Host "✓ .env file: EXISTS" -ForegroundColor Green
    
    # Load and display .env
    $envVars = @{}
    Get-Content "C:\dns-shield\.env" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $envVars[$matches[1]] = $matches[2]
        }
    }
    
    Write-Host "`n  Configuration Values:" -ForegroundColor Gray
    Write-Host "  FLASK_ENV: $($envVars['FLASK_ENV'])" -ForegroundColor Gray
    Write-Host "  POSTGRES_HOST: $($envVars['POSTGRES_HOST'])" -ForegroundColor Gray
    Write-Host "  POSTGRES_DB: $($envVars['POSTGRES_DB'])" -ForegroundColor Gray
    Write-Host "  REDIS_HOST: $($envVars['REDIS_HOST'])" -ForegroundColor Gray
    Write-Host "  REDIS_PORT: $($envVars['REDIS_PORT'])" -ForegroundColor Gray
} else {
    Write-Host "⚠ .env file: NOT FOUND" -ForegroundColor Yellow
    Write-Host "   Copy .env.example to .env and fill values" -ForegroundColor Yellow
}

# =============================================
# 7. Disk Space
# =============================================
Write-Host "`n[7] DISK SPACE" -ForegroundColor Green
Write-Host "-" * 70

$projectPath = "C:\dns-shield"
if (Test-Path $projectPath) {
    $size = (Get-ChildItem $projectPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "Project Size: $([Math]::Round($size, 2)) MB" -ForegroundColor Gray
    
    # Disk space
    $disk = Get-Volume -DriveLetter C
    $freeGB = $disk.SizeRemaining / 1GB
    $totalGB = $disk.Size / 1GB
    
    Write-Host "C: Drive Free: $([Math]::Round($freeGB, 2)) GB / $([Math]::Round($totalGB, 2)) GB" -ForegroundColor Gray
}

# =============================================
# SUMMARY
# =============================================
Write-Host "`n" + "=" * 70 -ForegroundColor Cyan
Write-Host "CONFIGURATION CHECK COMPLETE" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan

Write-Host "`nNext Steps:" -ForegroundColor Green
Write-Host "1. Verify all services above show ✓" -ForegroundColor Gray
Write-Host "2. If .env missing: copy .env.example to .env" -ForegroundColor Gray
Write-Host "3. Update .env values with your settings" -ForegroundColor Gray
Write-Host "4. Restart services after .env changes" -ForegroundColor Gray
Write-Host "5. Re-run this script to verify configuration" -ForegroundColor Gray