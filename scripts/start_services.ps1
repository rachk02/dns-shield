# Starconfiguret all DNS Shield microservices
Write-Host "Starting DNS Shield Services..." -ForegroundColor Green

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
    Write-Host "Using virtual environment: $Python" -ForegroundColor Cyan
}

# 1. DGA Detector
Write-Host "Starting DGA Detector (Port 8001)..."
Start-Process -FilePath $Python -ArgumentList "src/dga_detector.py" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 2. BERT Service
Write-Host "Starting BERT Service (Port 8002)..."
Start-Process -FilePath $Python -ArgumentList "src/bert_service.py" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 3. Ensemble ML Service
Write-Host "Starting Ensemble ML Service (Port 8003)..."
Start-Process -FilePath $Python -ArgumentList "src/ensemble_ml.py" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 4. API Gateway
Write-Host "Starting API Gateway (Port 9000)..."
Start-Process -FilePath $Python -ArgumentList "src/api_gateway.py"
