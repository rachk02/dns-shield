# Starconfiguret all DNS Shield microservices
Write-Host "Starting DNS Shield Services..." -ForegroundColor Green

$Python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
    Write-Host "Using virtual environment: $Python" -ForegroundColor Cyan
}

# 1. DGA Detector
Write-Host "Starting DGA Detector (Port 8001)..."
Start-Process -FilePath $Python -ArgumentList "-m src.dga_detector" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 2. BERT Service
Write-Host "Starting BERT Service (Port 8002)..."
Start-Process -FilePath $Python -ArgumentList "-m src.bert_service" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 3. Ensemble ML Service
Write-Host "Starting Ensemble ML Service (Port 8003)..."
Start-Process -FilePath $Python -ArgumentList "-m src.ensemble_ml" -WindowStyle Minimized
Start-Sleep -Seconds 2

# 4. API Gateway
Write-Host "Starting API Gateway (Port 9000)..."
Start-Process -FilePath $Python -ArgumentList "-m src.api_gateway"
