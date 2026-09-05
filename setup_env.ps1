# setup_env.ps1: Creates the clare_venv environment on Windows with GPU support
Write-Host "Setting up clare_venv for NVIDIA RTX GPU..." -ForegroundColor Cyan

# 1. Create the virtual environment
python -m venv clare_venv --clear

# 2. Upgrade pip and core packaging tools
Write-Host "Upgrading pip and build tools..." -ForegroundColor Cyan
.\clare_venv\Scripts\python -m pip install --upgrade pip setuptools wheel

# 3. Install CUDA 12.8 PyTorch for RTX 5080 and dependencies
Write-Host "Installing PyTorch with CUDA 12.8 for RTX 5080 GPU..." -ForegroundColor Yellow
.\clare_venv\Scripts\python -m pip install "numpy>=1.26.0"
.\clare_venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\clare_venv\Scripts\python -m pip install matplotlib pandas scikit-learn scipy datasets tqdm wandb spacepy

Write-Host "`n----------------------------------------------------" -ForegroundColor Green
Write-Host "Verifying GPU Acceleration in PyTorch..." -ForegroundColor Cyan
.\clare_venv\Scripts\python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
Write-Host "----------------------------------------------------" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To activate this environment in PowerShell, run:" -ForegroundColor Green
Write-Host "    .\clare_venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "----------------------------------------------------" -ForegroundColor Green
