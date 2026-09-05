#!/bin/bash

# setup_env_local.sh: Creates the clare_venv environment on a local machine (with GPU support)
echo "Setting up local clare_venv..."

# 1. Create the virtual environment
python -m venv clare_venv --clear

# 2. Activate the environment (supporting Windows Git Bash and Linux/macOS)
if [ -f "clare_venv/Scripts/activate" ]; then
    source clare_venv/Scripts/activate
elif [ -f "clare_venv/bin/activate" ]; then
    source clare_venv/bin/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

# 3. Upgrade core packaging tools
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# 4. Install dependencies with GPU (CUDA 12.8) support for local RTX GPU
echo "Installing PyTorch with CUDA 12.8 support..."
pip install "numpy>=1.26.0"
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "Installing remaining project dependencies..."
pip install matplotlib pandas scikit-learn scipy datasets tqdm wandb spacepy

# 5. Verification
echo ""
echo "----------------------------------------------------"
echo "Verifying local GPU acceleration..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo "----------------------------------------------------"
echo "Setup Complete!"
if [ -f "clare_venv/Scripts/activate" ]; then
    echo "To use this environment in Git Bash:     source clare_venv/Scripts/activate"
    echo "To use this environment in PowerShell:   .\\clare_venv\\Scripts\\Activate.ps1"
else
    echo "To use this environment:                 source clare_venv/bin/activate"
fi
echo "----------------------------------------------------"
