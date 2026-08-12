#!/bin/bash

# setup_env.sh: Creates the clare_venv environment
echo "Setting up clare_venv..."

# 1. NCAR-specific module loading
module reset
module load ncarenv/24.12
module load gcc           # Required to compile SpacePy's C backend
module load cuda/12.8.0   # For GPU support
module load conda

# 2. Create the virtual environment
python -m venv clare_venv --clear

# 3. Activate the environment
source clare_venv/bin/activate

# 4. Install packages
echo "Installing dependencies..."
pip install --upgrade pip

# Install numpy first so SpacePy can build against its C-headers
pip install numpy==1.24.4

# Install everything else from requirements.txt
pip install -r requirements.txt

echo "----------------------------------------------------"
echo "Setup Complete!"
echo "To use this environment, run: source clare_venv/bin/activate"
echo "----------------------------------------------------"