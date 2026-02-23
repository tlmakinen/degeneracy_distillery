#!/bin/bash
# Installation script for degeneracy_distillery

set -e  # Exit on error

echo "========================================"
echo "Degeneracy Distillery Installation"
echo "========================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Determine the installation method
echo "Choose installation method:"
echo "1) Full environment (recommended) - Creates environment from yml file"
echo "2) Quick install - Install to existing/new environment with pip"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "Creating conda environment from degen_env.yml..."
        conda env create -f degen_env.yml
        
        echo ""
        echo "Environment created successfully!"
        echo "Activating environment..."
        
        # Source conda to make activate available
        eval "$(conda shell.bash hook)"
        conda activate degen
        
        echo ""
        echo "Installing package in editable mode..."
        pip install -e .
        
        echo ""
        echo "========================================"
        echo "Installation complete!"
        echo "========================================"
        echo ""
        echo "To use the package, run:"
        echo "  conda activate degen"
        echo ""
        ;;
        
    2)
        echo ""
        read -p "Enter environment name (or press Enter for 'degen'): " env_name
        env_name=${env_name:-degen}
        
        # Check if environment exists
        if conda env list | grep -q "^${env_name} "; then
            echo "Environment '${env_name}' already exists."
            read -p "Use existing environment? [y/N]: " use_existing
            if [[ ! $use_existing =~ ^[Yy]$ ]]; then
                echo "Installation cancelled."
                exit 0
            fi
        else
            echo "Creating new environment '${env_name}' with Python 3.12..."
            conda create -n ${env_name} python=3.12 -y
        fi
        
        # Source conda and activate
        eval "$(conda shell.bash hook)"
        conda activate ${env_name}
        
        echo ""
        echo "Installing system dependencies with conda..."
        conda install -c conda-forge eigen cmake -y
        
        echo ""
        echo "Installing package with pip..."
        pip install -e .
        
        echo ""
        echo "========================================"
        echo "Installation complete!"
        echo "========================================"
        echo ""
        echo "To use the package, run:"
        echo "  conda activate ${env_name}"
        echo ""
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Verify installation
echo "Verifying installation..."
python -c "import sys; sys.path.insert(0, 'src'); from training_loop_flatten import *; print('✓ Package imported successfully')" || echo "⚠ Warning: Import test failed"

echo ""
echo "Note: ESR package will be automatically installed via pip"
echo ""
echo "For more information, see INSTALL.md"
