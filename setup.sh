#!/bin/bash
# Installation script for degeneracy_distillery

set -e  # Exit on error

IS_APPLE_ARM64=0
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    IS_APPLE_ARM64=1
fi

PIP_EDITABLE_TARGET="-e ."
if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
    PIP_EDITABLE_TARGET="-e .[metal]"
fi

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
        echo "Choose environment file:"
        echo "  a) degen_env_minimal.yml (recommended, better compatibility)"
        echo "  b) degen_env.yml (full export, may have platform-specific issues)"
        read -p "Enter choice [a/b]: " env_choice
        
        env_file="degen_env_minimal.yml"
        if [[ $env_choice =~ ^[Bb]$ ]]; then
            env_file="degen_env.yml"
        fi
        
        echo ""
        echo "Creating conda environment from ${env_file}..."
        if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
            echo "Apple Silicon detected. Forcing osx-arm64 conda packages..."
            CONDA_SUBDIR=osx-arm64 conda env create -f ${env_file}
        else
            conda env create -f ${env_file}
        fi
        
        echo ""
        echo "Environment created successfully!"
        echo "Activating environment..."
        
        # Source conda to make activate available
        eval "$(conda shell.bash hook)"
        conda activate degen

        if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
            py_arch=$(python -c "import platform; print(platform.machine())")
            if [[ "${py_arch}" != "arm64" ]]; then
                echo "ERROR: Activated Python is '${py_arch}', expected 'arm64' on Apple Silicon."
                echo "Please remove/recreate the environment from a native ARM terminal (not Rosetta)."
                exit 1
            fi
        fi
        
        echo ""
        echo "Installing package in editable mode (${PIP_EDITABLE_TARGET})..."
        pip install ${PIP_EDITABLE_TARGET}
        
        echo ""
        echo "Installing ESR package (required dependency)..."
        if [ -d "ESR" ]; then
            echo "ESR directory already exists. Updating..."
            cd ESR && git pull && cd ..
        else
            git clone https://github.com/DeaglanBartlett/ESR.git
        fi
        pip install -e ESR
        echo "✓ ESR installed successfully"
        
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
            if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
                echo "Apple Silicon detected. Forcing osx-arm64 conda packages..."
                CONDA_SUBDIR=osx-arm64 conda create -n ${env_name} python=3.12 -y
            else
                conda create -n ${env_name} python=3.12 -y
            fi
        fi
        
        # Source conda and activate
        eval "$(conda shell.bash hook)"
        conda activate ${env_name}

        if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
            py_arch=$(python -c "import platform; print(platform.machine())")
            if [[ "${py_arch}" != "arm64" ]]; then
                echo "ERROR: Activated Python is '${py_arch}', expected 'arm64' on Apple Silicon."
                echo "Please remove/recreate the environment from a native ARM terminal (not Rosetta)."
                exit 1
            fi
        fi
        
        echo ""
        echo "Installing system dependencies with conda..."
        if [[ ${IS_APPLE_ARM64} -eq 1 ]]; then
            CONDA_SUBDIR=osx-arm64 conda install -c conda-forge eigen cmake -y
        else
            conda install -c conda-forge eigen cmake -y
        fi
        
        echo ""
        echo "Installing package with pip (${PIP_EDITABLE_TARGET})..."
        pip install ${PIP_EDITABLE_TARGET}
        
        echo ""
        echo "Installing ESR package (required dependency)..."
        if [ -d "ESR" ]; then
            echo "ESR directory already exists. Updating..."
            cd ESR && git pull && cd ..
        else
            git clone https://github.com/DeaglanBartlett/ESR.git
        fi
        pip install -e ESR
        echo "✓ ESR installed successfully"
        
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
echo ""
echo "Verifying installation..."
python -c "import degeneracy_distillery; from degeneracy_distillery import training_loop_flatten; print('✓ Package imported successfully')" 2>&1 | grep -q "✓" && echo "✓ Installation verified" || echo "⚠ Warning: Import test had issues (this may be OK if dependencies are properly installed)"

echo ""
echo "For more information, see INSTALL.md"
