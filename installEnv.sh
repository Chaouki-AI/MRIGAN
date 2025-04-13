#!/bin/bash

source ~/.bashrc

# Step 7: Create a new conda environment and install the required packages
echo "Creating the conda environment 'MRIGAN' with Python 3.9..."

# Create the environment
conda create -n MRIGAN python=3.9 -y

# Activate the environment
source activate MRIGAN

# Install PyTorch with CUDA support
echo "Installing PyTorch 1.13.0, torchvision, and CUDA 11.6..."
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y

# Install other required conda packages
echo "Installing iopath, nvidiacub, jupyter, and JupyterLab..."
conda install -c iopath iopath -y
conda install -c bottler nvidiacub -y
conda install jupyter -y
conda install jupyterlab -y

# Install additional Python packages via pip
echo "Installing Python packages with pip..."
pip install scikit-image matplotlib imageio plotly opencv-python

# Install linting and testing tools
echo "Installing testing/linting tools..."
conda install -c fvcore -c conda-forge fvcore -y
pip install black usort flake8 flake8-bugbear flake8-comprehensions tensorboard

# Install numpy with version less than 2
echo "Installing numpy <2..."
pip install "numpy<2"

# Additional installations
pip install pydicom
pip install easydict
pip install tensorboard
pip install tqdm


# Install mmcv using mim
echo "Installing mmcv via openmim..."
mim install mmcv

echo "Environment setup complete! You can now activate the environment using 'conda activate MRIGAN'."
