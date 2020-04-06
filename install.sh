#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n rsp_env python=3.7
conda activate rsp_env

# Dependencies
conda install -y numpy==1.18.1
conda install -y -c pytorch pytorch=1.4.0 torchvision cudatoolkit=10.1
pip install albumentations
pip install tifffile

# Model code
pip install git+https://github.com/cfld/amdim.git --ignore-installed

pip install -e .
