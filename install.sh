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

# --
# Download models

function download_google_drive {
    SRC=$1
    DST=$2
    echo "$SRC -> $DST"
    
    curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$SRC" > /tmp/intermezzo.html
    DL_LINK=$(cat /tmp/intermezzo.html |\
        grep -Po 'uc-download-link" [^>]* href="\K[^"]*' |\
        sed 's/\&amp;/\&/g'
    )
    curl -L -b /tmp/cookies https://drive.google.com$DL_LINK > $DST    
}

# amdim
mkdir -p weights/amdim
download_google_drive 15ikQ_P5KTWzmW8KDw_8H3ToCYlPzET79 weights/amdim/amdim_weights_dummy.pth

# moco_r50
mkdir -p weights/moco_r50
download_google_drive 1TxmvNV6PDn_hlFVMWfU_Gg2uxP1LbhWo weights/moco_r50/checkpoint_0049.pth.tar