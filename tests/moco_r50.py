#!/usr/bin/env python

"""
    tests/moco_r50.py
"""

import torch
import numpy as np

from rsp.moco_r50.data import load_patch
from rsp.moco_r50.inference import moco_r50

# --
# Load model

model = moco_r50(state_dict_path='weights/moco_r50/moco_naip_v0.pth.tar')
model = model.eval()

# --
# Load example (preformatted) + run inference

img = load_patch('data/naip/dr7dpc.npy')
img = img[None] # unsqueeze first dimension for pytorch

with torch.no_grad():
    out = model(img)

print(out[0])
