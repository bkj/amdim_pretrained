import torch
import numpy as np

from rsp.data import load_patch
from rsp.amdim.inference import amdim

# --
# Load model

model = amdim(state_dict_path='weights/amdim/amdim_weights_dummy.pth')

# --
# Load example (preformatted) + run inference

img       = np.load('data/bigearthnet/S2A_MSIL2A_20170613T101031_0_54.npy')
img       = img.astype(np.float32)
img_batch = np.stack([img] * 10)
img_batch = torch.FloatTensor(img_batch)

out = model(img_batch)
print(out[0])

# --
# Load example (raw) + run inference

img       = load_patch('data/bigearthnet/S2A_MSIL2A_20170613T101031_0_54')
img       = img.astype(np.float32)
img_batch = np.stack([img] * 10)
img_batch = torch.FloatTensor(img_batch)

out = model(img_batch)
print(out[0])
