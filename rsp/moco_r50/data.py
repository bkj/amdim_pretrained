#!/usr/bin/env python

"""
    moco_r50/data.py
"""

import numpy as np

from albumentations import Compose as ACompose
from albumentations.pytorch.transforms import ToTensor as AToTensor
from albumentations.augmentations import transforms as atransforms

NAIP_BAND_STATS = {
    'mean' : np.array([0.38194386, 0.38695849, 0.35312921, 0.45349037])[None,None],
    'std'  : np.array([0.21740159, 0.18325207, 0.15651401, 0.20699527])[None,None],
}

def _naip_normalize(x, **kwargs):
    return (x - NAIP_BAND_STATS['mean']) / NAIP_BAND_STATS['std']

def naip_augmentation_valid():
    return ACompose([
        atransforms.Lambda(name='normalize', image=_naip_normalize),
        AToTensor(),
    ])

def load_patch(inpath):
    X = np.load(inpath)
    X = X[:4].transpose(1, 2, 0).astype(np.float32) / 255
    
    transform = naip_augmentation_valid()
    X = transform(image=X)['image']
    return X
