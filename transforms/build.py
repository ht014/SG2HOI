# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = 600
        max_size = 800
        flip_horizontal_prob = 0  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = 0 #cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = 0.
        contrast = 0.
        saturation = 0.
        hue = 0.
    else:
        min_size = 600
        max_size = 800
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    normalize_transform = T.Normalize(
        mean=[102.9801, 115.9465, 122.7717], std=[0.26862954, 0.26130258, 0.27577711], to_bgr255=False
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.ToTensor(),
            # normalize_transform,
        ]
    )
    return transform
