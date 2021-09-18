# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


from .roi_align import ROIAlign
from .roi_align import roi_align

__all__ = [
    "roi_align",
    "ROIAlign",
]

