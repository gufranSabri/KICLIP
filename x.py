from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 

import torch


model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

print(model)

x = torch.rand(1, 3, 8, 224, 224)

print(model(x).shape)
