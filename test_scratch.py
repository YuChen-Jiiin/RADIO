from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio.radio_model import RADIOModel
from radio.enable_cpe_support import *
from radio.input_conditioner import get_default_conditioner

import timm
from pprint import pprint
# model_names = timm.list_models('vit*')
# pprint(model_names)

input_conditioner = get_default_conditioner()
patch_size = 16
max_resolution = 2048
preferred_resolution = (768, 768)
summary_idxs = torch.tensor([0,1])
num_cls_tokens = 2
register_multiple = None
num_registers = 2

model = RADIOModel(timm.create_model('vit_large_patch16_224', pretrained=True),
                   input_conditioner=input_conditioner,
                   patch_size=patch_size,
                   max_resolution=max_resolution,
                   preferred_resolution=preferred_resolution,
                   summary_idxs=summary_idxs,
                   )

enable_cpe(model.model,
           max_resolution,
           num_cls_tokens=num_cls_tokens,
           register_multiple=register_multiple,
           num_registers=num_registers)