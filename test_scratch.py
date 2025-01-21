from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from radio.radio_model import RADIOModel
from radio.enable_cpe_support import enable_cpe
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

model.cuda().eval()

# print(getattr(model, 'model', None))
# print(getattr(model, 'radio_model', None))
# print(getattr(model, '_patch_size', None))
# print(getattr(model.model, 'patch_generator', None))
# print(hasattr(model, 'num_summary_tokens'))
# print(model.num_summary_tokens)
# print(hasattr(model, 'num_cls_tokens'))
# print(model.num_cls_tokens)
# print(model.model.global_pool)
# print(model.summary_idxs)
# print(model.model.__dict__)

# print(model.model)
print(model.input_conditioner.__dict__)
print(model.patch_size)
print(model.max_resolution)
print(model.preferred_resolution)
print(model.summary_idxs)
print(model.window_size)
print(model.adaptors)
print(model.feature_normalizer)
print(model.inter_feature_normalizer)

print(model.model.num_cls_tokens)
print(model.model.num_registers)

for name, module in model.adaptors.named_children():
    print(f"Module {name}: {module}")

x = Image.open('assets/radio.png').convert('RGB')
x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension

print(x.shape)

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
x = F.interpolate(x, (384, 384), mode='bilinear', align_corners=False)

print(x.shape)


# RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
summary, spatial_features = model(x)
print(f"{summary.shape} {spatial_features.shape}")

# By default, RADIO will return the spatial_features in NLC format, with L being a combined height/width dimension.
# You can alternatively ask for the features in the more computer-vision-convenient format NCHW the following way:
summary, spatial_features = model(x, feature_fmt='NCHW')
assert spatial_features.ndim == 4

# RADIO also supports running in mixed precision:
with torch.autocast('cuda', dtype=torch.bfloat16):
    summary, spatial_features = model(x)

# If you'd rather pre-normalize the inputs, then you can do this:
conditioner = model.make_preprocessor_external()

# Now, the model won't change the inputs, and it's up to the user to call `cond_x = conditioner(x)` before
# calling `model(cond_x)`. You most likely would do this if you want to move the conditioning into your
# existing data processing pipeline.
with torch.autocast('cuda', dtype=torch.bfloat16):
    cond_x = conditioner(x)
    summary, spatial_features = model(cond_x)