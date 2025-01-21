from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
# model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
# model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
#model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
#model_version="e-radio_v2" # for E-RADIO
model = torch.hub.load('/root/.cache/torch/hub/RADIO', 'radio_model', version=model_version, adaptor_names=['clip', 'dino_v2'], progress=True, skip_validation=True, source='local')
model.cuda().eval()

# print(getattr(model, 'model', None))
# print(getattr(model, 'radio_model', None))
# print(getattr(model, '_patch_size', None))
# print(getattr(model.model, 'patch_generator', None))
print(hasattr(model, 'num_summary_tokens'))
print(model.num_summary_tokens)
print(hasattr(model, 'num_cls_tokens'))
print(model.num_cls_tokens)
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

if "e-radio" in model_version:
    model.model.set_optimal_window_size(x.shape[2:]) #where it expects a tuple of (height, width) of the input image.

# RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.

output = model(x)
bb_summary, bb_features = output['backbone']
clip_summary, clip_features = output['clip']
dino_summary, dino_features = output['dino_v2']
print(f"{bb_summary.shape} {bb_features.shape}")
print(f"{clip_summary.shape} {clip_features.shape}")
print(f"{dino_summary.shape} {dino_features.shape}")

# summary, spatial_features = model(x)
# print(f"{summary.shape} {spatial_features.shape}")

# # By default, RADIO will return the spatial_features in NLC format, with L being a combined height/width dimension.
# # You can alternatively ask for the features in the more computer-vision-convenient format NCHW the following way:
# summary, spatial_features = model(x, feature_fmt='NCHW')
# assert spatial_features.ndim == 4

# # RADIO also supports running in mixed precision:
# with torch.autocast('cuda', dtype=torch.bfloat16):
#     summary, spatial_features = model(x)

# # If you'd rather pre-normalize the inputs, then you can do this:
# conditioner = model.make_preprocessor_external()

# # Now, the model won't change the inputs, and it's up to the user to call `cond_x = conditioner(x)` before
# # calling `model(cond_x)`. You most likely would do this if you want to move the conditioning into your
# # existing data processing pipeline.
# with torch.autocast('cuda', dtype=torch.bfloat16):
#     cond_x = conditioner(x)
#     summary, spatial_features = model(cond_x)