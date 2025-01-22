# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
from collections import defaultdict
from hashlib import sha256
import math
import os
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.hub as hub
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision.transforms.v2 as transforms

from datasets import load_dataset_builder, load_dataset
from datasets.iterable_dataset import DistributedConfig
from datasets.distributed import split_dataset_by_node

from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

from common import collate, round_up, get_standard_transform, get_rank, get_world_size, rank_print, load_model


class MixedCosineL1Loss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cos_criterion = nn.CosineEmbeddingLoss()
        self.smooth_l1_criterion = nn.SmoothL1Loss()

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        y1 = y1.reshape(y1.shape[0], -1)
        y2 = y2.reshape(y2.shape[0], -1)

        target = torch.ones(y1.shape[0], device=y1.device)

        return self.alpha * self.cos_criterion(y1, y2, target) * self.beta * self.smooth_l1_criterion(y1, y2)



def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='ZeroShot Classification Demo')

    parser.add_argument('-v', '--model-version', default='radio_v2',
                        help='Which radio model to load.'
    )
    parser.add_argument('-a', '--adaptor-name', default=None, help='Which head to use')
    parser.add_argument('-r', '--resolution', nargs='+', type=int, default=None,
                        help='The input image resolution.'
                             ' If one value is specified, the shortest dimension is resized to this.'
                             ' If two, the image is center cropped.'
                             ' If not specified, center cropped 378px is used.'
                             ' Default: The RADIO model\'s preferred resolution.'
    )
    parser.add_argument('-d', '--dataset', default='zh-plus/tiny-imagenet',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='train',
                        help='The dataset split to use.'
    )
    parser.add_argument('--resize-multiple', type=int, default=16,
                        help='Resize images with dimensions a multiple of this value.'
                             ' This should be equal to the patch size of a ViT (e.g. RADIOv1)'
    )
    parser.add_argument('--batch-size', type=int, default=16,
                        help='The batch size. If the input is variable sized, then this argument becomes a maximum.'
    )
    parser.add_argument('-w', '--workers', default=8, type=int,
                        help='The number of data loader workers to use per GPU'
    )
    parser.add_argument('--vitdet-window-size', default=None, type=int,
                        help='The ViTDet window size to use, if desired. Default: Global attention'
    )
    parser.add_argument('--use-local-lib', default=False, action='store_true',
                        help='Use the library locally, instead of through TorchHub'
    )
    parser.add_argument('--force-reload', default=False, action='store_true',
                        help='Force reload RADIO library'
    )
    parser.add_argument('--amp', default=False, action='store_true', help='Run in amp')
    parser.add_argument('--torchhub-repo',
                        help="Path to the Torchhub repo", default="NVlabs/RADIO"
    )
    parser.add_argument('-o', '--output', type=str, default='',
                        help='Output the histogram to the specified csv file')

    args, _ = parser.parse_known_args()

    summary_c = MixedCosineL1Loss(1, 0)
    feature_c = MixedCosineL1Loss(0.9, 0.1)
    rank_print('Loading model...')
    
    model_radio, preprocessor_radio, info_radio = load_model('radio_v2.5-l', adaptor_names=['clip','dino_v2'], return_spatial_features=False,
                                           vitdet_window_size=None, force_reload=False,
                                           torchhub_repo="NVlabs/RADIO")
    model_dino, preprocessor_dino, info_dino = load_model('dinov2_vitg14_reg', adaptor_names=None, return_spatial_features=False,
                                           vitdet_window_size=None, force_reload=False,
                                           torchhub_repo="NVlabs/RADIO")
    model_clip, preprocessor_clip, info_clip = load_model('open_clip,ViT-H-14-378-quickgelu,dfn5b', adaptor_names=None, return_spatial_features=False,
                                           vitdet_window_size=None, force_reload=False,
                                           torchhub_repo="NVlabs/RADIO")
    model_radio.to(device=device).train()
    model_dino.to(device=device).eval()
    model_clip.to(device=device).eval()

    opt = torch.optim.Adam(model_radio.parameters())

    # print(f"test random weight: {model.model.state_dict()['blocks.0.attn.qkv.weight']}")
    rank_print('Done')

    rank_print('Loading dataset...')
    ds_builder = load_dataset_builder(args.dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    num_examples = min(ds_builder.info.splits[args.split].num_examples, 1000000)

    transform_radio = get_standard_transform((432,432), 16, preprocessor=preprocessor_radio)
    transform_dino = get_standard_transform((224,224), 14, preprocessor=preprocessor_dino)
    transform_clip = get_standard_transform((378,378), 14, preprocessor=preprocessor_clip)
    dataset = ds_builder.as_dataset(split=args.split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, args.workers))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.shuffle(42)
    dataset = dataset.map(lambda ex: dict(image=transform_radio(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=args.workers > 0,
                        drop_last=False,
    )
    rank_print('Done')
    rank_print(f'Description: {ds_builder.info.description}')

    num_processed = 0
    with tqdm(total=num_examples, disable=rank > 0) as t:
        for batches in loader:
            for images, targets in batches:
                images_dino = transform_dino(images)
                images_clip = transform_clip(images)
                images = images.to(device=device, non_blocking=True)
                images_dino = images_dino.to(device=device, non_blocking=True)
                images_clip = images_clip.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                # print(f"input image shape: {images.shape}, targets shape: {targets.shape}")

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=False):
                    student = model_radio(images)
                    student_dino = student['dino_v2']
                    student_clip = student['clip']
                    with torch.no_grad():
                        teacher_dino = model_dino(images_dino)
                        teacher_clip = model_clip(images_clip)

                    # print(f"summary shape: {output[0].shape}")
                    # print(f"feature shape: {output[1].shape}")

                    loss = summary_c(student_dino[0], teacher_dino[0]) + summary_c(student_clip[0] + teacher_clip[0]) + feature_c(student_dino[1], teacher_dino[1]) + feature_c(student_clip[1] + teacher_clip[1])

                    opt.zero_grad()
                    loss.backward()
                    opt.step()






            num_processed += world_size * args.batch_size

            t.update(world_size * args.batch_size)
            if num_processed >= num_examples:
                break



if __name__ == '__main__':
    rank = 0
    world_size = 1

    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    main(rank, world_size)
