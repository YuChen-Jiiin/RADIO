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




rank_print('Loading dataset...')
ds_builder = load_dataset_builder('zh-plus/tiny-imagenet', trust_remote_code=True)
print(ds_builder.info.splits['valid'].num_examples)
ds_builder.download_and_prepare()
num_examples = 1000

transform = get_standard_transform((432,432), 16)
dataset = ds_builder.as_dataset(split='train')
dataset = dataset.to_iterable_dataset(num_shards=8)
dataset = dataset.shuffle(42)
dataset = dataset.map(lambda ex: dict(image=transform(ex['image']), label=torch.as_tensor(ex['label'], dtype=torch.int64)))

loader = DataLoader(dataset, batch_size=2, shuffle=False,
                    num_workers=8, collate_fn=collate,
                    pin_memory=1,
                    drop_last=False,
)