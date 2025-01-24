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
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter


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

        return self.alpha * self.cos_criterion(y1, y2, target) + self.beta * self.smooth_l1_criterion(y1, y2)



def build_loader(dataset: str, split: str, shuffle=False, upper_bound = 100000, workers=8, world_size=1, rank=0, batch_size=32, **kwargs):
    ds_builder = load_dataset_builder(dataset, trust_remote_code=True)
    ds_builder.download_and_prepare()
    num_examples = min(ds_builder.info.splits[split].num_examples, upper_bound)

    preprocessor_radio = kwargs.get("preprocessor_radio", None)
    preprocessor_dino = kwargs.get("preprocessor_dino", None)
    preprocessor_clip = kwargs.get("preprocessor_clip", None)

    transform_radio = get_standard_transform((432,432), 16, preprocessor=preprocessor_radio)
    transform_dino = get_standard_transform((224,224), 14, preprocessor=preprocessor_dino)
    transform_clip = get_standard_transform((378,378), 14, preprocessor=preprocessor_clip)
    dataset = ds_builder.as_dataset(split=split)
    dataset = dataset.to_iterable_dataset(num_shards=world_size * max(1, workers))
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    dataset = dataset.shuffle(42) if shuffle == True else dataset
    dataset = dataset.map(lambda ex: dict(
        image=transform_radio(ex['image']),
        image_dino=transform_dino(ex['image']),
        image_clip=transform_clip(ex['image']),
        label=torch.as_tensor(ex['label'], dtype=torch.int64)
    ))


    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate,
                        pin_memory=workers > 0,
                        drop_last=False,
    )
    rank_print('Done')
    rank_print(f'Description: {ds_builder.info.description}')

    return loader, num_examples

@torch.no_grad()
def calc_valid_loss(model, model_dino, model_clip, valid_loader, summary_c, feature_c, batch_size, device, num_examples):
    model.eval()
    loss = 0
    batch_num = 0
    num_processed = 0
    for batches in valid_loader:
        for images, images_dino, images_clip, targets in batches:
            # images_dino = transform_dino(images)
            # images_clip = transform_clip(images)
            # print("doing valid...")
            images = images.to(device=device, non_blocking=True)
            images_dino = images_dino.to(device=device, non_blocking=True)
            images_clip = images_clip.to(device=device, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            # print(f"input image shape: {images.shape}, targets shape: {targets.shape}")

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=False):
                student = model(images)
                student_dino = student['dino_v2']
                student_clip = student['clip']
                with torch.no_grad():
                    teacher_dino = model_dino(images_dino)
                    teacher_clip = model_clip(images_clip)

                # print(f"summary shape: {teacher_dino[0].shape}")
                # print(f"feature shape: {teacher_dino[1].shape}")
                dino_features_upsampled = F.interpolate(teacher_dino[1].reshape(batch_size, 16, 16, 1536).permute(0, 3, 1, 2), size=(27, 27), mode='bilinear')
                dino_features_upsampled = dino_features_upsampled.permute(0, 2, 3, 1).reshape(batch_size, 729, 1536)

                loss += summary_c(student_dino[0], teacher_dino[0]) + summary_c(student_clip[0], teacher_clip[0]) + feature_c(student_dino[1], dino_features_upsampled) + feature_c(student_clip[1], teacher_clip[1])
                # loss = summary_c(student_dino[0], teacher_dino[0]) + summary_c(student_clip[0], teacher_clip[0])
                batch_num += 1
        num_processed += batch_size
        if num_processed >= num_examples:
            break
    loss /= batch_num
    model.train()
    return loss.item()




def main(rank: int = 0, world_size: int = 1):
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    writer = SummaryWriter()

    device = torch.device('cuda', local_rank)
    parser = argparse.ArgumentParser(description='One GPU Train Demo')

    parser.add_argument('-d', '--dataset', default='zh-plus/tiny-imagenet',
                        help='The name of the dataset to classify'
    )
    parser.add_argument('--split', default='train',
                        help='The dataset split to use.'
    )
    parser.add_argument('--batch-size', type=int, default=32,
                        help='The batch size. If the input is variable sized, then this argument becomes a maximum.'
    )
    parser.add_argument('-w', '--workers', default=8, type=int,
                        help='The number of data loader workers to use per GPU'
    )


    args, _ = parser.parse_known_args()

    summary_c = MixedCosineL1Loss(1, 0).cuda()
    feature_c = MixedCosineL1Loss(0.9, 0.1).cuda()
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

    opt = torch.optim.AdamW(model_radio.parameters())
    

    # print(f"test random weight: {model.model.state_dict()['blocks.0.attn.qkv.weight']}")
    rank_print('Done')


    train_loader, num_examples = build_loader(args.dataset, args.split, True, preprocessor_radio=preprocessor_radio, preprocessor_dino=preprocessor_dino, preprocessor_clip=preprocessor_clip)
    valid_loader, _ = build_loader(args.dataset, 'valid', preprocessor_radio=preprocessor_radio, preprocessor_dino=preprocessor_dino, preprocessor_clip=preprocessor_clip)

    num_processed = 0
    valid_cnt = 0
    scheduler = CosineAnnealingLR(opt, T_max=num_examples / args.batch_size + 1, eta_min=1e-5)
    with tqdm(total=num_examples, disable=rank > 0) as t:
        for batches in train_loader:
            for images, images_dino, images_clip, targets in batches:
                # images_dino = transform_dino(images)
                # images_clip = transform_clip(images)
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

                    # print(f"summary shape: {teacher_dino[0].shape}")
                    # print(f"feature shape: {teacher_dino[1].shape}")
                    dino_features_upsampled = F.interpolate(teacher_dino[1].reshape(teacher_dino[1].shape[0], 16, 16, 1536).permute(0, 3, 1, 2), size=(27, 27), mode='bilinear')
                    dino_features_upsampled = dino_features_upsampled.permute(0, 2, 3, 1).reshape(teacher_dino[1].shape[0], 729, 1536)

                    loss = summary_c(student_dino[0], teacher_dino[0]) + summary_c(student_clip[0], teacher_clip[0]) + feature_c(student_dino[1], dino_features_upsampled) + feature_c(student_clip[1], teacher_clip[1])
                    # print(student_dino[0][0,0],teacher_dino[0][0,0])
                    # loss =  summary_c(student_dino[0], teacher_dino[0]) + summary_c(student_clip[0], teacher_clip[0])

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                
            if num_processed % 8192 == 0:
                val_loss = calc_valid_loss(model_radio, model_dino, model_clip, valid_loader, summary_c, feature_c, args.batch_size, device, 512)
                print(f"valid loss: {val_loss:.10f}")
                writer.add_scalar('Loss/valid', val_loss, valid_cnt)
                valid_cnt += 1


            num_processed += world_size * args.batch_size

            t.update(world_size * args.batch_size)
            t.set_postfix(loss=loss.item())
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
