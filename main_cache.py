import argparse
import datetime
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode, transforms

import utils.misc as misc
from models.vqvae import VQVAE
from utils.data import (CenterCropTransform, ImageFolderWithFilename,
                        normalize_01_into_pm1)
from utils.engine_mvar import cache_latents


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    parser.add_argument('--train', action='store_true',
                        help='whether to use horizontal flip')

    return parser

def compile_model(m, fast):
    if fast == 0:
        return m
    return (
        torch.compile(
            m,
            mode={
                1: "reduce-overhead",
                2: "max-autotune",
                3: "default",
            }[fast],
        )
        if hasattr(torch, "compile")
        else m
    )


def main(args):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.cached_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.cached_path, timeout=30)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # augmentation following DiT and ADM
    transform_train = transforms.Compose(
        [
            CenterCropTransform(args.img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    transform_val = transforms.Compose(
        [
            CenterCropTransform(args.img_size),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    dataset_train = ImageFolderWithFilename(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)
    dataset_val = ImageFolderWithFilename(os.path.join(args.data_path, 'val'), transform=transform_val)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )

    print("Sampler_val = %s" % str(sampler_val))

    if args.train:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,  # Don't drop in cache
        )
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,  # Don't drop in cache
        )

    # define the vae
    if args.img_size == 256:
        v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    elif args.img_size == 512:
        v_patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    else:
        raise ValueError(f'Unsupported image size: {args.img_size}')
    print(f'Patch numbers: {v_patch_nums}, {len(v_patch_nums)}, img_size: {args.img_size}')
    vae = (
        VQVAE(
            vocab_size=4096,
            z_channels=32,
            ch=160,
            test_mode=True,
            share_quant_resi=4,
            v_patch_nums=v_patch_nums,
        )
        .to(device)
        .eval()
    )
    vae.load_state_dict(torch.load(args.vae_path, map_location="cpu"), strict=True)


    # training
    if args.train:
        print(f"Start caching VAE train latents")
        start_time = time.time()
        cache_latents(
            vae,
            data_loader_train,
            device,
            args=args
        )
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Caching time {}'.format(total_time_str))
    else:
        # valing
        print(f"Start caching VAE val latents")
        start_time = time.time()
        cache_latents(vae, data_loader_val, device, args=args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Caching time {}".format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
