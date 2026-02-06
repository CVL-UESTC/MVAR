import argparse
import os
import os.path as osp
import random
import time
from re import M

import numpy as np
import PIL.Image as PImage
import torch
import torchvision
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from models import build_vae_mvar
from utils.data import build_dataset_for_eval
from utils.misc import create_npz_from_sample_folder

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed DataLoader Example")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch Size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--cfg", type=float, default=1.4)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--more_smooth", action="store_true")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mvar_ckpt", type=str, default="")
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="./pretrained/FoundationVision/var/vae_ch160v4096z32.pth",
    )
    parser.add_argument(
        "--infer_patch_nums", type=str, default="1_2_3_4_5_6_8_10_13_16"
    )
    parser.add_argument(
        "--keep_images",
        action="store_true",
        help="Keep PNG images after .npz conversion",
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=1000, 
        help="Number of distinct classes in the evaluation dataset."
    )

    parser.add_argument(
        "--num_samples_per_class", 
        type=int, 
        default=50, 
        help="Number of samples to generate per class. Total samples = num_classes × num_samples_per_class."
    )
    parser.add_argument("--kernel_size", type=int, default=7, help="Model Depth")
    parser.add_argument("--refine_step", type=int, default=8, help="Scale of starting spatial attention")

    return parser.parse_args()


def build_dataloader(args):
    num_classes, dataset = build_dataset_for_eval(
        num_classes=args.num_classes,
        num_samples_per_class=args.num_samples_per_class,
    )

    accelerator.print(
        f"############ Eval Dataset built [ [val : {len(dataset)}], [num_classes : {num_classes}], [num_samples_per_class : {args.num_samples_per_class}] ############"
    )

    total_len = len(dataset)
    group_size = accelerator.num_processes * args.batch_size
    last_dataset_num = total_len % group_size
    main_dataset_size = total_len - last_dataset_num

    main_dataset = Subset(dataset, list(range(main_dataset_size)))
    last_dataset = (
        Subset(dataset, list(range(main_dataset_size, total_len)))
        if last_dataset_num > 0
        else None
    )

    dataloader = DataLoader(
        main_dataset,
        num_workers=args.workers,
        pin_memory=True,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    if last_dataset is not None:
        last_dataloader = DataLoader(
            last_dataset,
            num_workers=args.workers,
            pin_memory=True,
            batch_size=max(
                1, int(np.ceil(len(last_dataset) / accelerator.num_processes))
            ),
            shuffle=False,
            drop_last=False,
        )
    else:
        last_dataloader = None

    return dataloader, last_dataloader


def main():
    args = parse_args()
    assert args.depth in {12, 16, 20, 24, 30}

    exp_name = os.path.splitext(os.path.basename(args.mvar_ckpt))[0] if args.mvar_ckpt else "debug"

    args.patch_nums = tuple(
        map(int, args.infer_patch_nums.replace("-", "_").split("_"))
    )
    args.resos = tuple(pn * 16 for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)
    args.workers = min(max(0, args.workers), args.batch_size)

    # build vae & var
    vae, mvar = build_vae_mvar(
        device=device,
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        patch_nums=args.patch_nums,
        num_classes=1000,
        depth=args.depth,
        shared_aln=False,
        kernel_size=args.kernel_size,
        refine_step=args.refine_step,
    )

    # load checkpoints
    vae.load_state_dict(
        torch.load(args.vae_ckpt, map_location="cpu", weights_only=True), strict=True
    )
    checkpoint = torch.load(args.mvar_ckpt, map_location="cpu", weights_only=False)
    if "mvar_wo_ddp" in checkpoint:
        mvar.load_state_dict(
            checkpoint["mvar_wo_ddp"],
            strict=True,
        )
    else:
        raise ValueError(f"mvar.ckpt is ERROR!")


    total_samples = args.num_classes * args.num_samples_per_class
    sample_str = f"{total_samples // 1000}k"  # e.g., 12000 -> "12k"

    recon_path = (
        f"outputs/Fid/recon_{exp_name}"
        f"_seed{args.seed}_cfg{args.cfg}_topk{args.top_k}_topp{args.top_p}_{sample_str}"
    )

    os.makedirs(recon_path, exist_ok=True)

    dataloader, last_dataloader = build_dataloader(args)
    vae, mvar, dataloader, last_dataloader = accelerator.prepare(
        vae, mvar, dataloader, last_dataloader
    )

    # Freeze models
    vae.eval(), mvar.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in mvar.parameters():
        p.requires_grad_(False)
    accelerator.print(f"################# prepare finished #################")

    vae_param = sum(p.numel() for p in vae.parameters())
    mvar_param = sum(p.numel() for p in mvar.parameters())
    accelerator.print(f"######## VAE Param {vae_param / 1e6:.2f} M ########")
    accelerator.print(f"######## MVAR Param {mvar_param / 1e6:.2f} M ########")

    # Seed setup
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Start sampling
    total_sample_time = 0
    accelerator.print(
        f"################ start sampling [use cfg {args.cfg != 1.0}] ################"
    )

    raw_mvar = accelerator.unwrap_model(mvar)
    raw_vae = accelerator.unwrap_model(vae)

    for i, batch in enumerate(tqdm(dataloader)):
        label_B = batch.to(device)
        start_time = time.time()

        with accelerator.autocast():
            recon_f_hat = raw_mvar.autoregressive_infer(
                B=label_B.shape[0],
                label_B=label_B,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                g_seed=seed,
                more_smooth=args.more_smooth
            )
            recon_B3HW = raw_vae.fhat_to_img(recon_f_hat).add_(1).mul_(0.5)

        total_sample_time += time.time() - start_time

        # print("label_B:", label_B.shape, "recon_B3HW:", recon_B3HW.shape)
        for j in range(label_B.shape[0]):
            img = (
                recon_B3HW[j]
                .permute(1, 2, 0)
                .mul(255)
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
            )
            recon_img = PImage.fromarray(img)
            recon_img.save(
                os.path.join(
                    recon_path,
                    f"num_{(accelerator.process_index):01d}_b_{i:03d}_samples_{j:03d}_label_{int(label_B[j]):03d}.png",
                )
            )

    if last_dataloader is not None:

        for i, batch in enumerate(tqdm(last_dataloader)):
            label_B = batch.to(device)
            start_time = time.time()

            with accelerator.autocast():
                recon_f_hat = raw_mvar.autoregressive_infer(
                    B=label_B.shape[0],
                    label_B=label_B,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    g_seed=seed,
                    more_smooth=args.more_smooth
                )
                recon_B3HW = raw_vae.fhat_to_img(recon_f_hat).add_(1).mul_(0.5)

            total_sample_time += time.time() - start_time

            # print("label_B:", label_B.shape, "recon_B3HW:", recon_B3HW.shape)
            for j in range(label_B.shape[0]):
                img = (
                    recon_B3HW[j]
                    .permute(1, 2, 0)
                    .mul(255)
                    .clamp(0, 255)
                    .byte()
                    .cpu()
                    .numpy()
                )
                recon_img = PImage.fromarray(img)
                recon_img.save(
                    os.path.join(
                        recon_path,
                        f"num_{(accelerator.process_index):01d}_last_b_{i:03d}_samples_{j:03d}_label_{int(label_B[j]):03d}.png",
                    )
                )

    accelerator.print(
        f"######## avg sample time = {total_sample_time / len(dataloader) / args.batch_size:.4f} sec ########"
    )

    accelerator.wait_for_everyone()

    # Save results
    if accelerator.is_main_process:
        create_npz_from_sample_folder(recon_path, total_samples=total_samples)
        accelerator.print("Done! Results saved at:", recon_path)

        # Optional: Clean up
        if not args.keep_images:
            import shutil

            try:
                shutil.rmtree(recon_path)
                print(f"Clean: {recon_path}")
            except Exception as e:
                print(f"Clean {recon_path} Fail: {e}")


if __name__ == "__main__":
    main()
