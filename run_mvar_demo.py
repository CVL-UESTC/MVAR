################## 1. Download checkpoints and build models
import argparse
import os
import random

import numpy as np
import PIL.Image as PImage
import torch
import torchvision

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
from models import build_vae_mvar


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed DataLoader Example")
    parser.add_argument("--cfg", type=float, default=1.4, help="")
    parser.add_argument("--top_k", type=int, default=900, help="")
    parser.add_argument("--top_p", type=float, default=0.96, help="")
    parser.add_argument("--more_smooth", action="store_true", help="")
    parser.add_argument("--depth", type=int, default=16, help="Model Depth")
    parser.add_argument("--seed", type=int, default=0, help="used in sampler")
    parser.add_argument("--mvar_ckpt", type=str, default="", help="")
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="./pretrained/FoundationVision/var/vae_ch160v4096z32.pth",
        help="",
    )
    parser.add_argument(
        "--infer_patch_nums",
        type=str,
        default="1_2_3_4_5_6_8_10_13_16",
    )
    parser.add_argument("--refine_step", type=int, default=8, help="")
    parser.add_argument("--kernel_size", type=int, default=7, help="")
    parser.add_argument("--label", type=int, default=22, help="image label")
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.depth in {12, 16, 20, 24, 30}

    args.patch_nums = tuple(
        map(int, args.infer_patch_nums.replace("-", "_").split("_"))
    )

    vae_ckpt = args.vae_ckpt
    mvar_ckpt = args.mvar_ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build vae, mvar

    if "vae" not in globals() or "mvar" not in globals():
        vae, mvar = build_vae_mvar(
            device=device,
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,  # hard-coded VQVAE hyperparameters
            patch_nums=args.patch_nums,
            num_classes=1000,
            depth=args.depth,
            shared_aln=False,
            kernel_size=args.kernel_size,
            refine_step=args.refine_step,
        )
    # load checkpoints
    vae.load_state_dict(
        torch.load(vae_ckpt, map_location="cpu", weights_only=True), strict=True
    )

    checkpoint = torch.load(mvar_ckpt, map_location="cpu", weights_only=False)

    if "mvar_wo_ddp" in checkpoint:
        mvar.load_state_dict(
            checkpoint["mvar_wo_ddp"],
            strict=True,
        )
    else:
        raise ValueError(f"mvar.ckpt is ERROR!")
 

    vae.eval(), mvar.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in mvar.parameters():
        p.requires_grad_(False)
    print(f"################# prepare finished. #######################")

    ############################# 
    # 2. Sample with classifier-free guidance
    vae_param = 0
    for p in vae.parameters():
        vae_param += p.numel()

    print(f"######## VAE Param {vae_param / 10**6} M ###########")

    mvar_param = 0
    for p in mvar.parameters():
        mvar_param += p.numel()
    print(f"######## MVAR Param {mvar_param / 10**6} M #########")

    ############################# 2. Sample with classifier-free guidance

    # set args
    seed = args.seed  # @param {type:"number"}
    torch.manual_seed(seed)

    class_labels = [args.label for i in range(24)]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    save_path = os.path.splitext(os.path.basename(args.mvar_ckpt))[0]
    test_save_path = os.path.join("outputs", "test", save_path)
    os.makedirs(test_save_path, exist_ok=True)

    # sample
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    
    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):  
            recon_f_hat = mvar.autoregressive_infer(
                B=B,
                label_B=label_B,
                g_seed=seed,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=args.more_smooth
            )
            recon_B3HW = vae.fhat_to_img(recon_f_hat.clone()).add_(1).mul_(0.5)

            chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)

            chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
            chw = PImage.fromarray(chw.astype(np.uint8))
            # chw.show()

            num_png = os.listdir(test_save_path).__len__()
            save_name = os.path.join(
                test_save_path,
                f"samples_{num_png}_seed{seed}_cfg{args.cfg}_topk{args.top_k}_topp{args.top_p}.png",
            )
            chw.save(save_name)
            print(f"samples saved to {save_name}")
            print(f"recon_B3HW.shape: {recon_B3HW.shape}")

if __name__ == "__main__":
    main()
