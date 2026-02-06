import os
from typing import Iterable

import numpy as np
import torch

import utils.misc as misc


def cache_latents(vae, data_loader: Iterable, device: torch.device, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Caching: "
    print_freq = 20

    for _, (samples, _, paths) in enumerate(
        metric_logger.cache_log_every(data_loader, print_freq, header)
    ):
        inp_B3HW = samples.to(device, non_blocking=True)

        with torch.no_grad():
            gt_idx_Bl = vae.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = vae.quantize.idxBl_to_mvar_input(gt_idx_Bl)

            gt_idx_Bl_flip = vae.img_to_idxBl(inp_B3HW.flip(dims=[3]))
            gt_BL_flip = torch.cat(gt_idx_Bl_flip, dim=1)
            x_BLCv_wo_first_l_flip = vae.quantize.idxBl_to_mvar_input(gt_idx_Bl_flip)

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + ".npz")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                moments_gt=gt_BL[i].cpu().numpy(),
                moments_gt_flip=gt_BL_flip[i].cpu().numpy(),
                moments_x=x_BLCv_wo_first_l[i].cpu().numpy(),
                moments_x_flip=x_BLCv_wo_first_l_flip[i].cpu().numpy(),
            )

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
