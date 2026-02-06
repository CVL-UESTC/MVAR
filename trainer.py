import os
import random
import time
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
import torch
import torch.nn as nn
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models.mvar import MVAR
from models.vqvae import VQVAE, VectorQuantizer2
from utils import arg_util, dist, misc
from utils.amp_sc import AmpOptimizer
from utils.lr_control import lr_wd_annealing
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class MVARTrainer(object):

    def __init__(
        self,
        *,
        device,
        patch_nums: Tuple[int, ...],
        resos: Tuple[int, ...],
        mvar_wo_ddp: MVAR,
        mvar: DDP,
        mvar_opt: AmpOptimizer,
        label_smooth: float,
        refine_step: int,
        vae_local: VQVAE,
        total_iters: int = 5000,
        warmup_iters: int = 500,
    ):
        super(MVARTrainer, self).__init__()

        self.mvar, self.vae_local, self.quantize_local = (
            mvar,
            vae_local,
            vae_local.quantize,
        )
        self.quantize_local: VectorQuantizer2

        self.mvar_wo_ddp: MVAR = mvar_wo_ddp  # after torch.compile
        self.mvar_opt = mvar_opt

        del self.mvar_wo_ddp.rng
        self.mvar_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.refine_step = refine_step
        
    @torch.no_grad()
    def eval_ep_cached(
        self,
        ld_val: DataLoader,
    ):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.mvar_wo_ddp.training
        self.mvar_wo_ddp.eval()

        kw = dict()
        for si, (bg, ed) in enumerate(self.begin_ends):
            kw[f"vacc_{self.resos[si]}"] = 0
            kw[f"vL_{self.resos[si]}"] = 0

        for gt_BL, x_BLCv_wo_first_l, label_B in ld_val:
            gt_BL, x_BLCv_wo_first_l, label_B = (
                gt_BL.to(dist.get_device(), non_blocking=True),
                x_BLCv_wo_first_l.to(dist.get_device(), non_blocking=True),
                label_B.to(dist.get_device(), non_blocking=True),
            )

            for i, prog_ed in enumerate(range(self.refine_step - 1, self.num_stages_minus_1 + 1)):
                if i == 0:
                    L_bg, L_ed = self.begin_ends[0][0], self.begin_ends[prog_ed][1]
                else:
                    L_bg, L_ed = (
                        self.begin_ends[prog_ed][0],
                        self.begin_ends[prog_ed][1],
                    )

                gt_Bl = gt_BL[:, L_bg:L_ed, ...]
                x_BlCv_wo_first_l: Ten = x_BLCv_wo_first_l[
                    :, int(max(0, L_bg - self.first_l)) : L_ed - self.first_l, ...
                ]

                with self.mvar_opt.amp_ctx:
                    self.mvar_wo_ddp.forward
                    logits_BlV = self.mvar_wo_ddp(
                        label_B=label_B,
                        x_BlCv_wo_first_l=x_BlCv_wo_first_l,
                        L_bg=L_bg,
                        L_ed=L_ed,
                        using_nattn=(i > 0),
                    )
                B, V = logits_BlV.shape[0], logits_BlV.shape[-1]
                L_mean += (
                    self.val_loss(logits_BlV.data.view(-1, V), gt_Bl.reshape(-1)) * B
                ) 

                acc_mean += (logits_BlV.data.argmax(dim=-1) == gt_Bl).sum() * (100 / gt_Bl.shape[1])

                if prog_ed == self.num_stages_minus_1:
                    L_tail += (
                        self.val_loss(
                            logits_BlV.data.reshape(-1, V),
                            gt_Bl.reshape(-1),
                        ) * B
                    )
                    acc_tail += (
                        logits_BlV.data.argmax(dim=-1)
                        == gt_Bl
                    ).sum() * (100 / self.last_l)

                if i == 0:
                    for si in range(0, prog_ed + 1):
                        bg, ed = self.begin_ends[si]

                        pred, tar = logits_BlV.data[:, bg:ed].reshape(-1, V), gt_Bl[
                            :, bg:ed
                        ].reshape(-1)
                        acc = (pred.argmax(dim=-1) == tar).sum().item() * (100 / (ed - bg))
                        ce = self.val_loss(pred, tar).item() * B

                        kw[f"vacc_{self.resos[si]}"] += acc
                        kw[f"vL_{self.resos[si]}"] += ce
                else:
                    pred, tar = logits_BlV.data.reshape(-1, V), gt_Bl.reshape(-1) 
                    acc = (pred.argmax(dim=-1) == tar).sum().item() * (
                        100 / gt_Bl.shape[1]
                    )
                    ce = self.val_loss(pred, tar).item() * B

                    kw[f"vacc_{self.resos[prog_ed]}"] += acc
                    kw[f"vL_{self.resos[prog_ed]}"] += ce
            tot += B

        for si, (bg, ed) in enumerate(self.begin_ends):
            kw[f"vacc_{self.resos[si]}"] /=  tot
            kw[f"vL_{self.resos[si]}"] /= tot

        self.mvar_wo_ddp.train(training)

        stats = L_mean.new_tensor(
            [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot]
        )
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()

        return L_mean, L_tail, acc_mean, acc_tail, kw, tot, time.time() - stt

    @torch.no_grad()
    def eval_ep(
        self,
        ld_val: DataLoader,
    ):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.mvar_wo_ddp.training
        self.mvar_wo_ddp.eval()

        kw = dict()
        for si, (bg, ed) in enumerate(self.begin_ends):
            kw[f"vacc_{self.resos[si]}"] = 0
            kw[f"vL_{self.resos[si]}"] = 0

        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_mvar_input(gt_idx_Bl)

            for i, prog_ed in enumerate(
                range(self.refine_step - 1, self.num_stages_minus_1 + 1)
            ):
                if i == 0:
                    L_bg, L_ed = self.begin_ends[0][0], self.begin_ends[prog_ed][1]
                else:
                    L_bg, L_ed = (
                        self.begin_ends[prog_ed][0],
                        self.begin_ends[prog_ed][1],
                    )

                gt_Bl = gt_BL[:, L_bg:L_ed, ...]
                x_BlCv_wo_first_l: Ten = x_BLCv_wo_first_l[
                    :, int(max(0, L_bg - self.first_l)) : L_ed - self.first_l, ...
                ]

                with self.mvar_opt.amp_ctx:
                    self.mvar_wo_ddp.forward
                    logits_BlV = self.mvar_wo_ddp(
                        label_B=label_B, 
                        x_BlCv_wo_first_l=x_BlCv_wo_first_l, 
                        L_bg=L_bg, 
                        L_ed=L_ed, 
                        using_nattn = (i > 0))

                B, V = logits_BlV.shape[0], logits_BlV.shape[-1]
                L_mean += (
                    self.val_loss(logits_BlV.data.view(-1, V), gt_Bl.reshape(-1)) * B
                ) 

                acc_mean += (logits_BlV.data.argmax(dim=-1) == gt_Bl).sum() * (100 / gt_Bl.shape[1])

                if prog_ed == self.num_stages_minus_1:
                    L_tail += (
                        self.val_loss(
                            logits_BlV.data.reshape(-1, V),
                            gt_Bl.reshape(-1),
                        )
                        * B
                    )
                    acc_tail += (logits_BlV.data.argmax(dim=-1) == gt_Bl).sum() * (
                        100 / self.last_l
                    )

                if i == 0:
                    for si in range(0, prog_ed + 1):
                        bg, ed = self.begin_ends[si]

                        pred, tar = logits_BlV.data[:, bg:ed].reshape(-1, V), gt_Bl[
                            :, bg:ed
                        ].reshape(-1)
                        acc = (pred.argmax(dim=-1) == tar).sum().item() * (
                            100 / (ed - bg)
                        )
                        ce = self.val_loss(pred, tar).item() * B

                        kw[f"vacc_{self.resos[si]}"] += acc
                        kw[f"vL_{self.resos[si]}"] += ce
                else:
                    pred, tar = logits_BlV.data.reshape(-1, V), gt_Bl.reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).sum().item() * (
                        100 / gt_Bl.shape[1]
                    )
                    ce = self.val_loss(pred, tar).item() * B

                    kw[f"vacc_{self.resos[prog_ed]}"] += acc
                    kw[f"vL_{self.resos[prog_ed]}"] += ce
            tot += B

        for si, (bg, ed) in enumerate(self.begin_ends):
            kw[f"vacc_{self.resos[si]}"] /= tot
            kw[f"vL_{self.resos[si]}"] /= tot

        self.mvar_wo_ddp.train(training)

        stats = L_mean.new_tensor(
            [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot]
        )
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()

        return L_mean, L_tail, acc_mean, acc_tail, kw, tot, time.time() - stt

    def train_step(
        self,
        *,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: MetricLogger,
        tb_lg: TensorboardLogger,
        using_nattn: bool,
        L_bg: int,
        L_ed: int,
        prog_ed: int,
        gt_Bl: FTen,
        x_BlCv_wo_first_l: FTen,
        label_B: Union[ITen, FTen],
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:

        assert prog_ed > 0, f"prog_ed must be greater than 0, but got prog_ed = {prog_ed}"
        self.mvar.require_backward_grad_sync = stepping

        with self.mvar_opt.amp_ctx:
            self.mvar_wo_ddp.forward
            logits_BlV = self.mvar(
                label_B=label_B,
                x_BlCv_wo_first_l=x_BlCv_wo_first_l,
                L_bg=L_bg,
                L_ed=L_ed,
                using_nattn=using_nattn,
            )
            B, V = logits_BlV.shape[0], logits_BlV.shape[-1]
            loss = self.train_loss(logits_BlV.view(-1, V), gt_Bl.reshape(-1)).view(
                B, -1
            )
            assert logits_BlV.shape[1] == gt_Bl.shape[1] == (L_ed - L_bg)
            lw = self.loss_weight[:, L_bg:L_ed].clone()
            loss = loss.mul(lw).sum(dim=-1).mean()

        # backward
        grad_norm, scale_log2 = self.mvar_opt.backward_clip_step(
            stepping,
            loss,
        )
        # log
        pred_Bl = logits_BlV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BlV.data.view(-1, V), gt_Bl.reshape(-1)).item()
            acc_mean = (pred_Bl == gt_Bl).float().mean().item() * 100
            
            Ltail = self.val_loss(
                logits_BlV.data[:, -self.patch_nums[prog_ed] ** 2 :].reshape(-1, V),
                gt_Bl[:, -self.patch_nums[prog_ed] ** 2 :].reshape(-1),
            ).item()
            acc_tail = (
                pred_Bl[:, -self.patch_nums[prog_ed] ** 2 :]
                == gt_Bl[:, -self.patch_nums[prog_ed] ** 2 :]
            ).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm
            )

        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            Lmean = self.val_loss(logits_BlV.data.view(-1, V), gt_Bl.reshape(-1)).item()
            acc_mean = (pred_Bl == gt_Bl).float().mean().item() * 100

            prob_per_class_is_chosen = pred_Bl.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (
                prob_per_class_is_chosen > 0.001 / V
            ).float().mean().item() * 100
            if dist.is_master():

                kw = dict(z_voc_usage=cluster_usage)
                kw[f"acc_mean"] = acc_mean
                kw[f"L_mean"] = Lmean
                tb_lg.update(head="iter_loss", **kw, step=g_it)

        return grad_norm, scale_log2

    def train_one_ep_ratio_k(
        self,
        *,
        ep: int,
        is_first_ep: bool,
        start_it: int,
        args: arg_util.Args,
        tb_lg: misc.TensorboardLogger,
        ld_or_itrt,
        iters_train: int,
        ratio_k: int = 8,
    ):

        step_cnt = 0
        me = misc.MetricLogger(delimiter="  ")
        me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
        me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
        [
            me.add_meter(x, misc.SmoothedValue(fmt="{median:.3f} ({global_avg:.3f})"))
            for x in ["Lm", "Lt"]
        ]
        [
            me.add_meter(x, misc.SmoothedValue(fmt="{median:.2f} ({global_avg:.2f})"))
            for x in ["Accm", "Acct"]
        ]

        header = f"[Ep]: [{ep:4d}/{args.ep}]"

        if is_first_ep:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

        max_it = args.ep * iters_train

        for it, (inp_B3HW, label_B) in me.log_every(
            start_it,
            iters_train,
            ld_or_itrt,
            30 if iters_train > 8000 else 20,
            # iters_train,
            header,
        ):

            inp_B3HW = inp_B3HW.to(args.device, non_blocking=True)  # tensor[B, 3, H, W]
            label_B = label_B.to(args.device, non_blocking=True)
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)

            assert (
                len(gt_idx_Bl) == self.num_stages_minus_1 + 1
            ), f"Expected len(gt_idx_Bl) = {self.num_stages_minus_1 + 1}, but got len(gt_idx_Bl) = {len(gt_idx_Bl)}"

            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_mvar_input(gt_idx_Bl)

            g_it = ep * iters_train + it
            if it < start_it:
                continue
            if is_first_ep and it == start_it:
                warnings.resetwarnings()

            args.cur_it = f"{it+1}/{iters_train}"

            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
                args.sche,
                self.mvar_opt.optimizer,
                args.tlr,
                args.twd,
                args.twde,
                g_it,
                wp_it,
                max_it,
                wp0=args.wp0,
                wpe=args.wpe,
            )
            args.cur_lr, args.cur_wd = max_tlr, max_twd

            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)

            cur_prog = g_it % (ratio_k + self.num_stages_minus_1 + 1 - self.refine_step)

            if cur_prog < ratio_k:
                prog_ed = self.refine_step - 1
                L_bg, L_ed = self.begin_ends[0][0], self.begin_ends[prog_ed][1]
            else:
                prog_ed = self.refine_step + cur_prog % ratio_k if ratio_k != 1 else self.refine_step + (cur_prog - 1)
                L_bg, L_ed = self.begin_ends[prog_ed][0], self.begin_ends[prog_ed][1]

            print(f"[VARTrainer.train_one_ep] ratio_k:{ratio_k} cur_prog: {cur_prog}, prog_ed: {prog_ed}, L_bg: {L_bg}, L_ed: {L_ed}, len {L_ed - L_bg}")


            gt_Bl = gt_BL[:, L_bg:L_ed, ...]  # l = sum(i^2)

            x_BlCv_wo_first_l: Ten = x_BLCv_wo_first_l[
                        :, int(max(0, L_bg - self.first_l)) : L_ed - self.first_l, ...
            ]

            grad_norm, scale_log2 = self.train_step(
                it=it,
                g_it=g_it,
                stepping=stepping,
                metric_lg=me,
                tb_lg=tb_lg,
                using_nattn=(prog_ed >= self.refine_step),
                L_bg=L_bg,
                L_ed=L_ed,
                prog_ed=prog_ed,
                gt_Bl=gt_Bl,
                x_BlCv_wo_first_l=x_BlCv_wo_first_l,
                label_B=label_B,
            )

            me.update(tlr=max_tlr)
            tb_lg.set_step(step=g_it)
            tb_lg.update(head=f"opt_lr/lr_min", sche_tlr=min_tlr)
            tb_lg.update(head=f"opt_lr/lr_max", sche_tlr=max_tlr)
            tb_lg.update(head=f"opt_wd/wd_max", sche_twd=max_twd)
            tb_lg.update(head=f"opt_wd/wd_min", sche_twd=min_twd)
            tb_lg.update(head=f"opt_grad/fp16", scale_log2=scale_log2)

            if args.tclip > 0:
                tb_lg.update(head=f"opt_grad/grad", grad_norm=grad_norm)
                tb_lg.update(head=f"opt_grad/grad", grad_clip=args.tclip)

        me.synchronize_between_processes()
        return {
            k: meter.global_avg for k, meter in me.meters.items()
        }, me.iter_time.time_preds(
            max_it - (g_it + 1) + (args.ep - ep) * 15
        )  # +15: other cost

    def train_one_ep_cached_ratio_k(
        self,
        *,
        ep: int,
        is_first_ep: bool,
        start_it: int,
        args: arg_util.Args,
        tb_lg: misc.TensorboardLogger,
        ld_or_itrt,
        iters_train: int,
        ratio_k: int = 8,
    ):

        step_cnt = 0
        me = misc.MetricLogger(delimiter="  ")
        me.add_meter("tlr", misc.SmoothedValue(window_size=1, fmt="{value:.2g}"))
        me.add_meter("tnm", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
        [
            me.add_meter(x, misc.SmoothedValue(fmt="{median:.3f} ({global_avg:.3f})"))
            for x in ["Lm", "Lt"]
        ]
        [
            me.add_meter(x, misc.SmoothedValue(fmt="{median:.2f} ({global_avg:.2f})"))
            for x in ["Accm", "Acct"]
        ]

        header = f"[Ep]: [{ep:4d}/{args.ep}]"

        if is_first_ep:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

        max_it = args.ep * iters_train

        for it, (gt_BL, x_BLCv_wo_first_l, label_B) in me.log_every(
            start_it,
            iters_train,
            ld_or_itrt,
            30 if iters_train > 8000 else 20,
            # iters_train,
            header,
        ):

            gt_BL = gt_BL.to(args.device, non_blocking=True)
            x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(args.device, non_blocking=True)
            label_B = label_B.to(args.device, non_blocking=True)

            g_it = ep * iters_train + it
            if it < start_it:
                continue
            if is_first_ep and it == start_it:
                warnings.resetwarnings()

            args.cur_it = f"{it+1}/{iters_train}"

            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
                args.sche,
                self.mvar_opt.optimizer,
                args.tlr,
                args.twd,
                args.twde,
                g_it,
                wp_it,
                max_it,
                wp0=args.wp0,
                wpe=args.wpe,
            )
            args.cur_lr, args.cur_wd = max_tlr, max_twd

            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)

            cur_prog = g_it % (ratio_k + self.num_stages_minus_1 + 1 - self.refine_step)

            if cur_prog < ratio_k:
                prog_ed = self.refine_step - 1
                L_bg, L_ed = self.begin_ends[0][0], self.begin_ends[prog_ed][1]
            else:
                prog_ed = self.refine_step + cur_prog % ratio_k if ratio_k != 1 else self.refine_step + (cur_prog - 1)
                L_bg, L_ed = self.begin_ends[prog_ed][0], self.begin_ends[prog_ed][1]

            print(f"[VARTrainer.train_one_ep] ratio_k:{ratio_k} cur_prog: {cur_prog}, prog_ed: {prog_ed}, L_bg: {L_bg}, L_ed: {L_ed}, len {L_ed - L_bg}")


            gt_Bl = gt_BL[:, L_bg:L_ed, ...]  # l = sum(i^2)

            x_BlCv_wo_first_l: Ten = x_BLCv_wo_first_l[
                        :, int(max(0, L_bg - self.first_l)) : L_ed - self.first_l, ...
            ]

            grad_norm, scale_log2 = self.train_step(
                it=it,
                g_it=g_it,
                stepping=stepping,
                metric_lg=me,
                tb_lg=tb_lg,
                using_nattn=(prog_ed >= self.refine_step),
                L_bg=L_bg,
                L_ed=L_ed,
                prog_ed=prog_ed,
                gt_Bl=gt_Bl,
                x_BlCv_wo_first_l=x_BlCv_wo_first_l,
                label_B=label_B,
            )

            me.update(tlr=max_tlr)
            tb_lg.set_step(step=g_it)
            tb_lg.update(head=f"opt_lr/lr_min", sche_tlr=min_tlr)
            tb_lg.update(head=f"opt_lr/lr_max", sche_tlr=max_tlr)
            tb_lg.update(head=f"opt_wd/wd_max", sche_twd=max_twd)
            tb_lg.update(head=f"opt_wd/wd_min", sche_twd=min_twd)
            tb_lg.update(head=f"opt_grad/fp16", scale_log2=scale_log2)

            if args.tclip > 0:
                tb_lg.update(head=f"opt_grad/grad", grad_norm=grad_norm)
                tb_lg.update(head=f"opt_grad/grad", grad_clip=args.tclip)

        me.synchronize_between_processes()
        return {
            k: meter.global_avg for k, meter in me.meters.items()
        }, me.iter_time.time_preds(
            max_it - (g_it + 1) + (args.ep - ep) * 15
        )  # +15: other cost
    
    def get_config(self):
        return {
            "patch_nums": self.patch_nums,
            "resos": self.resos,
            "label_smooth": self.label_smooth,
        }

    def state_dict(self):
        state = {"config": self.get_config()}
        for k in ("mvar_wo_ddp", "mvar_opt"):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, "_orig_mod"):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=True):
        for k in ("mvar_wo_ddp", "mvar_opt"):

            m = getattr(self, k)
            if m is not None:
                if hasattr(m, "_orig_mod"):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f"[MVARTrainer.load_state_dict] {k} missing:  {missing}")
                    print(f"[MVARTrainer.load_state_dict] {k} unexpected:  {unexpected}")

        config: dict = state.pop("config", None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f"[MVAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})"
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)