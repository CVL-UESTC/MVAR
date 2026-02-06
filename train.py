import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils import arg_util, dist, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume


def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(
        args, "mvar-ckpt*.pth"
    )
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'_{misc.time_str("%m%d_%H%M")}',
            ),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    # log args
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")
    print(f"initial args:\n{str(args)}")

    # build data
    num_classes, dataset_train, dataset_val = build_dataset(
        args.data_path,
        final_reso=args.data_load_reso,
        hflip=args.hflip,
        ### for latents cache ###
        use_cached=args.use_cached,
    )
    if args.local_debug:
        dataset_train = (
            Subset(dataset_train, list(range(args.batch_size * 4)))
        )
        dataset_val = (
            Subset(dataset_val, list(range(args.batch_size * 4)))
        )


    types = str((type(dataset_train).__name__, type(dataset_val).__name__))

    ld_val = DataLoader(
        dataset_val,
        num_workers=0,
        pin_memory=True,
        batch_size=round(args.batch_size * 1.5),
        sampler=EvalDistributedSampler(
            dataset_val, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank()
        ),
        shuffle=False,
        drop_last=False,
    )
    del dataset_val

    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_ep=start_ep,
            start_it=start_it,
        ),
    )
    del dataset_train

    [print(line) for line in auto_resume_info]
    print(f"[dataloader multi processing] ...", end="", flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(
        f"     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)",
        flush=True,
        clean=True,
    )
    print(
        f"[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}"
    )

    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP

    from models import MVAR, VQVAE, build_vae_mvar
    from trainer import MVARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params

    vae_local, mvar_wo_ddp = build_vae_mvar(
        device=dist.get_device(),
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=args.patch_nums,
        num_classes=num_classes,
        depth=args.depth,
        shared_aln=args.saln,
        attn_l2_norm=args.anorm,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        refine_step=args.refine_step,
        ### NA args ###
        kernel_size=args.kernel_size,
    )
    if dist.is_local_master():
        if not os.path.exists(args.vae_ckpt):
            raise FileNotFoundError(args.vae_ckpt)
        if args.finetune_from_var:
            if not os.path.exists(args.var_ckpt):
                raise FileNotFoundError(args.vae_ckpt)
            mvar_wo_ddp.load_state_dict(
                torch.load(args.var_ckpt, map_location="cpu", weights_only=False), strict=True
            )
    dist.barrier()
    vae_local.load_state_dict(
        torch.load(args.vae_ckpt, map_location="cpu", weights_only=False), strict=True
    )
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)

    mvar_wo_ddp: MVAR = args.compile_model(mvar_wo_ddp, args.tfast)
    mvar: DDP = (DDP if dist.initialized() else NullDDP)(
        mvar_wo_ddp,
        device_ids=[dist.get_local_rank()],
        # performs additional checks to detect parameters that do not receive gradients during backpropagation
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    warnings.filterwarnings(
        "ignore", message="find_unused_parameters=True was specified in DDP constructor"
    )

    print(f"############ [INIT] MVAR model = {mvar_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters())/1e6:.2f}"

    print(
        f"[INIT][#para] "
        + ", ".join(
            [
                f"{k}={count_p(m)}"
                for k, m in (
                    ("VAE", vae_local),
                    ("VAE.enc", vae_local.encoder),
                    ("VAE.dec", vae_local.decoder),
                    ("VAE.quant", vae_local.quantize),
                )
            ]
        )
    )

    print(
        f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("MVAR", mvar_wo_ddp),)])
        + "\n\n"
    )

    # build optimizer
    names, paras, para_groups = filter_params(
        mvar_wo_ddp,
        nowd_keys={
            "cls_token",
            "start_token",
            "task_token",
            "cfg_uncond",
            "pos_embed",
            "pos_1LC",
            "pos_start",
            "start_pos",
            "lvl_embed",
            "gamma",
            "beta",
            "ada_gss",
            "moe_bias",
            "scale_mul",
        },
    )
    opt_clz = {
        "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f"[INIT] optim={opt_clz}, opt_kw={opt_kw}\n")

    mvar_optim = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=opt_clz(params=para_groups, **opt_kw),
        names=names,
        paras=paras,
        grad_clip=args.tclip,
        n_gradient_accumulation=args.ac,
    )
    del names, paras, para_groups

    # build trainer
    trainer = MVARTrainer(
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        mvar_wo_ddp=mvar_wo_ddp,
        mvar=mvar,
        mvar_opt=mvar_optim,
        label_smooth=args.ls,
        refine_step=args.refine_step,
        vae_local=vae_local,
        total_iters=iters_train * args.ep,
        warmup_iters=args.wp * iters_train,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(
            trainer_state, strict=False, skip_vae=True
        )  # don't load vae again
    del vae_local, mvar_wo_ddp, mvar, mvar_optim

    dist.barrier()
    return (tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val)


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)

    (tb_lg, trainer, start_ep, start_it, iters_train, ld_train, ld_val) = (
        build_everything(args)
    )

    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999.0, 999.0, -1.0, -1.0
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = (
        999,
        999,
        -1,
        -1,
    )

    L_mean, L_tail, acc_mean, acc_tail, grad_norm = -1, -1, -1, -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, "sampler") and hasattr(ld_train.sampler, "set_epoch"):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(
                    f"[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]",
                    flush=True,
                    force=True,
                )


        tb_lg.set_step(ep * iters_train)

        AR_ep_loss = dict(
            vL_mean=-1,
            vL_tail=-1,
            vacc_mean=-1,
            vacc_tail=-1,
            L_mean=-1,
            L_tail=-1,
            acc_mean=-1,
            acc_tail=-1,
        )

        is_val_and_also_saving = (
            (ep > 0 and ep % 5 == 0)
            or (ep + 1) == args.ep
        )
        if is_val_and_also_saving:
            if not args.use_cached:
                ### use image to train ####
                (
                    val_loss_mean,
                    val_loss_tail,
                    val_acc_mean,
                    val_acc_tail,
                    kw,
                    tot,
                    cost,
                ) = trainer.eval_ep(
                    ld_val,
                )
            else:
                val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, kw, tot, cost = (
                    trainer.eval_ep_cached(
                        ld_val,
                    )
                )

            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(
                best_val_loss_mean, val_loss_mean
            ), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(
                best_val_acc_mean, val_acc_mean
            ), max(best_val_acc_tail, val_acc_tail)

            AR_ep_loss.update(
                vL_mean=val_loss_mean,
                vL_tail=val_loss_tail,
                vacc_mean=val_acc_mean,
                vacc_tail=val_acc_tail,
            )

            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = (
                val_loss_mean,
                val_loss_tail,
                val_acc_mean,
                val_acc_tail,
            )
            print(
                f" [*] [ep{ep}]  (val {tot})  Lm: {val_loss_mean:.4f}({best_val_loss_mean}), Lt: {val_loss_tail:.4f}({best_val_loss_tail}), Acc m&t: {val_acc_mean:.2f} {val_acc_tail:.2f}({best_val_acc_mean} {best_val_acc_tail}),  Val cost: {cost:.2f}s"
            )
            tb_lg.update(head=f"ep_loss", step=ep + 1, **kw)

            if dist.is_local_master():
                local_out_ckpt = os.path.join(
                    args.local_out_dir_path, f"mvar-ckpt-ep-{ep + 1}.pth"
                )
                # local_out_ckpt = os.path.join(args.local_out_dir_path, f"var-ckpt-last.pth")
                local_out_ckpt_best = os.path.join(
                    args.local_out_dir_path, "mvar-ckpt-best.pth"
                )
                print(f"[saving ckpt] ...", end="", flush=True)
                torch.save(
                    {
                        "epoch": ep + 1,
                        "iter": 0,
                        "trainer": trainer.state_dict(),
                        "args": args.state_dict(),
                    },
                    local_out_ckpt,
                )
                if best_updated:
                    print(f"[best_updated] ...", end="", flush=True)
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(
                    f"     [saving ckpt](*) finished!  @ {local_out_ckpt}",
                    flush=True,
                    clean=True,
                )
            dist.barrier()

        if not args.use_cached:
            stats, (sec, remain_time, finish_time) = (
                trainer.train_one_ep_ratio_k(
                    ep=ep,
                    is_first_ep=(ep == start_ep),
                    start_it=(start_it if ep == start_ep else 0),
                    args=args,
                    tb_lg=tb_lg,
                    ld_or_itrt=ld_train,
                    iters_train=iters_train,
                    ratio_k=args.ratio_k,
                )
            )
        else:
            stats, (sec, remain_time, finish_time) = trainer.train_one_ep_cached_ratio_k(
                ep=ep,
                is_first_ep=(ep == start_ep),
                start_it=(start_it if ep == start_ep else 0),
                args=args,
                tb_lg=tb_lg,
                ld_or_itrt=ld_train,
                iters_train=iters_train,
                ratio_k=args.ratio_k,
            )


        L_mean, L_tail, acc_mean, acc_tail, grad_norm = (
            stats["Lm"],
            stats["Lt"],
            stats["Accm"],
            stats["Acct"],
            stats["tnm"],
        )
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(
            best_acc_mean, acc_mean
        )
        if L_tail != -1:
            best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(
                best_acc_tail, acc_tail
            )
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = (
            L_mean,
            L_tail,
            acc_mean,
            acc_tail,
            grad_norm,
        )
        args.cur_ep = f"{ep+1}/{args.ep}"
        args.remain_time, args.finish_time = remain_time, finish_time

        if is_val_and_also_saving:
            AR_ep_loss.update(
                L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail
            )

            print(
                f"     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {acc_mean:.2f} {acc_tail:.2f} ({best_acc_mean:.2f} {best_acc_tail:.2f}),  Remain: {remain_time},  Finish: {finish_time}",
                flush=True,
            )
            tb_lg.update(head=f"ep_loss", step=ep + 1, **AR_ep_loss)
            tb_lg.update(
                head=f"z_burnout",
                step=ep + 1,
                rest_hours=round(sec / 60 / 60, 2),
            )
            args.dump_log()
            tb_lg.flush()
        dist.barrier()

    total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"
    print("\n\n")
    print(
        f"  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})"
    )
    print("\n\n")

    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)

    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
