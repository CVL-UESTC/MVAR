import math
import time
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch.nn import functional as F

from models.basic_mvar import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from utils import dist

def generate_backward_attn_mask_only_next(patch_nums) -> torch.Tensor:
    L = sum(pn * pn for pn in patch_nums)  # Total length
    attn_bias_for_masking = torch.full((L, L), float("-inf"))
    attn_bias_for_masking[0][0] = 0
    cum_sums = torch.cumsum(torch.tensor([pn**2 for pn in patch_nums]), dim=0)
    pn_list = list(zip(cum_sums[:-1].tolist(), cum_sums[1:].tolist()))
    last_cul = 1
    for pn in pn_list:
        for i in range(pn[0], pn[1]):
            attn_bias_for_masking[i, last_cul : pn[1]] = 0
        last_cul = pn[1]

    return attn_bias_for_masking


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class MVAR(nn.Module):

    def __init__(
        self,
        # VQVAE args
        vae_local: VQVAE,
        # MVAR args
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        ######### using NATTN #########
        kernel_size=7,
        refine_step=8,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.kernel_size = kernel_size
        self.cond_drop_rate = cond_drop_rate

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2

        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full(
            (1, num_classes),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device=dist.get_device(),
        )
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for _, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C))
            if shared_aln
            else nn.Identity()
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(depth)
            ]
        )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f"\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n"
            f"    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n"
            f"    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})",
            end="\n\n",
            flush=True,
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)

        attn_bias_for_masking = generate_backward_attn_mask_only_next(
            self.patch_nums
        ).reshape(1, 1, self.L, self.L)

        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )

        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        self.refine_step = refine_step

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer(
        self,
        *,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(
                self.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.num_classes if label_B < 0 else label_B,
                device=self.lvl_1L.device,
            )
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC


        if not cfg == 1.0:
            sos = cond_BD = self.class_emb(
                torch.cat(
                    (label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0
                )
            )
            next_token_map = (
                sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            sos = cond_BD = self.class_emb(label_B)

            next_token_map = (
                sos.unsqueeze(1).expand(B, self.first_l, -1)
                + self.pos_start.expand(B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1

            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map  # [B, l, C]
            AdaLNSelfAttn.forward
            using_nattn = si >= self.refine_step
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    using_nattn=using_nattn,
                    kernel_size=self.kernel_size,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            
            if not cfg == 1.0:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    self.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                )
                if not cfg == 1.0:
                    next_token_map = next_token_map.repeat(
                        2, 1, 1
                    )  # double the batch sizes due to CFG

        return f_hat

    def update_patch_related(self, x_BlC, attn_bias):
        B, L, C = x_BlC.shape
        assert L == attn_bias.shape[-1]

        L_ = (L + 7) // 8 * 8
        attn_bias_for_masking_ = torch.full(
            (1, 1, L_, L_), -torch.inf, dtype=attn_bias.dtype, device=attn_bias.device
        )
        attn_bias_for_masking_[:, :, :L, :L] = attn_bias

        x_BlC_ = torch.cat(
            [
                x_BlC,
                torch.zeros((B, L_ - L, C), device=x_BlC.device, dtype=x_BlC.dtype),
            ],
            dim=1,
        )

        return x_BlC_, attn_bias_for_masking_

    def forward(
        self,
        *,
        label_B: torch.LongTensor,
        x_BlCv_wo_first_l: torch.Tensor = None,
        L_bg: int = 0,
        L_ed: int = 0,
        using_nattn: bool = False,
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BlCv_wo_first_l: teacher forcing input (B, self.l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        B = label_B.shape[0]
        with torch.amp.autocast("cuda", enabled=False):
            label_B = torch.where(
                torch.rand(B, device=label_B.device) < self.cond_drop_rate,
                self.num_classes,
                label_B,
            )
            cond_BD = self.class_emb(label_B)

            if L_bg == 0:
                sos = self.class_emb(label_B).unsqueeze(1).expand(
                    B, self.first_l, -1
                ) + self.pos_start.expand(B, self.first_l, -1)

                if x_BlCv_wo_first_l is not None:
                    x_BlC = torch.cat(
                        (sos, self.word_embed(x_BlCv_wo_first_l.float())), dim=1
                    )
                else:
                    x_BlC = sos
            else:
                x_BlC = self.word_embed(x_BlCv_wo_first_l.float())

            x_BlC += (
                self.lvl_embed(self.lvl_1L[:, L_bg:L_ed].expand(B, -1))
                + self.pos_1LC[:, L_bg:L_ed]
            )  # lvl: BLC;  pos: 1LC
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BlC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BlC = x_BlC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)

        concat_flag = False if (L_ed - L_bg) % 8 == 0 else True

        if L_bg == 0 and (L_ed - L_bg) != self.first_l:
            assert using_nattn == False, f"using_nattn is not supported when L_bg == 0"
            attn_bias = self.attn_bias_for_masking[
                :,
                :,
                L_bg:L_ed,
                L_bg:L_ed,
            ]
            attn_bias = attn_bias.to(dtype=main_type)

            if concat_flag:
                x_BlC, attn_bias = self.update_patch_related(
                    x_BlC=x_BlC, attn_bias=attn_bias
                )
        else:
            # assert using_nattn == True, f"using_nattn is must be True when L_bg != 0"
            attn_bias = None
                

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BlC = b(
                x=x_BlC,
                cond_BD=cond_BD_or_gss,
                attn_bias=attn_bias,
                using_nattn=using_nattn,
                kernel_size=self.kernel_size,
            )

        if concat_flag and L_bg == 0:
            x_BlC = x_BlC[:, L_bg:L_ed]


        x_BlV = self.get_logits(x_BlC.float(), cond_BD)  # BLC -> BLV

        return x_BlV  # logits BLV, V is vocab_size

    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
        conv_std_or_gain=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, "ada_lin"):
                sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(sab.ada_lin[-1], "bias")
                    and sab.ada_lin[-1].bias is not None
                ):
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, "ada_gss"):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"
