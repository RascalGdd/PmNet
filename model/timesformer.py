# 模型的base版本，重构了TimeSFormer模型，基本上与TimeSFormer模型结果一致；
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("C:\DD\surgformer_pmlr50\LungSeg-diandian")
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math
import torchvision
from model.mambapy import Mamba_CSM, MambaConfig
def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 7,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_Spatial(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, B):
        BT, K, C = x.shape
        T = BT // B
        qkv = self.qkv(x)
        # For Intra-Spatial: (BT, heads, K, C)
        # Atten: K*K, Values: K*C
        qkv = rearrange(
            qkv,
            "(b t) k (qkv num_heads c) -> qkv (b t) num_heads k c",
            t=T,
            qkv=3,
            num_heads=self.num_heads,
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(
            x,
            "(b t) num_heads k c -> (b t) k (num_heads c)",
            b=B,
        )
        x = self.proj(x)
        return self.proj_drop(x)


class Attention_Temporal(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        with_qkv=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, B):
        BK, T, C = x.shape
        K = BK // B
        qkv = self.qkv(x)

        # For Intra-Spatial: (BK, heads, T, C)
        # Atten: T*T, Values: T*C
        qkv = rearrange(
            qkv,
            "(b k) t (qkv num_heads c) -> qkv (b k) num_heads t c",
            k=K,
            qkv=3,
            num_heads=self.num_heads,
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = rearrange(
            x,
            "(b k) num_heads t c -> (b k) t (num_heads c)",
            b=B,
        )

        x = self.proj(x)
        return self.proj_drop(x)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Separate Linear layers for Query, Key, and Value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, key_value, B):
        """
        Args:
            query: Tensor of shape (B * K, T_query, C)
            key_value: Tensor of shape (B * K, T_key, C)
            B: Batch size
        """
        BK_query, T_query, C = query.shape
        BK_key, T_key, _ = key_value.shape
        K_query = BK_query // B
        K_key = BK_key // B

        # Generate Query, Key, Value
        q = self.q_proj(query)  # (B * K_query, T_query, C)
        k = self.k_proj(key_value)  # (B * K_key, T_key, C)
        v = self.v_proj(key_value)  # (B * K_key, T_key, C)

        # Reshape for multi-head attention
        q = rearrange(q, "(b k) t (num_heads c) -> (b k) num_heads t c", k=K_query, num_heads=self.num_heads)
        k = rearrange(k, "(b k) t (num_heads c) -> (b k) num_heads t c", k=K_key, num_heads=self.num_heads)
        v = rearrange(v, "(b k) t (num_heads c) -> (b k) num_heads t c", k=K_key, num_heads=self.num_heads)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B * K_query, num_heads, T_query, T_key)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = attn @ v  # (B * K_query, num_heads, T_query, C_head)
        x = rearrange(x, "(b k) num_heads t c -> (b k) t (num_heads c)", b=B)

        # Project back to original dimension
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Spatial(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        ## Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention_Temporal(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T, K):
        # 如果alpha以及beta初始化为0，则xs、xt初始化为0, 在训练过程中降低了学习难度；
        # 仿照其余模型可以使用alpha.sigmoid()以及beta.sigmoid()；
        B, M, C = x.shape
        assert T * K + 1 == M

        # Temporal_Self_Attention
        xt = x[:, 1:, :]
        xt = rearrange(xt, "b (k t) c -> (b k) t c", t=T)
        res_temporal = self.drop_path(
            self.temporal_attn.forward(self.temporal_norm1(xt), B)
        )

        res_temporal = rearrange(
                res_temporal, "(b k) t c -> b (k t) c", b=B
            )  # 通过FC时需要将时空tokens合并，再通过残差连接连接输入特征
        xt = self.temporal_fc(res_temporal) + x[:, 1:, :]

        # Spatial_Self_Attention
        init_cls_token = x[:, 0, :].unsqueeze(1)  # B, 1, C
        cls_token = init_cls_token.repeat(1, T, 1)  # B, T, C
        cls_token = rearrange(cls_token, "b t c -> (b t) c", b=B, t=T).unsqueeze(1)
        xs = xt
        xs = rearrange(xs, "b (k t) c -> (b t) k c", t=T)

        xs = torch.cat((cls_token, xs), 1)  # BT, K+1, C
        res_spatial = self.drop_path(self.attn.forward(self.norm1(xs), B))

        ### Taking care of CLS token
        cls_token = res_spatial[:, 0, :]  # BT, C 表示了在每帧单独学习的class token
        cls_token = rearrange(cls_token, "(b t) c -> b t c", b=B, t=T)
        cls_token = torch.mean(cls_token, 1, True)  # 通过在全局帧上平均来建立时序关联（适用于视频分类任务）
        res_spatial = res_spatial[:, 1:, ]  # BT, xK, C
        res_spatial = rearrange(
            res_spatial, "(b t) k c -> b (k t) c", b=B)
        res = res_spatial
        x = xt
        ## Mlp
        x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过MLP学习时序对应的cls_token?

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=8,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
        )
        # 直接使用3D卷积来映射时序帧到视频序列tokens，在过程中进行Temporal Sample
        # 对于逐帧计算的Tool以及Phase，怎么处理模型结构的变化？降低视频序列长度并且放弃时序采样

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2)
        x = rearrange(x, "(b t) c k -> b t k c", b=B)

        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        fc_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        all_frames=16,
    ):
        super().__init__()
        embed_dim = 1536
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        backbone = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.reduce_dim = nn.Linear(1536, 1536)
        self.NumberToVector = nn.Linear(1, 1536)
        self.norm1 = norm_layer(embed_dim)
        # Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(embed_dim)
        self.temporal_attn = Attention_Temporal(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.temporal_fc = nn.Linear(embed_dim, embed_dim)

        self.ca_norm = norm_layer(embed_dim)
        self.ca_attn = CrossAttention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.ca_fc = nn.Linear(embed_dim, embed_dim)




        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_swap = nn.Parameter(torch.zeros(1, 5, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)


        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        ## SSM
        config = MambaConfig(d_model=self.embed_dim, n_layers=2)
        self.ssm = Mamba_CSM(config)
        # self.dilate_dim = nn.Linear(3, embed_dim)
        self.head_blocking = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def get_num_layers(self):
    #     return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, timestamp):
        B, _, T, H, W = x.size()

        # 对最后两维进行 average pooling
        pooled_q = F.adaptive_avg_pool2d(x, (1, 1))  # 输出形状是 [2, 3, 20, 1, 1]
        ssm_q = pooled_q.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # 去掉最后两个大小为1的维度，得到 [2, 3, 20]
        # ssm_q = self.dilate_dim(pooled_q)


        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.backbone(x)
        x = torch.squeeze(x)
        x = self.reduce_dim(x)
        x = rearrange(x, "(b t) c -> b t c", b=B)
        xt = x + self.time_embed  # B, T, C
        timestamp_embed = self.NumberToVector(timestamp.unsqueeze(1))
        xt = xt + timestamp_embed.unsqueeze(1)
        xt = self.time_drop(xt)

        groups = rearrange(xt, "b (n k) c -> b n k c", k=4)
        cls_token = self.cls_token_swap.expand(xt.size(0), -1, -1, -1)
        groups_with_cls = torch.cat([cls_token, groups], dim=2)  # (B, 4, 5, C)

        num_interactions = 4  # 假设需要交互 3 次
        for _ in range(num_interactions):
            # 并行对每组进行 self-attention
            # 将 (B, 4, 5, C) 转换为 (B*4, 5, C) 以并行处理
            groups_reshaped = rearrange(groups_with_cls, "b n k c -> (b n) k c")
            updated_groups = self.temporal_attn(self.temporal_norm1(groups_reshaped), B)  # 输出 (B*4, 5, C)
            updated_groups = self.temporal_fc(updated_groups) + groups_reshaped
            # for block in self.blocks:
            #     updated_groups = block(updated_groups, B)
            # updated_groups = self.blocks(groups_reshaped) + groups_reshaped
            groups_with_cls = rearrange(updated_groups, "(b n) k c -> b n k c", b=B)  # 恢复为 (B, 4, 5, C)

            # 提取每组的 CLS token 并交换
            cls_tokens = groups_with_cls[:, :, 0, :]  # (B, 4, C)
            cls_tokens = cls_tokens.roll(shifts=1, dims=1)  # 循环交换 CLS token
            groups_with_cls[:, :, 0, :] = cls_tokens  # 重新附加 CLS token

        groups_without_cls = groups_with_cls[:, :, 1:, :]
        # 结果
        xt = rearrange(groups_without_cls, "b n k c -> b (n k) c")  # (B, 20, C)，将结果展平回去
        # ssm_xt = xt
        # ssm_out = self.ssm(ssm_xt, ssm_q)
        xt = xt[:, -1, :] + x[:, -1, :]

        # fused_xt = self.ca_attn(self.ca_norm(xt.unsqueeze(1)), self.ca_norm(ssm_out), B)  # 输出 (B*4, 5, C)
        # fused_xt = self.ca_fc(fused_xt)[:, -1, :] + xt


        # res_temporal = self.temporal_attn.forward(self.temporal_norm1(xt), B)
        # xt = self.temporal_fc(res_temporal)
        # xt = xt[:, -1, :] + x[:, -1, :]


        return xt




        # # B, C, T, H, W
        # x = self.patch_embed(x)
        # # B, T, K, C
        # B, T, K, C = x.size()
        # W = int(math.sqrt(K))
        #
        # # 添加Spatial Position Embedding
        # x = rearrange(x, "b t k c -> (b t) k c")
        # cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # BT, 1, C
        # x = torch.cat((cls_tokens, x), dim=1)  # BT, HW+1, C  ---> 2*8, 196+1, 768
        # x = x + self.pos_embed  # BT, HW, C  ---> 2*8, 196, 768
        # x = self.pos_drop(x)
        #
        # # 添加Temporal Position Embedding
        # cls_tokens = x[:B, 0, :].unsqueeze(1)
        # x = x[:, 1:]  # 过滤掉cls_tokens
        # x = rearrange(x, "(b t) k c -> (b k) t c", b=B)
        # x = x + self.time_embed  # BK, T, C  ---> 2*196, 8, 768
        # x = self.time_drop(x)
        #
        # # 添加Cls token
        # x = rearrange(x, "(b k) t c -> b (k t) c", b=B)  # Spatial-Temporal tokens
        # x = torch.cat((cls_tokens, x), dim=1)  # 时空tokens对应的class token的添加；
        #
        # for blk in self.blocks:
        #     x = blk(x, B, T, K)
        #
        # x = self.norm(x)
        #
        # return x[:, 0]

    def forward(self, x, timestamp):
        x = self.forward_features(x, timestamp)
        # timestamp_embed = self.NumberToVector((timestamp / 500.).unsqueeze(1))
        # x += timestamp_embed
        x = self.head(self.fc_dropout(x))
        return x


@register_model
def timesformer(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    if pretrained:
        print("Load ckpt from %s" % pretrain_path)
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        state_dict = model.state_dict()
        if "model_state" in checkpoint.keys():
            checkpoint = checkpoint["model_state"]
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[6:] if k.startswith("model") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))

            # if 'time_embed' in checkpoint and state_dict['time_embed'].size(1) != checkpoint['time_embed'].size(1):
            #     print('Resize the Time Embedding, from %s to %s' % (str(checkpoint['time_embed'].size(1)), str(state_dict['time_embed'].size(1))))
            #     time_embed = checkpoint['time_embed'].transpose(1, 2)
            #     new_time_embed = F.interpolate(time_embed, size=(state_dict['time_embed'].size(1)), mode='nearest')
            #     checkpoint['time_embed'] = new_time_embed.transpose(1, 2)
            utils.load_state_dict(model, checkpoint)

        elif "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # strip `model.` prefix
                name = k[8:] if k.startswith("encoder") else k
                new_state_dict[name] = v
            checkpoint = new_state_dict

            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "temporal_attn" in k:
                    k_init = k.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]

            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)

        else:
            add_list = []
            for k in state_dict.keys():
                if "blocks" in k and "temporal_attn" in k:
                    k_init = k.replace("temporal_attn", "attn")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)
                if "blocks" in k and "temporal_norm1" in k:
                    k_init = k.replace("temporal_norm1", "norm1")
                    if k_init in checkpoint:
                        checkpoint[k] = checkpoint[k_init]
                        add_list.append(k)

            print("Adding keys from pretrained checkpoint:", ", ".join(add_list))

            remove_list = []
            for k in state_dict.keys():
                if (
                    ("head" in k or "patch_embed" in k)
                    and k in checkpoint
                    and k in state_dict
                    and checkpoint[k].shape != state_dict[k].shape
                ):
                    remove_list.append(k)
                    del checkpoint[k]
            print(f"Removing keys from pretrained checkpoint:", ", ".join(remove_list))
            utils.load_state_dict(model, checkpoint)

    return model