import torch
import torch.nn as nn
from functools import partial
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import torch.nn.functional as F
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


def crop_and_pool(images, bboxes):
    """
    使用 bbox 裁剪图片并进行平均池化。

    参数:
        images: 形状为 (B, 3, T, 224, 224) 的张量。
        bboxes: 形状为 (B, T, 2, 2) 的张量，表示每个时间步的 bbox。

    返回:
        形状为 (B, 3, T, 1, 1) 的张量。
    """
    B, C, T, H, W = images.shape
    output = torch.zeros(B, C, T, 1, 1, device=images.device)  # 初始化输出张量

    for b in range(B):  # 遍历 batch
        for t in range(T):  # 遍历时间步
            # 获取当前时间步的 bbox
            x1, y1 = bboxes[b, t, 0, 0], bboxes[b, t, 0, 1]
            x2, y2 = bboxes[b, t, 1, 0], bboxes[b, t, 1, 1]

            # 裁剪图片
            cropped = images[b, :, t, y1:y2, x1:x2]  # 形状为 (3, h, w)
            # 对裁剪后的图片进行全局平均池化
            pooled = F.avg_pool2d(cropped.unsqueeze(0), (cropped.shape[1], cropped.shape[2]))  # 形状为 (1, 3, 1, 1)

            # 将结果存入输出张量
            output[b, :, t] = pooled.squeeze(0)

    return output

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

class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=7,
        num_heads=12,
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
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        backbone = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.reduce_dim = nn.Linear(1536, 1536)
        self.NumberToVector = nn.Linear(1, 1536)
        self.norm1 = norm_layer(embed_dim)

        # knot and release feats
        self.act_feats = torch.zeros(1536).cuda()
        self.release_feats = torch.zeros(1536).cuda()
        self.knot_feats = torch.zeros(1536).cuda()
        self.act_cnt = torch.tensor(0, device='cuda')
        self.release_cnt = torch.tensor(0, device='cuda')
        self.knot_cnt = torch.tensor(0, device='cuda')
        self.alpha = 0.9

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

        # SSM
        self.ca_norm = norm_layer(embed_dim)
        self.ca_attn = CrossAttention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        config = MambaConfig(d_model=self.embed_dim, n_layers=2)
        self.ssm = Mamba_CSM(config)
        self.ca_fc = nn.Linear(embed_dim, embed_dim)

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_swap = nn.Parameter(torch.zeros(1, 5, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)
        self.mask = nn.Parameter(torch.zeros(embed_dim))


        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc_dropout = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.fc_dropout_blocking = (
            nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        )
        self.head_blocking = (
            nn.Linear(embed_dim, 2) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def forward(self, x, timestamp, bboxes, target=None):
        B, _, T, H, W = x.size()

        #  pooling for RGB img/crop
        pooled_q = F.adaptive_avg_pool2d(x, (1, 1))
        pooled_q += crop_and_pool(x, bboxes)
        ssm_q = pooled_q.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.backbone(x)
        x = torch.squeeze(x)
        x = self.reduce_dim(x)
        x = rearrange(x, "(b t) c -> b t c", b=B)
        xt = x + self.time_embed  # B, T, C
        timestamp_embed = self.NumberToVector(timestamp.unsqueeze(1))
        xt = xt + timestamp_embed.unsqueeze(1)
        xt = self.time_drop(xt)

        # temporal pooling
        kernel = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float16).view(1, 1, -1)
        kernel = kernel.repeat(1536, 1, 1).to(xt.device)
        output = F.conv1d(xt.permute(0, 2, 1), kernel, padding=1, groups=1536)
        pooled = output.permute(0, 2, 1)  # (B, N, C)
        pooled_last_group_0 = pooled[:, -1, :]

        # add masks
        if self.act_cnt != 0:
            dot_product_b = torch.matmul(pooled_last_group_0, self.act_feats.to(torch.float16))
            norm_a = torch.norm(pooled_last_group_0, p=2, dim=1)
            norm_act = torch.norm(self.act_feats, p=2)

            similarity_act = dot_product_b / (norm_a * norm_act + 1e-8)
            similarity_idx = (similarity_act < 0)
            time_idx = (timestamp > 0.5)
            mask_idx = similarity_idx * time_idx
            if True in mask_idx:
                xt[mask_idx, -1, :] += self.mask

        groups = rearrange(xt, "b (n k) c -> b n k c", k=4)
        cls_token = self.cls_token_swap.expand(xt.size(0), -1, -1, -1)
        groups_with_cls = torch.cat([cls_token, groups], dim=2)

        num_interactions = 4 # 4 interactions for temporal integration
        for _ in range(num_interactions):
            groups_reshaped = rearrange(groups_with_cls, "b n k c -> (b n) k c")
            updated_groups = self.temporal_attn(self.temporal_norm1(groups_reshaped), B)
            updated_groups = self.temporal_fc(updated_groups) + groups_reshaped
            groups_with_cls = rearrange(updated_groups, "(b n) k c -> b n k c", b=B)

            # CLS token exchange
            cls_tokens = groups_with_cls[:, :, 0, :]
            cls_tokens = cls_tokens.roll(shifts=1, dims=1)
            groups_with_cls[:, :, 0, :] = cls_tokens

        groups_without_cls = groups_with_cls[:, :, 1:, :]

        last_group_feats = groups_without_cls[:, -1, :, :]
        pooled_last_group = last_group_feats[:, -1, :]

        # contrastive learning update prototype
        if self.training:
            act_idx = ((target == 1) + (target == 3))
            knot_idx = (target == 1)
            release_idx = (target == 3)
            if True in act_idx:
                act_cnt = sum(act_idx)
                act_feat = pooled_last_group[act_idx]
                summed_act = act_feat.sum(dim=0)
                # self.act_feats = self.act_feats * (self.act_cnt/(self.act_cnt+act_cnt)) + summed_act/(self.act_cnt+act_cnt)
                self.act_feats = self.act_feats * self.alpha + (1 - self.alpha) * (summed_act / act_cnt)
                self.act_cnt += act_cnt

            if True in knot_idx:
                knot_cnt = sum(knot_idx)
                knot_feat = pooled_last_group[knot_idx]
                summed_knot = knot_feat.sum(dim=0)
                self.knot_feats = self.knot_feats * self.alpha + (1 - self.alpha) * (summed_knot / knot_cnt)
                self.knot_cnt += knot_cnt

            if True in release_idx:
                release_cnt = sum(release_idx)
                release_feat = pooled_last_group[release_idx]
                summed_release = release_feat.sum(dim=0)
                self.release_feats = self.release_feats * self.alpha + (1 - self.alpha) * (summed_release / release_cnt)
                self.release_cnt += release_cnt

        xt = rearrange(groups_without_cls, "b n k c -> b (n k) c")  # (B, 20, C)，将结果展平回去

        ssm_xt = xt
        ssm_out = self.ssm(ssm_xt, ssm_q)

        xt = xt[:, -1, :] + x[:, -1, :]

        fused_xt = self.ca_attn(self.ca_norm(xt.unsqueeze(1)), self.ca_norm(ssm_out), B)  # 输出 (B*4, 5, C)
        fused_xt = self.ca_fc(fused_xt)[:, -1, :] + xt

        output = self.head(self.fc_dropout(fused_xt))
        output_blocking = self.head_blocking(self.fc_dropout_blocking(fused_xt))

        output_idx = output.argmax(-1)
        knot_FP = ((target == 3) * (output_idx == 1))
        release_FP = ((target == 1) * (output_idx == 3))
        contrastive_dict = {}
        if True in knot_FP:
            knot_FP_feats = pooled_last_group[knot_FP]
            contrastive_dict['knot_FP_feats'] = knot_FP_feats
        if True in release_FP:
            release_FP_feats = pooled_last_group[release_FP]
            contrastive_dict['release_FP_feats'] = release_FP_feats

        return output, output_blocking, contrastive_dict

@register_model
def pmnet(pretrained=False, pretrain_path=None, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_heads=12,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()

    return model