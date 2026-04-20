import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class MCAS_MultiHead(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0., num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=attn_bias)
        self.oper_q = nn.ModuleList([
            nn.Sequential(SpatialOperation(self.head_dim), ChannelOperation(self.head_dim))
            for _ in range(num_heads)
        ])
        self.oper_k = nn.ModuleList([
            nn.Sequential(SpatialOperation(self.head_dim), ChannelOperation(self.head_dim))
            for _ in range(num_heads)
        ])
        self.dwc = nn.ModuleList([
            nn.Conv2d(self.head_dim, self.head_dim, 3, 1, 1, groups=self.head_dim)
            for _ in range(num_heads)
        ])
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)


        self.adaptive_norm = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=dim),
            nn.Conv2d(dim, dim, 1, bias=False)
        )
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W).permute(2, 1, 0, 3, 4, 5)
        q, k, v = qkv.unbind(1)

        out_heads = []
        for i in range(self.num_heads):
            q_i = self.oper_q[i](q[i])
            k_i = self.oper_k[i](k[i])
            v_i = v[i]
            out_i = self.dwc[i](q_i + k_i) * v_i
            out_heads.append(out_i)

        out = torch.cat(out_heads, dim=1)
        out = self.proj(out)
        out = self.proj_drop(out) + x  # 残差连接

        out = self.adaptive_norm(out)
        out = out * self.scale
        out = F.normalize(out, p=2, dim=1)

        return out

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
            self,
            nclass,
            in_channels,
            features=256,
            use_bn=False,
            out_channels=[256, 512, 1024, 1024],
            cocl_dim=None,
            MCAS_proj_drop=0.1,
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )

        cocl_out_dim = 64
        mid = max(features // 2, 1)
        self.cocl_head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1, bias=False),
            MCAS_MultiHead(dim=features, proj_drop=MCAS_proj_drop, num_heads=4),
            nn.Conv2d(features, cocl_out_dim, kernel_size=1, bias=True)
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            # permute to [B, N, C] then use last dim as channel C
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        cocl_feat = self.cocl_head(path_1)
        cocl_feat = torch.nn.functional.normalize(cocl_feat, p=2, dim=1)

        return out, cocl_feat


class DPT(nn.Module):
    def __init__(
            self,
            encoder_size='base',
            nclass=21,
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            cocl_dim=None,
            MCAS_proj_drop=0.1,
    ):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)

        self.head = DPTHead(
            nclass,
            self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            cocl_dim=cocl_dim,
            MCAS_proj_drop=MCAS_proj_drop
        )

        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, comp_drop=False):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )

        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]

            device = features[0].device
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2, device=device)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0

            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))

            features = [feature * dropout_mask.unsqueeze(1) for feature in features]

            seg_out, cocl_feat = self.head(features, patch_h, patch_w)
            seg_out = F.interpolate(seg_out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            cocl_feat = F.interpolate(cocl_feat, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)

            return seg_out, cocl_feat

        seg_out, cocl_feat = self.head(features, patch_h, patch_w)
        seg_out = F.interpolate(seg_out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        cocl_feat = F.interpolate(cocl_feat, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        return seg_out, cocl_feat


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPT(
        encoder_size='small',
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        cocl_dim=64,
        MCAS_proj_drop=0.1
    ).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)
    seg_out, cocl_feat = model(x)

    print(f"分割输出形状: {seg_out.shape}")
    print(f"对比学习特征形状: {cocl_feat.shape}")
    print("模型运行正常！")
