import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# 原有导入模块（假设这些模块已存在于对应路径）
from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch

# 导入 CASAtt 相关模块（需确保原 CASAtt 代码可访问）
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


class CASAtt(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out) + x
        return out


class CASAtt_MultiHead_v1(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0., num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # 多头QKV生成：输出维度 = num_heads * 3 * head_dim = 3*dim
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=attn_bias)
        # 每个头独立的空间-通道注意力
        self.oper_q = nn.ModuleList([
            nn.Sequential(SpatialOperation(self.head_dim), ChannelOperation(self.head_dim))
            for _ in range(num_heads)
        ])
        self.oper_k = nn.ModuleList([
            nn.Sequential(SpatialOperation(self.head_dim), ChannelOperation(self.head_dim))
            for _ in range(num_heads)
        ])
        # ========== 关键修复：每个头独立的深度卷积 ==========
        self.dwc = nn.ModuleList([
            nn.Conv2d(self.head_dim, self.head_dim, 3, 1, 1, groups=self.head_dim)
            for _ in range(num_heads)
        ])
        # 多头融合卷积（保持不变）
        self.proj = nn.Conv2d(dim, dim, 1)  # 1x1卷积融合多头特征
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        # QKV拆分：先reshape再按头拆分
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H, W).permute(2, 1, 0, 3, 4, 5)
        # qkv shape: [num_heads, 3, B, head_dim, H, W]
        q, k, v = qkv.unbind(1)  # 每个头的q/k/v: [num_heads, B, head_dim, H, W]

        out_heads = []
        for i in range(self.num_heads):
            q_i = self.oper_q[i](q[i])  # [B, head_dim, H, W]
            k_i = self.oper_k[i](k[i])  # [B, head_dim, H, W]
            v_i = v[i]  # [B, head_dim, H, W]
            # ========== 关键修复：用对应头的dwc ==========
            out_i = self.dwc[i](q_i + k_i) * v_i  # 单头深度卷积
            out_heads.append(out_i)

        # 拼接多头特征: [B, num_heads*head_dim, H, W] = [B, C, H, W]
        out = torch.cat(out_heads, dim=1)
        out = self.proj(out)  # 1x1卷积融合多头
        out = self.proj_drop(out) + x  # 残差连接
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持SpatialOperation/ChannelOperation不变（你的原有代码）
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    def forward(self, x):
        return self.conv(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, 1),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# 修复后的CASAtt_MultiHead（核心修改：归一化逻辑）
class CASAtt_MultiHead(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0., num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # 原有QKV/注意力算子/深度卷积逻辑（保留）
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

        # ========== 终极修复：替换为GroupNorm（无维度问题） ==========
        # GroupNorm适配[B,C,H,W]，无需维度置换，小批次（batch=4）效果更优
        self.adaptive_norm = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=dim),  # 8组，适配dim=64（64/8=8）
            nn.Conv2d(dim, dim, 1, bias=False)             # 微调尺度
        )
        # 可学习缩放因子（匹配记忆库特征尺度）
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 原有多头注意力前向逻辑（保留）
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

        # ========== 归一化+尺度适配（无维度问题） ==========
        out = self.adaptive_norm(out)  # GroupNorm + 1x1卷积
        out = out * self.scale         # 可学习尺度
        out = F.normalize(out, p=2, dim=1)  # L2归一化（匹配记忆库）

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
            casatt_proj_drop=0.1,  # 新增 CASAtt 的 dropout 参数（默认0.1，可调整）
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

        # 对比学习投影头：插入 CASAtt 模块（核心修改）
        cocl_out_dim = 64 # cocl_dim if cocl_dim is not None else features
        mid = max(features // 2, 1)
        self.cocl_head = nn.Sequential(
            nn.Conv2d(features, 64, kernel_size=1, bias=False),  # 通道对齐
            # CASAtt(dim=features, proj_drop=casatt_proj_drop),  # 插入 CASAtt 增强特征
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(features, mid, kernel_size=1, bias=False),  # 瓶颈压缩
            # nn.BatchNorm2d(mid),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid, cocl_out_dim, kernel_size=1, bias=True)  # 最终投影
        )
        # 在DPTHead的cocl_head中替换
        # self.cocl_head = nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=1, bias=False),
        #     CASAtt_MultiHead(dim=features, proj_drop=casatt_proj_drop, num_heads=4),  # 多头注意力
        #     nn.Conv2d(features, cocl_out_dim, kernel_size=1, bias=True)
        # )

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

        # projection for contrastive learning（CASAtt 已在 cocl_head 中生效）
        cocl_feat = self.cocl_head(path_1)
        # L2-normalize embeddings along channel dimension for stable contrastive training
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
            casatt_proj_drop=0.1,  # 新增 CASAtt 的 dropout 参数（传递给 DPTHead）
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

        # 初始化 DPTHead 时传入 CASAtt 的 dropout 参数
        self.head = DPTHead(
            nclass,
            self.backbone.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            cocl_dim=cocl_dim,
            casatt_proj_drop=casatt_proj_drop  # 传递参数
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

            # Convert to list so head can be called multiple times
            features = [feature * dropout_mask.unsqueeze(1) for feature in features]

            seg_out, cocl_feat = self.head(features, patch_h, patch_w)
            seg_out = F.interpolate(seg_out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            cocl_feat = F.interpolate(cocl_feat, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)

            return seg_out, cocl_feat

        seg_out, cocl_feat = self.head(features, patch_h, patch_w)
        seg_out = F.interpolate(seg_out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        cocl_feat = F.interpolate(cocl_feat, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        return seg_out, cocl_feat


# 测试代码（验证是否可正常运行）
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型（使用 small 编码器，减少显存占用）
    model = DPT(
        encoder_size='small',
        nclass=21,
        features=128,
        out_channels=[96, 192, 384, 768],
        cocl_dim=64,
        casatt_proj_drop=0.1
    ).to(device)

    # 测试输入（batch_size=2, channels=3, height=224, width=224）
    x = torch.randn(2, 3, 224, 224).to(device)
    # 前向传播
    seg_out, cocl_feat = model(x)

    # 打印输出形状
    print(f"分割输出形状: {seg_out.shape}")  # 应为 (2, 21, 224, 224)
    print(f"对比学习特征形状: {cocl_feat.shape}")  # 应为 (2, 64, 224, 224)
    print("模型运行正常！")