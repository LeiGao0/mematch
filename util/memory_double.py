import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class MemoryBankContrastLoss(nn.Module):
    def __init__(self, memory_size=512, feature_dim=None, temperature=0.10,
                 ema_momentum=0.999, main_momentum=0.9, max_views=512, num_classes=20):
        """
        双内存库对比损失（像素级版本）- 外部筛选伪标签版
        - ema_momentum: EMA库（无标签）更新动量（高动量保证稳定）
        - main_momentum: MAIN库（有标签）更新动量（低动量适配扰动）
        说明：置信度筛选已在外部完成，内部仅过滤 ignore_idx=255 的标签
        """
        super(MemoryBankContrastLoss, self).__init__()
        self.temperature = temperature
        self.ignore_idx = 255
        self.aux_idx = 254
        self.max_views = max_views
        self.ema_momentum = ema_momentum
        self.main_momentum = main_momentum

        # 双内存库参数
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.memory_bank_size = memory_size

        # 初始化标记（关键：用布尔值标记是否已初始化，避免重复注册）
        self.ema_bank_init = False
        self.main_bank_init = False

        # 预注册空buffer（解决多GPU环境下device不匹配问题）
        self.register_buffer('ema_memory_bank', torch.empty(0))
        self.register_buffer('main_memory_bank', torch.empty(0))

    def _init_memory_bank(self, feature_dim, device, is_ema):
        """初始化单个内存库（仅初始化一次，避免重复注册）"""
        # 确保feature_dim是正确的（优先使用传入的真实维度）
        final_dim = feature_dim if self.feature_dim is None else self.feature_dim

        # 定义内存库形状
        bank_shape = (self.num_classes, self.memory_bank_size, final_dim)

        # 初始化内存库（用极小值避免全0，保证首次采样有有效特征）
        bank = torch.randn(*bank_shape, device=device) * 1e-4
        bank = F.normalize(bank, p=2, dim=-1)  # 归一化

        # 更新buffer（避免重复注册，直接赋值而非再次register）
        if is_ema and not self.ema_bank_init:
            self.ema_memory_bank = bank
            self.ema_bank_init = True
        elif not is_ema and not self.main_bank_init:
            self.main_memory_bank = bank
            self.main_bank_init = True

    def _update_memory_bank(self, features, labels, is_ema):
        """
        更新单个内存库（仅过滤 ignore_idx=255，置信度筛选已在外部完成）
        - is_ema: True=更新EMA库（无标签），False=更新MAIN库（有标签）
        """
        # 先初始化内存库（仅首次调用时生效）
        self._init_memory_bank(features.size(1), features.device, is_ema)

        # 获取目标内存库
        target_bank = self.ema_memory_bank if is_ema else self.main_memory_bank
        momentum = self.ema_momentum if is_ema else self.main_momentum

        with torch.no_grad():
            # 处理aux_idx（254）映射到最后一类
            labels = labels.clone()
            aux_mask = (labels == self.aux_idx)
            labels[aux_mask] = self.num_classes - 1

            # 仅过滤 ignore_idx=255 的标签
            valid_mask = (labels != self.ignore_idx)
            if not valid_mask.any():
                return

            # 应用有效掩码
            features = features[valid_mask].detach()
            labels = labels[valid_mask]

            # 向量化更新（替代循环，提升性能）
            # 1. 生成每个类别的采样索引
            cls_indices = []
            feat_indices = []
            replace_indices = []

            # 先获取每个类别的特征索引
            for cls_idx in range(self.num_classes):
                cls_mask = (labels == cls_idx)
                cls_feat_idx = torch.nonzero(cls_mask, as_tuple=True)[0]
                if cls_feat_idx.numel() == 0:
                    continue

                # 计算采样数量
                num_new = min(cls_feat_idx.numel(), self.memory_bank_size)
                # 随机采样特征和替换位置
                rand_feat_idx = cls_feat_idx[torch.randperm(cls_feat_idx.numel(), device=features.device)[:num_new]]
                rand_replace_idx = torch.randperm(self.memory_bank_size, device=features.device)[:num_new]

                cls_indices.extend([cls_idx] * num_new)
                feat_indices.append(rand_feat_idx)
                replace_indices.append(rand_replace_idx)

            if not cls_indices:
                return

            # 2. 批量更新内存库
            cls_indices = torch.tensor(cls_indices, device=features.device, dtype=torch.long)
            feat_indices = torch.cat(feat_indices)
            replace_indices = torch.cat(replace_indices)

            # 提取选中的特征并归一化
            selected_feats = F.normalize(features[feat_indices], p=2, dim=1)

            # 动量更新
            target_bank[cls_indices, replace_indices] = (
                    momentum * target_bank[cls_indices, replace_indices] +
                    (1 - momentum) * selected_feats
            )

            # 重新归一化
            target_bank[cls_indices, replace_indices] = F.normalize(
                target_bank[cls_indices, replace_indices], p=2, dim=1
            )

    def _sample_anchors(self, embd, label):
        """像素级锚点采样（和单库逻辑一致）"""
        # 处理aux_idx
        label = label.clone()
        aux_mask = (label == self.aux_idx)
        label[aux_mask] = self.num_classes - 1

        sampled_embeds = []
        sampled_labels = []

        for cls_idx in range(self.num_classes):
            cls_mask = (label == cls_idx)
            if cls_mask.sum() == 0:
                continue

            cls_embeds = embd[cls_mask]
            sample_num = min(self.max_views, cls_embeds.size(0))
            selected = cls_embeds[torch.randperm(cls_embeds.size(0), device=embd.device)[:sample_num]]

            sampled_embeds.append(selected)
            sampled_labels.append(torch.full((sample_num,), cls_idx, device=embd.device))

        if len(sampled_embeds) == 0:
            return torch.empty(0, embd.size(1), device=embd.device), torch.empty(0, device=embd.device)

        return torch.cat(sampled_embeds, dim=0), torch.cat(sampled_labels, dim=0)

    def _sample_contrastive(self, device, is_ema):
        """从单个内存库采样对比特征（像素级）"""
        # 检查初始化状态
        if (is_ema and not self.ema_bank_init) or (not is_ema and not self.main_bank_init):
            return torch.empty(0, self.feature_dim or 0, device=device), torch.empty(0, device=device)

        target_bank = self.ema_memory_bank if is_ema else self.main_memory_bank
        target_bank = target_bank.to(device)

        sampled_embeds = []
        sampled_labels = []

        for cls_idx in range(self.num_classes):
            cls_features = target_bank[cls_idx]
            # 放宽有效特征判断（允许极小值，避免首次采样为空）
            valid_mask = (cls_features.norm(dim=1) > 1e-6)
            if valid_mask.sum() == 0:
                continue

            valid_features = cls_features[valid_mask]
            sample_num = min(self.max_views, valid_features.size(0))
            selected = valid_features[torch.randperm(valid_features.size(0), device=device)[:sample_num]]

            sampled_embeds.append(selected)
            sampled_labels.append(torch.full((sample_num,), cls_idx, device=device))

        if len(sampled_embeds) == 0:
            return torch.empty(0, target_bank.size(-1), device=device), torch.empty(0, device=device)

        return torch.cat(sampled_embeds, dim=0), torch.cat(sampled_labels, dim=0)

    def info_nce(self, anchors, anchor_labels, contras, contra_labels):
        """像素级InfoNCE损失（修复维度错误，提升效率）"""
        if anchors.numel() == 0 or contras.numel() == 0:
            return torch.tensor(0.0, device=anchors.device)

        # 1D标签直接计算mask（修复核心维度问题）
        mask = torch.eq(anchor_labels.unsqueeze(1), contra_labels.unsqueeze(0)).float()

        # 余弦相似度（温度系数）
        anchor_dot_contrast = torch.matmul(anchors, contras.T) / self.temperature

        # 数值稳定处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 计算exp(logits)并防止溢出
        exp_logits = torch.exp(logits)

        # 计算分母（所有样本的和）
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 计算正样本的平均对数概率
        pos_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 过滤无效样本（正样本数为0的情况）
        valid_mask = (mask.sum(1) > 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=anchors.device)

        return -pos_log_prob[valid_mask].mean()

    def forward(self, main_proj, main_gt, aux_proj, aux_gt, epoch=None):
        """
        前向传播（像素级双库对比）- 外部筛选伪标签版
        - main_proj: 主模型有标签像素特征 [B, C, H, W]
        - main_gt: 有标签真实像素标签 [B, H, W]
        - aux_proj: EMA模型无标签像素特征 [B, C, H, W]
        - aux_gt: 无标签伪像素标签（已外部筛选，低置信=255） [B, H, W]
        - epoch: 当前epoch（预留动态权重）
        """
        # 1. 像素级标签下采样（匹配特征尺寸）
        main_gt = F.interpolate(main_gt.unsqueeze(1).float(),
                                size=main_proj.shape[2:],
                                mode='nearest').squeeze().long()
        aux_gt = F.interpolate(aux_gt.unsqueeze(1).float(),
                               size=aux_proj.shape[2:],
                               mode='nearest').squeeze().long()

        # 2. 像素特征归一化
        main_proj = F.normalize(main_proj, p=2, dim=1)
        aux_proj = F.normalize(aux_proj, p=2, dim=1)

        # 3. 像素级特征展平
        main_embd = main_proj.flatten(start_dim=2).permute(0, 2, 1).reshape(-1, main_proj.size(1))
        main_label = main_gt.flatten()
        aux_embd = aux_proj.flatten(start_dim=2).permute(0, 2, 1).reshape(-1, aux_proj.size(1))
        aux_label = aux_gt.flatten()

        # 4. 独立更新双库
        self._update_memory_bank(main_embd, main_label, is_ema=False)
        self._update_memory_bank(aux_embd, aux_label, is_ema=True)

        # 5. 共享像素级锚点
        all_embd = torch.cat([main_embd, aux_embd], dim=0)
        all_label = torch.cat([main_label, aux_label], dim=0)
        valid_mask = (all_label != self.ignore_idx)
        all_embd = all_embd[valid_mask]
        all_label = all_label[valid_mask]
        anchor_embeds, anchor_labels = self._sample_anchors(all_embd, all_label)

        # 6. 分别从双库采样对比特征
        ema_contras_embeds, ema_contras_labels = self._sample_contrastive(device=all_embd.device, is_ema=True)
        main_contras_embeds, main_contras_labels = self._sample_contrastive(device=all_embd.device, is_ema=False)

        # 7. 分别计算双库损失（修复传入标签的维度问题）
        loss_ema = self.info_nce(anchor_embeds, anchor_labels, ema_contras_embeds, ema_contras_labels)
        loss_main = self.info_nce(anchor_embeds, anchor_labels, main_contras_embeds, main_contras_labels)

        # 8. 加权融合损失
        loss_mem_contrast = 0.5 * loss_ema + 0.5 * loss_main

        # 9. 诊断信息（优化：避免CPU/GPU同步）
        try:
            self._last_stats = {
                'anchor_count': anchor_embeds.size(0) if anchor_embeds.numel() > 0 else 0,
                'ema_contras_count': ema_contras_embeds.size(0) if ema_contras_embeds.numel() > 0 else 0,
                'main_contras_count': main_contras_embeds.size(0) if main_contras_embeds.numel() > 0 else 0
            }
        except Exception:
            self._last_stats = {}

        return loss_mem_contrast