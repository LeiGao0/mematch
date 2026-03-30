import torch
from abc import ABC
import torch.nn as nn


class MultiClassContrastLoss(nn.Module, ABC):
    def __init__(self, num_classes=19, temperature=0.1, ignore_idx=255, max_views=512):
        super(MultiClassContrastLoss, self).__init__()
        self.temperature = temperature
        self.ignore_idx = ignore_idx  # 忽略的标签值
        self.max_views = max_views  # 最大样本数量
        self.num_classes = num_classes  # 类别总数

    def forward(self, feat1, label1, feat2, label2):
        # 将标签插值到与特征图相同的空间尺寸
        label1 = torch.nn.functional.interpolate(
            label1.unsqueeze(1).float(),
            size=feat1.shape[2:],
            mode='nearest'
        ).squeeze().long()

        label2 = torch.nn.functional.interpolate(
            label2.unsqueeze(1).float(),
            size=feat2.shape[2:],
            mode='nearest'
        ).squeeze().long()

        # 归一化特征
        feat1 = torch.nn.functional.normalize(feat1, p=2, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, p=2, dim=1)

        # 随机提取样本点用于对比学习
        anchor_embeds, anchor_labels, contrs_embeds, contrs_labels = self.extract_samples(
            feat1, label1, feat2, label2
        )

        # 计算对比损失，如果没有有效的样本则返回0
        if anchor_embeds.nelement() > 0 and contrs_embeds.nelement() > 0:
            loss = self.info_nce(
                anchors_=anchor_embeds,
                a_labels_=anchor_labels.unsqueeze(1),
                contras_=contrs_embeds,
                c_labels_=contrs_labels.unsqueeze(1)
            )
        else:
            loss = torch.tensor([0.0], device=feat1.device)

        return loss

    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        # 计算掩码：相同类别为1，不同类别为0
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()

        # 计算点积相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
            self.temperature
        )

        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 避免自对比问题（一个样本与自身对比）
        mask = mask.fill_diagonal_(0.0)

        # 计算分母：所有负样本的指数和
        exp_logits = torch.exp(logits)
        neg_logits_sum = exp_logits * (1 - mask)
        neg_logits_sum = neg_logits_sum.sum(1, keepdim=True)

        # 计算对数概率
        log_prob = logits - torch.log(exp_logits + neg_logits_sum)

        # 计算每个锚点的平均正样本对数概率
        # 处理没有正样本的情况，避免除以0
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.tensor(1.0, device=mask.device), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 返回负的平均对数概率作为损失
        return -mean_log_prob_pos.mean()

    def extract_samples(self, feat1, label1, feat2, label2):
        # 重塑特征和标签的形状，便于采样
        # 特征形状从 [B, C, H, W] 变为 [B, H*W, C]
        feat1 = feat1.flatten(start_dim=2).permute(0, 2, 1)
        label1 = label1.flatten(start_dim=1)  # 标签形状从 [B, H, W] 变为 [B, H*W]

        feat2 = feat2.flatten(start_dim=2).permute(0, 2, 1)
        label2 = label2.flatten(start_dim=1)

        # 过滤掉需要忽略的标签
        valid_mask1 = (label1 != self.ignore_idx)
        valid_feat1 = feat1[valid_mask1]
        valid_label1 = label1[valid_mask1]

        valid_mask2 = (label2 != self.ignore_idx)
        valid_feat2 = feat2[valid_mask2]
        valid_label2 = label2[valid_mask2]

        # 如果有效样本太少，返回空
        if len(valid_feat1) == 0 or len(valid_feat2) == 0:
            return (torch.empty(0, device=feat1.device),
                    torch.empty(0, device=feat1.device),
                    torch.empty(0, device=feat1.device),
                    torch.empty(0, device=feat1.device))

        # 确定要采样的数量（不超过最大视图数和有效样本数）
        sample_num = min(self.max_views, len(valid_feat1), len(valid_feat2))

        # 从第一个特征图中随机采样作为锚点
        perm1 = torch.randperm(len(valid_feat1))[:sample_num]
        anchor_embed = valid_feat1[perm1]
        anchor_label = valid_label1[perm1]

        # 从两个特征图中采样作为对比样本
        perm1_contr = torch.randperm(len(valid_feat1))[:sample_num]
        perm2_contr = torch.randperm(len(valid_feat2))[:sample_num]

        # 组合对比样本
        contrs_embed = torch.cat([
            valid_feat1[perm1_contr],  # 来自第一个特征图的对比样本
            valid_feat2[perm2_contr]  # 来自第二个特征图的对比样本
        ], dim=0)

        contrs_label = torch.cat([
            valid_label1[perm1_contr],  # 来自第一个特征图的对比样本标签
            valid_label2[perm2_contr]  # 来自第二个特征图的对比样本标签
        ], dim=0)

        return anchor_embed, anchor_label, contrs_embed, contrs_label
