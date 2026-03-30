import torch
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F


class MemoryBankContrastLoss(nn.Module):
    def __init__(self, engine=None, memory_size=512, feature_dim=None, temperature=0.10,
                 base_temperature=None, momentum=0.999, max_views=512, num_classes=20):
        """Memory-bank based contrastive loss.

        Backward-compatible constructor: previous code created this object with
        keyword args like memory_size, feature_dim, temperature, momentum. We
        accept them here and set internal attributes accordingly.
        """
        super(MemoryBankContrastLoss, self).__init__()
        self.engine = engine
        self.temperature = temperature if temperature is not None else 0.10
        self.base_temperature = base_temperature
        self.ignore_idx = 255
        self.aux_idx = 254
        self.max_views = max_views
        self.momentum = momentum  # momentum factor for updating memory bank

        # Initialize memory bank parameters
        self.num_classes = num_classes
        self.memory_bank = None
        self.memory_bank_size = memory_size
        # feature_dim can be set here or inferred at first batch
        self.feature_dim = feature_dim

    def _init_memory_bank(self, feature_dim, device):
        """Initialize memory bank with zeros on the specified device.

        We register it as a buffer so that .to(device) and state_dict() behave
        as expected.
        """
        # set/confirm feature dim
        if self.feature_dim is None:
            self.feature_dim = feature_dim
        else:
            # ensure consistency
            feature_dim = self.feature_dim

        bank = torch.zeros(self.num_classes, self.memory_bank_size, feature_dim, device=device)
        # register as buffer so it is moved with the module and saved in checkpoints
        try:
            # If memory_bank already exists as buffer this will overwrite it
            self.register_buffer('memory_bank', bank)
        except Exception:
            # fallback to assigning attribute
            self.memory_bank = bank

    def _update_memory_bank(self, features, labels):
        # features: [N, C], labels: [N]
        if self.memory_bank is None:
            # initialize using actual feature dim from features to avoid mismatch
            feat_dim = features.size(1)
            # If user provided a different feature_dim in constructor, warn and override
            if self.feature_dim is not None and self.feature_dim != feat_dim:
                try:
                    print(f"[MemoryBankContrastLoss] warning: provided feature_dim={self.feature_dim} "
                          f"does not match actual feature dim={feat_dim}. Overriding to actual.")
                except Exception:
                    pass
                self.feature_dim = feat_dim
            self._init_memory_bank(feat_dim, features.device)

        with torch.no_grad():
            # Convert aux_idx (254) to last class index using mask assignment to
            # avoid TorchScript/type-checker warnings about mixed argument types
            aux_mask = (labels == self.aux_idx)
            if aux_mask.any():
                labels = labels.clone()
                labels[aux_mask] = self.num_classes - 1

            features = features.detach()

            # For each class, update memory bank slots with random replacement + momentum
            for cls_idx in range(self.num_classes):
                cls_mask = (labels == cls_idx)
                if cls_mask.sum() == 0:
                    continue

                cls_features = features[cls_mask]
                num_new = min(cls_features.size(0), self.memory_bank_size)
                if num_new == 0:
                    continue

                selected_new = cls_features[torch.randperm(cls_features.size(0))[:num_new]]
                replace_idx = torch.randperm(self.memory_bank_size)[:num_new]

                # Momentum update (old on device memory_bank may be on a different device; ensure same)
                self.memory_bank[cls_idx, replace_idx] = (
                    self.momentum * self.memory_bank[cls_idx, replace_idx] +
                    (1 - self.momentum) * selected_new
                )

                # Normalize updated slots
                self.memory_bank[cls_idx, replace_idx] = F.normalize(
                    self.memory_bank[cls_idx, replace_idx], p=2, dim=1
                )

    def forward(self, main_proj, main_gt, aux_proj, aux_gt):  # *args
        """Flexible forward to accept either 4-arg call
        (main_proj, main_gt, aux_proj, aux_gt)
        or 6-arg call
        (main_proj, main_gt, main_pred, aux_proj, aux_gt, aux_pred)
        Older `unimatch_v2_cl_memory.py` calls with 4 args; we support both.
        """
        # map inputs
        # if len(args) == 2:
        #     aux_proj, aux_gt = args
        #     main_pred = None
        #     aux_pred = None
        # elif len(args) == 4:
        #     main_pred, aux_proj, aux_gt, aux_pred = args
        # else:
        #     raise TypeError('MemoryBankContrastLoss.forward expected 4 or 6 args (received %d)' % (2 + len(args)))
        main_gt = torch.nn.functional.interpolate(main_gt.unsqueeze(1).float(),
                                                  size=main_proj.shape[2:],
                                                  mode='nearest').squeeze().long()
        aux_gt = torch.nn.functional.interpolate(aux_gt.unsqueeze(1).float(),
                                                 size=aux_proj.shape[2:],
                                                 mode='nearest').squeeze().long()

        # Normalize the embeddings
        main_proj = torch.nn.functional.normalize(main_proj, p=2, dim=1)
        aux_proj = torch.nn.functional.normalize(aux_proj, p=2, dim=1)

        # Extract features and labels
        main_embd = main_proj.flatten(start_dim=2).permute(0, 2, 1)  # [B, H*W, C]
        main_label = main_gt.flatten(start_dim=1)  # [B, H*W]
        aux_embd = aux_proj.flatten(start_dim=2).permute(0, 2, 1)
        aux_label = aux_gt.flatten(start_dim=1)

        # Combine main and aux features
        all_embd = torch.cat([main_embd, aux_embd], dim=1)  # [B, H*W + H'*W', C]
        all_label = torch.cat([main_label, aux_label], dim=1)  # [B, H*W + H'*W']

        # Flatten batch dimension
        all_embd = all_embd.reshape(-1, all_embd.size(-1))  # [N, C]
        all_label = all_label.reshape(-1)  # [N]

        # Filter out ignore_idx
        valid_mask = (all_label != self.ignore_idx)
        all_embd = all_embd[valid_mask]
        all_label = all_label[valid_mask]

        # Update memory bank with current features
        self._update_memory_bank(all_embd, all_label)

        # Sample anchors from current batch
        anchor_embeds, anchor_labels = self._sample_anchors(all_embd, all_label)

        # Get contrastive features from memory bank
        contrs_embeds, contrs_labels = self._sample_contrastive(device=all_embd.device)

        # Calculate contrastive loss
        if anchor_embeds.nelement() > 0 and contrs_embeds.nelement() > 0:
            loss = self.info_nce(anchors_=anchor_embeds,
                                 a_labels_=anchor_labels.unsqueeze(1),
                                 contras_=contrs_embeds,
                                 c_labels_=contrs_labels.unsqueeze(1))
        else:
            loss = torch.tensor([0.0], device=main_proj.device)

        # Record diagnostics for this forward pass to help debugging/tracing
        try:
            stats = {}
            stats['anchor_count'] = int(anchor_embeds.size(0)) if anchor_embeds.nelement() > 0 else 0
            stats['contras_count'] = int(contrs_embeds.size(0)) if contrs_embeds.nelement() > 0 else 0

            if stats['anchor_count'] > 0 and stats['contras_count'] > 0:
                # per-anchor positive counts
                a_lbl = anchor_labels.unsqueeze(1)  # [A,1]
                c_lbl = contrs_labels.unsqueeze(0)  # [1,C]
                pos_counts = (a_lbl == c_lbl).sum(1).float()  # [A]
                stats['pos_counts_min'] = float(pos_counts.min())
                stats['pos_counts_mean'] = float(pos_counts.mean())
                stats['pos_counts_max'] = float(pos_counts.max())
                stats['valid_anchor_count'] = int((pos_counts > 0).sum())
            else:
                stats['pos_counts_min'] = 0.0
                stats['pos_counts_mean'] = 0.0
                stats['pos_counts_max'] = 0.0
                stats['valid_anchor_count'] = 0

            # memory occupancy per class (non-zero slots)
            if self.memory_bank is None:
                stats['memory_occupancy'] = [0] * self.num_classes
            else:
                with torch.no_grad():
                    occ = (self.memory_bank.norm(dim=2) > 0).sum(dim=1).cpu().tolist()
                stats['memory_occupancy'] = [int(x) for x in occ]

            # store last stats on the module for external inspection
            self._last_stats = stats
        except Exception:
            # never fail the forward due to diagnostics
            self._last_stats = {}

        return loss

    def _sample_anchors(self, embd, label):
        """Sample anchor features from current batch"""
        # Convert aux_idx to last class index using mask assignment to avoid
        # TorchScript/type-checker warnings about mixed argument types
        aux_mask = (label == self.aux_idx)
        if aux_mask.any():
            label = label.clone()
            label[aux_mask] = self.num_classes - 1

        # Sample equal number of features per class
        sampled_embeds = []
        sampled_labels = []

        for cls_idx in range(self.num_classes):
            cls_mask = (label == cls_idx)
            if cls_mask.sum() == 0:
                continue

            cls_embeds = embd[cls_mask]
            sample_num = min(self.max_views, cls_embeds.size(0))

            # Random sample
            selected = cls_embeds[torch.randperm(cls_embeds.size(0))[:sample_num]]
            sampled_embeds.append(selected)
            sampled_labels.append(torch.full((sample_num,), cls_idx,
                                             device=embd.device))

        if len(sampled_embeds) == 0:
            return torch.empty(0, embd.size(1), device=embd.device), \
                torch.empty(0, device=embd.device)

        return torch.cat(sampled_embeds, dim=0), torch.cat(sampled_labels, dim=0)

    def _sample_contrastive(self, device):
        """Sample contrastive features from memory bank"""
        if self.memory_bank is None:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        # Ensure memory bank is on the right device
        memory_bank = self.memory_bank.to(device)

        # Sample from each class in memory bank
        sampled_embeds = []
        sampled_labels = []

        for cls_idx in range(self.num_classes):
            # Get non-zero features from memory bank
            cls_features = memory_bank[cls_idx]
            valid_mask = (cls_features.norm(dim=1) > 0)
            if valid_mask.sum() == 0:
                continue

            valid_features = cls_features[valid_mask]
            sample_num = min(self.max_views, valid_features.size(0))

            # Random sample
            selected = valid_features[torch.randperm(valid_features.size(0))[:sample_num]]
            sampled_embeds.append(selected)
            sampled_labels.append(torch.full((sample_num,), cls_idx,
                                             device=device))

        if len(sampled_embeds) == 0:
            return torch.empty(0, self.feature_dim, device=device), \
                torch.empty(0, device=device)

        return torch.cat(sampled_embeds, dim=0), torch.cat(sampled_labels, dim=0)

    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        """Same as original info_nce implementation"""
        # mask: [num_anchors, num_contras], 1 for positive (same class), 0 otherwise
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()

        # compute logits and stable log-softmax across contrastive dim
        anchor_dot_contrast = torch.div(
            torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
            self.temperature)

        # Use log_softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(anchor_dot_contrast, dim=1)

        # sum log-probs of positives per anchor, but guard against anchors with zero positives
        pos_sum = (mask * log_probs).sum(1)  # [num_anchors]
        pos_count = mask.sum(1)  # number of positives per anchor

        # Only consider anchors that have at least one positive; avoid division by zero
        valid = pos_count > 0
        if valid.sum() == 0:
            # no valid anchors with positives -> return zero loss
            return torch.tensor(0.0, device=anchors_.device)

        mean_log_prob_pos = torch.zeros_like(pos_sum)
        mean_log_prob_pos[valid] = pos_sum[valid] / pos_count[valid].clamp(min=1.0)

        # average over valid anchors only
        return - mean_log_prob_pos[valid].mean()




