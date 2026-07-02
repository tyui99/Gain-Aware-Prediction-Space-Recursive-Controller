# utils/losses.py
# 【损失函数库】统一管理所有损失函数。

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) long
        
        pred = torch.softmax(pred, dim=1)
        
        # Ensure target is long for one_hot
        if target.dtype != torch.long:
            target = target.long()
        
        # One-hot encode target
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        # Ensure target is long for cross_entropy
        if target.dtype != torch.long:
            target = target.long()
            
        logpt = -self.ce(pred, target)
        pt = torch.exp(logpt)
        loss = self.alpha * (1 - pt) ** self.gamma * self.ce(pred, target)
        return loss.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.focal_weight * self.focal(pred, target)

class DiceFocal6040Loss(nn.Module):
    """Dice/Focal hybrid with fixed weights (0.6 Dice, 0.4 Focal)."""

    def __init__(self):
        super().__init__()
        self.inner = DiceFocalLoss(dice_weight=0.6, focal_weight=0.4)

    def forward(self, pred, target):
        return self.inner(pred, target)


class BinaryDiceLoss(nn.Module):
    """Dice loss for one-channel binary logits (B,1,H,W)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred, target):
        if pred.ndim != 4 or pred.shape[1] != 1:
            raise ValueError(f"BinaryDiceLoss expects pred shape (B,1,H,W), got {tuple(pred.shape)}")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target_bin = (target.float() > 0.1).float()
        prob = torch.sigmoid(pred[:, 0, :, :])

        inter = (prob * target_bin).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BinaryBCEDiceLoss(nn.Module):
    """BCEWithLogits + Dice for one-channel binary segmentation."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = BinaryDiceLoss(smooth=smooth)

    def forward(self, pred, target):
        if pred.ndim != 4 or pred.shape[1] != 1:
            raise ValueError(f"BinaryBCEDiceLoss expects pred shape (B,1,H,W), got {tuple(pred.shape)}")
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target_bin = (target.float() > 0.1).float()
        bce = F.binary_cross_entropy_with_logits(pred[:, 0, :, :], target_bin)
        return self.bce_weight * bce + self.dice_weight * self.dice(pred, target)


class ZeroLoss(nn.Module):
    """Explicit no-op supervision used when a method provides its own full objective."""

    def forward(self, pred, _target):
        if isinstance(pred, torch.Tensor):
            return pred.sum() * 0.0
        if isinstance(pred, (list, tuple)):
            for item in pred:
                if isinstance(item, torch.Tensor):
                    return item.sum() * 0.0
        return torch.tensor(0.0, dtype=torch.float32)


class BCEIoULoss(nn.Module):
    """Binary CE + IoU loss for binary/foreground-vs-background tasks.

    Expects logits shaped (B, C, H, W) with C>=2 where channel 1 is foreground.
    Targets can be (B, H, W) integers or floats; values are thresholded at 0.1.
    """

    def __init__(self, bce_weight=0.5, iou_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.eps = eps

    def forward(self, pred, target):
        # Foreground prob from softmax
        fg_prob = torch.softmax(pred, dim=1)[:, 1, :, :]

        # Normalize target to binary mask
        if target.ndim == 4 and target.shape[1] == 1:
            target_proc = target.squeeze(1)
        else:
            target_proc = target

        target_bin = (target_proc.float() > 0.1).float()

        bce = F.binary_cross_entropy(fg_prob, target_bin)

        intersection = (fg_prob * target_bin).sum(dim=(1, 2))
        union = fg_prob.sum(dim=(1, 2)) + target_bin.sum(dim=(1, 2)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        iou_loss = 1.0 - iou.mean()

        return self.bce_weight * bce + self.iou_weight * iou_loss


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore=None):
    # probas: (B,C,H,W), labels: (B,H,W)
    if probas.dim() != 4:
        raise ValueError('probas must have shape (B,C,H,W)')
    _, C, _, _ = probas.shape
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]
    return probas, labels


class LovaszSoftmaxLoss(nn.Module):
    """Multi-class Lovász-Softmax loss (IoU surrogate)."""

    def __init__(self, per_image=False, classes='present', ignore_index=None):
        super().__init__()
        self.per_image = per_image
        self.classes = classes
        self.ignore_index = ignore_index

    def _lovasz_softmax_flat(self, probas, labels):
        if probas.numel() == 0:
            return probas * 0.0

        C = probas.shape[1]
        losses = []
        class_iter = range(C) if self.classes in ['all', 'present'] else self.classes

        for c in class_iter:
            fg = (labels == c).float()
            if self.classes == 'present' and fg.sum() == 0:
                continue
            if probas.shape[1] == 1:
                prob = probas[:, 0]
            else:
                prob = probas[:, c]
            errors = (fg - prob).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = _lovasz_grad(fg_sorted)
            loss = torch.dot(F.relu(errors_sorted), grad)
            losses.append(loss)
        if len(losses) == 0:
            return probas.sum() * 0.0
        return sum(losses) / len(losses)

    def forward(self, probas, labels):
        if self.per_image:
            loss = 0.0
            for prob, lab in zip(probas, labels):
                prob_flat, lab_flat = _flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore_index)
                loss = loss + self._lovasz_softmax_flat(prob_flat, lab_flat)
            return loss / max(1, probas.shape[0])
        prob_flat, lab_flat = _flatten_probas(probas, labels, self.ignore_index)
        return self._lovasz_softmax_flat(prob_flat, lab_flat)


class DiceFocalBoundaryOHEM(nn.Module):
    """Dice + Focal + optional boundary loss with epoch-gated boundary and OHEM.

    - Boundary term activates after `boundary_start_epoch` (inclusive).
    - OHEM keeps top `ohem_ratio` samples (by combined loss) in the batch.
    """

    def __init__(self, dice_weight=0.6, focal_weight=0.4, boundary_weight=0.5,
                 boundary_start_epoch=10, focal_alpha=1.0, focal_gamma=2.0,
                 smooth=1.0, ohem_ratio=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.boundary_start_epoch = int(boundary_start_epoch)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.ohem_ratio = float(ohem_ratio)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _prep_target(self, target):
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.dtype != torch.long:
            target = target.long()
        return target

    def _dice_loss_per_sample(self, prob, target_onehot):
        inter = (prob * target_onehot).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        # average over classes, keep batch dimension
        return loss.mean(dim=1)

    def _focal_loss_per_sample(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')  # (B,H,W)
        pt = torch.exp(-ce)
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce
        # mean over spatial dims per sample
        return focal.mean(dim=(1, 2))

    def _boundary_loss_per_sample(self, fg_prob, target_bin):
        # target_bin: (B,1,H,W) binary mask
        dilated = F.max_pool2d(target_bin, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - target_bin, kernel_size=3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(min=0.0)
        bce = F.binary_cross_entropy(fg_prob, boundary, reduction='none')
        return bce.mean(dim=(1, 2, 3))

    def forward(self, pred, target):
        # pred logits: (B,C,H,W); target: (B,H,W) or (B,1,H,W)
        target_proc = self._prep_target(target)
        prob = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target_proc, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        dice_loss = self._dice_loss_per_sample(prob, target_onehot)
        focal_loss = self._focal_loss_per_sample(pred, target_proc)

        fg_prob = prob[:, 1:2, :, :]
        target_bin = (target_proc.unsqueeze(1) > 0).float()
        if self.current_epoch >= self.boundary_start_epoch and self.boundary_weight > 0:
            boundary_loss = self._boundary_loss_per_sample(fg_prob, target_bin)
            bw = self.boundary_weight
        else:
            boundary_loss = torch.zeros_like(dice_loss)
            bw = 0.0

        total = self.dice_weight * dice_loss + self.focal_weight * focal_loss + bw * boundary_loss

        if self.ohem_ratio < 1.0:
            keep = max(1, math.ceil(total.shape[0] * self.ohem_ratio))
            topk_vals, _ = torch.topk(total, k=keep, largest=True)
            return topk_vals.mean()
        return total.mean()


class DiceFocalBoundaryLovasz(nn.Module):
    """Blend Dice+Focal+Boundary (with gating/OHEM) and Lovasz-Softmax for IoU.

    total = dfb_weight * DFB_OHEM + lovasz_weight * LovaszSoftmax
    """

    def __init__(self, dfb_weight=0.5, lovasz_weight=0.5, **kwargs):
        super().__init__()
        self.dfb = DiceFocalBoundaryOHEM(**kwargs)
        self.lovasz = LovaszSoftmaxLoss()
        self.dfb_weight = float(dfb_weight)
        self.lovasz_weight = float(lovasz_weight)

    def set_epoch(self, epoch: int):
        if hasattr(self.dfb, 'set_epoch'):
            self.dfb.set_epoch(epoch)

    def forward(self, pred, target):
        dfb_loss = self.dfb(pred, target)
        prob = torch.softmax(pred, dim=1)
        lovasz_loss = self.lovasz(prob, target.long())
        return self.dfb_weight * dfb_loss + self.lovasz_weight * lovasz_loss


class DeepSupervisionDiceFocal(nn.Module):
    """Deep supervision wrapper for dice-focal when model returns (main, aux...).

    If preds is a single tensor, falls back to DiceFocal6040Loss.
    If preds is list/tuple, applies base loss to each non-None head with weights.
    """

    def __init__(self, aux_weights=None):
        super().__init__()
        self.base = DiceFocal6040Loss()
        self.aux_weights = aux_weights or [1.0, 0.3, 0.3]

    def forward(self, preds, target):
        if not isinstance(preds, (list, tuple)):
            return self.base(preds, target)
        losses = []
        for i, p in enumerate(preds):
            if p is None:
                continue
            w = self.aux_weights[i] if i < len(self.aux_weights) else 0.0
            if w <= 0:
                continue
            losses.append(w * self.base(p, target))
        if len(losses) == 0:
            return self.base(preds[0], target)
        return sum(losses)

class BoundaryLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) or (B, 1, H, W) binary mask
        
        # 1. Generate GT boundary
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()
        
        # Use max_pool to simulate dilation/erosion for boundary extraction
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        gt_boundary = dilated - eroded
        
        # 2. Generate Pred boundary (approximate)
        # Use softmax probability
        prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]
        
        # Pred boundary can be approximated by spatial gradient or similar dilation/erosion on prob
        # Here we use the same dilation/erosion method on probability map (soft boundary)
        dilated_pred = F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        eroded_pred = -F.max_pool2d(-prob, kernel_size=3, stride=1, padding=1)
        pred_boundary = dilated_pred - eroded_pred
        
        # 3. MSE Loss or Dice on boundary
        # BASNet uses Intersection over Union or Cross Entropy.
        # Let's use simple MSE or L1 for stability, or BCE if we treat it as probability.
        # BASNet: BCE(pred_boundary, gt_boundary)
        
        # Clamp for stability
        pred_boundary = torch.clamp(pred_boundary, 0.0, 1.0)
        gt_boundary = torch.clamp(gt_boundary, 0.0, 1.0)
        
        loss = F.binary_cross_entropy(pred_boundary, gt_boundary)
        return loss

class DiceFocalBoundaryLoss(nn.Module):
    def __init__(self, dice_weight=0.6, focal_weight=0.4, boundary_weight=0.5):
        super().__init__()
        self.base = DiceFocal6040Loss() # 0.6 Dice + 0.4 Focal
        self.boundary = BoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        base_loss = self.base(pred, target)
        bd_loss = self.boundary(pred, target)
        return base_loss + self.boundary_weight * bd_loss

def get_loss(name, **kwargs):
    name = name.lower()
    if name in ('none', 'zero', 'noop', 'null'):
        return ZeroLoss()
    elif name == 'dice':
        return DiceLoss(**kwargs)
    elif name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'dice_focal':
        return DiceFocalLoss(**kwargs)
    elif name in ('dice_focal_06_04', 'dicefocal6040', 'dice_focal_6040'):
        return DiceFocal6040Loss()
    elif name in ('bce_iou', 'bce-iou', 'bceiou'):
        return BCEIoULoss(**kwargs)
    elif name in ('binary_dice', 'dice_binary', 'dice_1c'):
        return BinaryDiceLoss(**kwargs)
    elif name in ('binary_bce_dice', 'bce_dice_binary', 'bcedice_1c'):
        return BinaryBCEDiceLoss(**kwargs)
    elif name in ('dice_focal_boundary_ohem', 'dice_focal_boundary'):
        return DiceFocalBoundaryOHEM(**kwargs)
    elif name in ('dice_focal_boundary_v2', 'boundary_loss'): # New simple version
        return DiceFocalBoundaryLoss(**kwargs)
    elif name in ('lovasz', 'lovasz_softmax', 'iou_lovasz'):
        return LovaszSoftmaxLoss(**kwargs)
    elif name in ('dice_focal_boundary_lovasz', 'dfb_lovasz'):
        return DiceFocalBoundaryLovasz(**kwargs)
    elif name in ('dice_focal_deep_supervision', 'dicefocal_deepsup', 'df_deepsup'):
        return DeepSupervisionDiceFocal(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}")

def _extract_fg_prob(pred):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if pred.ndim != 4:
        raise ValueError(f"Expected pred to be 4D (B,C,H,W), got shape={tuple(pred.shape)}")
    if pred.shape[1] == 1:
        return torch.sigmoid(pred[:, 0, :, :])
    return torch.softmax(pred, dim=1)[:, 1, :, :]


def _normalize_binary_target(target, positive_threshold=0.5):
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    if target.ndim == 4 and target.shape[1] == 1:
        target_proc = target.squeeze(1)
    else:
        target_proc = target

    if target_proc.dtype == torch.long:
        tmax = int(target_proc.max().item()) if target_proc.numel() > 0 else 0
        if tmax > 1:
            target_proc = (target_proc.float() / 255.0 > float(positive_threshold)).long()
        else:
            target_proc = target_proc.long()
    else:
        try:
            target_proc = (target_proc.float() > float(positive_threshold)).long()
        except Exception:
            target_proc = target_proc.long()

    if target_proc.ndim == 2:
        target_proc = target_proc.unsqueeze(0)
    return target_proc


def prepare_binary_masks(pred, target, threshold=0.5, target_threshold=None):
    fg_prob = _extract_fg_prob(pred)
    pred_mask = (fg_prob > threshold).long()
    if target_threshold is None:
        target_threshold = threshold
    target_proc = _normalize_binary_target(target, positive_threshold=float(target_threshold))
    return pred_mask, target_proc


def _surface_mask(mask_2d):
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    surface = np.logical_and(mask_u8 == 1, eroded == 0)
    if not np.any(surface):
        surface = mask_u8.astype(bool)
    return surface


def _distance_transform_to_surface(surface_mask):
    inv_surface = (~surface_mask).astype(np.uint8)
    mask_size = cv2.DIST_MASK_PRECISE if hasattr(cv2, "DIST_MASK_PRECISE") else 5
    return cv2.distanceTransform(inv_surface, cv2.DIST_L2, mask_size)


def _compute_single_sample_hd(pred_mask_2d, target_mask_2d):
    pred_np = (pred_mask_2d > 0).astype(np.uint8)
    target_np = (target_mask_2d > 0).astype(np.uint8)
    pred_has_fg = bool(pred_np.any())
    target_has_fg = bool(target_np.any())
    if not pred_has_fg and not target_has_fg:
        return 0.0, 0.0, "both_empty"

    h, w = pred_np.shape
    if pred_has_fg != target_has_fg:
        diag = float(math.hypot(h, w))
        return diag, diag, "one_empty"

    pred_surface = _surface_mask(pred_np)
    target_surface = _surface_mask(target_np)
    dt_target = _distance_transform_to_surface(target_surface)
    dt_pred = _distance_transform_to_surface(pred_surface)
    d_pred_to_target = dt_target[pred_surface]
    d_target_to_pred = dt_pred[target_surface]
    all_surface_distances = np.concatenate([d_pred_to_target, d_target_to_pred], axis=0).astype(np.float64)
    hd95 = float(np.percentile(all_surface_distances, 95))
    hd = float(np.max(all_surface_distances))
    return hd95, hd, "normal"


def calculate_hd_metrics(pred, target, threshold=0.5):
    pred_mask, target_proc = prepare_binary_masks(pred, target, threshold=threshold)
    pred_np = pred_mask.detach().cpu().numpy()
    target_np = target_proc.detach().cpu().numpy()

    hd95_values = []
    hd_values = []
    normal_cases = 0
    one_empty_cases = 0
    both_empty_cases = 0

    for sample_pred, sample_target in zip(pred_np, target_np):
        hd95_val, hd_val, case_kind = _compute_single_sample_hd(sample_pred, sample_target)
        hd95_values.append(float(hd95_val))
        hd_values.append(float(hd_val))
        if case_kind == "normal":
            normal_cases += 1
        elif case_kind == "one_empty":
            one_empty_cases += 1
        else:
            both_empty_cases += 1

    sample_count = len(hd95_values)
    if sample_count == 0:
        hd95_mean = -1.0
        hd_mean = -1.0
    else:
        hd95_mean = float(sum(hd95_values) / sample_count)
        hd_mean = float(sum(hd_values) / sample_count)

    return {
        "hd95_mean": hd95_mean,
        "hd_mean": hd_mean,
        "hd95_sum": float(sum(hd95_values)),
        "hd_sum": float(sum(hd_values)),
        "sample_count": int(sample_count),
        "normal_cases": int(normal_cases),
        "one_empty_cases": int(one_empty_cases),
        "both_empty_cases": int(both_empty_cases),
        "hd95_values": hd95_values,
        "hd_values": hd_values,
        "unit": "pixel",
    }


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate Dice score for validation with robust handling of input formats.

    Supports models that output either:
      - Binary logits: pred.shape == (B, 1, H, W)  -> use sigmoid
      - Two-class logits: pred.shape == (B, 2, H, W) -> use softmax and class 1

    Target can be:
      - Long tensor in {0,1}
      - Long tensor in {0..255} (will be normalized by 255 and thresholded at 0.1)
      - Float tensor in [0,1] (thresholded at 0.1)

    Returns per-batch average Dice over non-empty GT slices (empty GT slices are ignored and cause
    the returned "dice" to be -1.0 if all slices are empty). Also returns average predicted foreground
    ratio across the batch.
    """
    pred_mask, target_proc = prepare_binary_masks(pred, target, threshold=threshold)
    B = pred_mask.shape[0]

    # Global Dice across whole batch (do not average per-sample)
    # Compute global sums for intersection and foreground counts
    p_flat = (pred_mask == 1).float()
    t_flat = (target_proc == 1).float()

    inter_sum = (p_flat * t_flat).sum()
    pred_sum = p_flat.sum()
    target_sum = t_flat.sum()
    union = pred_sum + target_sum

    # IoU / Recall building blocks
    tp = inter_sum
    fp = pred_sum - inter_sum
    fn = target_sum - inter_sum
    denom_iou = tp + fp + fn
    denom_rec = tp + fn

    # If there is no foreground in both pred and target -> mark as ignored (keep previous sentinel behavior)
    if union.item() == 0:
        avg_dice = -1.0
        avg_iou = -1.0
        avg_recall = -1.0
    else:
        avg_dice = float((2.0 * inter_sum) / (union + 1e-8))
        if denom_iou.item() == 0:
             avg_iou = -1.0
        else:
             avg_iou = float(tp / (denom_iou + 1e-8))
        
        if denom_rec.item() == 0:
             avg_recall = -1.0
        else:
             avg_recall = float(tp / (denom_rec + 1e-8))

    # Global pred foreground ratio (fraction of pixels predicted as foreground across batch)
    total_pixels = float(B * pred_mask.shape[1] * pred_mask.shape[2])
    avg_pred_fg = float(pred_sum) / max(1.0, total_pixels)

    return {
        "dice": avg_dice,
        "pred_fg_ratio": avg_pred_fg,
        "iou": avg_iou,
        "recall": avg_recall,
        # expose global sums so callers can aggregate across batches if desired
        "inter_sum": float(inter_sum),
        "pred_sum": float(pred_sum),
        "target_sum": float(target_sum),
    }


def calculate_multiclass_metrics(pred, target, num_classes=None, ignore_index=-1):
    """Macro mDice/mIoU for multi-class logits.

    pred: (B,C,H,W) logits
    target: (B,H,W) int class ids
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    target = target.long()

    if pred.ndim != 4 or pred.shape[1] < 2:
        raise ValueError(f"calculate_multiclass_metrics expects pred shape (B,C,H,W), C>=2, got {tuple(pred.shape)}")

    c = int(pred.shape[1] if num_classes is None else num_classes)
    pred_cls = torch.argmax(pred, dim=1).long()

    dices = []
    ious = []
    for cls_id in range(c):
        if cls_id == ignore_index:
            continue
        pred_m = (pred_cls == cls_id)
        tgt_m = (target == cls_id)
        valid = (target != ignore_index)
        pred_m = pred_m & valid
        tgt_m = tgt_m & valid

        inter = (pred_m & tgt_m).float().sum()
        ps = pred_m.float().sum()
        ts = tgt_m.float().sum()
        union = ps + ts
        denom_iou = ps + ts - inter

        if union.item() == 0:
            continue
        dices.append(float((2.0 * inter) / (union + 1e-8)))
        ious.append(float(inter / (denom_iou + 1e-8)))

    if len(dices) == 0:
        return {"mdice": -1.0, "miou": -1.0, "valid_classes": 0}
    return {
        "mdice": float(sum(dices) / len(dices)),
        "miou": float(sum(ious) / len(ious)),
        "valid_classes": int(len(dices)),
    }
