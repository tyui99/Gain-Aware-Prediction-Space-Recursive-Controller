from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorLike = Optional[Union[torch.Tensor, Dict[str, torch.Tensor], Sequence[torch.Tensor]]]


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _collect_tensors(payload: TensorLike) -> Sequence[torch.Tensor]:
    items = []
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 4:
            items.append(payload)
        return items
    if isinstance(payload, dict):
        for value in payload.values():
            items.extend(_collect_tensors(value))
        return items
    if isinstance(payload, (list, tuple)):
        for value in payload:
            items.extend(_collect_tensors(value))
    return items


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=_norm_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LazyConvNormAct(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=_norm_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightEvidenceCueEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        cue_mode: str = "single_scale_compressed",
        state_stride: int = 8,
        use_image_context: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.cue_mode = str(cue_mode)
        self.state_stride = max(2, int(state_stride))
        self.use_image_context = bool(use_image_context)
        self.logits_proj = ConvNormAct(num_classes, hidden_dim, kernel_size=1)
        self.image_proj = LazyConvNormAct(hidden_dim, kernel_size=3)
        self.feature_stats_proj = ConvNormAct(2, hidden_dim, kernel_size=1)
        self.merge = LazyConvNormAct(hidden_dim, kernel_size=1)

    @staticmethod
    def _stats(feature: torch.Tensor) -> torch.Tensor:
        mean = feature.mean(dim=1, keepdim=True)
        std = feature.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6).sqrt()
        return torch.cat([mean, std], dim=1)

    def _select_feature_tensor(self, backbone_features: TensorLike) -> Optional[torch.Tensor]:
        tensors = [t for t in _collect_tensors(backbone_features) if t.ndim == 4]
        if len(tensors) == 0:
            return None
        mode = self.cue_mode.lower()
        if mode == "none":
            return None
        if mode == "compressed_pyramid":
            pooled = []
            for feat in tensors:
                pooled.append(self._downsample_to_state_hw(self._stats(feat), feat.shape[-2:]))
            return torch.stack(pooled, dim=0).mean(dim=0)
        feat = tensors[-1]
        return self._downsample_to_state_hw(self._stats(feat), feat.shape[-2:])

    def _state_hw(self, base_hw: Tuple[int, int]) -> Tuple[int, int]:
        h, w = base_hw
        return (
            max(8, int(math.ceil(float(h) / float(self.state_stride)))),
            max(8, int(math.ceil(float(w) / float(self.state_stride)))),
        )

    def _downsample_to_state_hw(self, x: torch.Tensor, base_hw: Tuple[int, int]) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, output_size=self._state_hw(base_hw))

    def forward(
        self,
        base_logits: torch.Tensor,
        input_ref: Optional[torch.Tensor] = None,
        backbone_features: TensorLike = None,
    ) -> torch.Tensor:
        cue_hw = self._state_hw(base_logits.shape[-2:])
        base_small = F.adaptive_avg_pool2d(base_logits, output_size=cue_hw)
        parts = [self.logits_proj(base_small)]

        feature_stats = self._select_feature_tensor(backbone_features)
        if feature_stats is not None:
            if feature_stats.shape[-2:] != cue_hw:
                feature_stats = F.interpolate(feature_stats, size=cue_hw, mode="bilinear", align_corners=False)
            parts.append(self.feature_stats_proj(feature_stats))

        if self.use_image_context and input_ref is not None:
            img = F.adaptive_avg_pool2d(input_ref, output_size=cue_hw)
            parts.append(self.image_proj(img))

        merged = torch.cat(parts, dim=1)
        return self.merge(merged)


class CompressedStateInitializer(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int):
        super().__init__()
        self.init = nn.Sequential(
            ConvNormAct(num_classes + hidden_dim, hidden_dim, kernel_size=3),
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
        )

    def forward(self, base_logits_small: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        return self.init(torch.cat([base_logits_small, cue], dim=1))


class GroupedHybridTransitionCore(nn.Module):
    def __init__(self, hidden_dim: int, groups: int = 4, cheap_core: str = "grouped_hybrid"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = max(1, min(int(groups), hidden_dim))
        while self.hidden_dim % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        self.cheap_core = str(cheap_core)
        group_channels = self.hidden_dim // self.groups

        self.state_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.cue_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.mix_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, groups=self.groups, bias=False)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.group_mixer = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, groups=self.groups, bias=False)
        self.out = ConvNormAct(hidden_dim, hidden_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.group_channels = group_channels

    def forward(self, state: torch.Tensor, cue: torch.Tensor, step_id: int, max_steps: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled_state = F.adaptive_avg_pool2d(state, output_size=1).flatten(1)
        state_energy = state.pow(2).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        cue_energy = cue.pow(2).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        progress = state.new_full((state.shape[0], 1), float(step_id) / float(max(1, max_steps)))
        gate_state, gate_update = self.gate(torch.cat([pooled_state, state_energy, cue_energy, progress], dim=1)).unbind(dim=1)
        keep = (0.55 + 0.40 * torch.sigmoid(gate_state)).view(-1, 1, 1, 1)
        inject = (0.05 + 0.35 * torch.sigmoid(gate_update)).view(-1, 1, 1, 1)

        mixed = torch.cat([self.state_proj(state), self.cue_proj(cue)], dim=1)
        mixed = self.mix_proj(mixed)
        if self.cheap_core in {"grouped_hybrid", "hybrid"}:
            update = self.dw(mixed) + self.group_mixer(mixed)
        elif self.cheap_core in {"grouped_dwconv", "dwconv"}:
            update = self.dw(mixed)
        else:
            update = self.group_mixer(mixed)
        next_state = self.out(keep * state + inject * update)
        aux = {
            "keep_gate": keep.squeeze(-1).squeeze(-1).squeeze(-1),
            "inject_gate": inject.squeeze(-1).squeeze(-1).squeeze(-1),
        }
        return next_state, aux


class GlobalCorrectionHeadLite(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, state: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        delta = self.head(state)
        if delta.shape[-2:] != target_hw:
            delta = F.interpolate(delta, size=target_hw, mode="bilinear", align_corners=False)
        return delta


class LocalResidualAdapterLite(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            ConvNormAct(num_classes + hidden_dim + 1, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1:2, ...]
        return torch.sigmoid(logits)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
        return torch.abs(prob - smooth)

    def forward(self, logits_t: torch.Tensor, cue_up: torch.Tensor) -> torch.Tensor:
        fg = self._fg_prob(logits_t)
        boundary = self._boundary(fg)
        return self.proj(torch.cat([logits_t, cue_up, boundary], dim=1))


class _BudgetController(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        fusion_mode: str = "confidence_scalar",
        alpha_range: Tuple[float, float] = (0.15, 0.55),
        beta_range: Tuple[float, float] = (0.0, 0.25),
        max_steps: int = 5,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = str(fusion_mode)
        self.alpha_range = tuple(alpha_range)
        self.beta_range = tuple(beta_range)
        self.max_steps = max(1, int(max_steps))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.step_alpha = nn.Parameter(torch.zeros(self.max_steps))
        self.step_beta = nn.Parameter(torch.full((self.max_steps,), -1.0))

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1:2, ...]
        return torch.sigmoid(logits)

    def _scale_range(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(x)

    def forward(
        self,
        logits_t: torch.Tensor,
        global_delta: torch.Tensor,
        local_delta: torch.Tensor,
        state: torch.Tensor,
        step_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "fixed_scalar":
            alpha = logits_t.new_full((logits_t.shape[0],), 0.35)
            beta = logits_t.new_full((logits_t.shape[0],), 0.10 if local_delta.abs().sum().item() > 0 else 0.0)
            return alpha, beta

        if self.fusion_mode == "step_scalar":
            idx = min(max(0, int(step_id) - 1), self.max_steps - 1)
            alpha = self._scale_range(self.step_alpha[idx].expand(logits_t.shape[0]), *self.alpha_range)
            beta = self._scale_range(self.step_beta[idx].expand(logits_t.shape[0]), *self.beta_range)
            return alpha, beta

        fg = self._fg_prob(logits_t)
        uncertainty = (fg * (1.0 - fg)).mean(dim=(1, 2, 3))
        stats = torch.cat(
            [
                F.adaptive_avg_pool2d(state, output_size=1).flatten(1),
                uncertainty.unsqueeze(1),
                global_delta.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                local_delta.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                logits_t.new_full((logits_t.shape[0], 1), float(step_id) / float(self.max_steps)),
            ],
            dim=1,
        )
        alpha_logit, beta_logit = self.mlp(stats).unbind(dim=1)
        alpha = self._scale_range(alpha_logit, *self.alpha_range)
        beta = self._scale_range(beta_logit, *self.beta_range)
        return alpha, beta


class GainAwareLiteHalt(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.75,
        patience: int = 2,
        halt_mode: str = "gain_aware_act",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.threshold = float(threshold)
        self.patience = max(1, int(patience))
        self.halt_mode = str(halt_mode)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        logits_prev: torch.Tensor,
        logits_next: torch.Tensor,
        state_delta: torch.Tensor,
        uncertainty: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        step_id: int,
        max_steps: int,
    ) -> torch.Tensor:
        _ = step_id, max_steps
        if self.halt_mode == "double_evidence_patience":
            return ((state_delta < 1e-3) & (uncertainty_drop < 1e-3)).float()
        pooled = F.adaptive_avg_pool2d(state, output_size=1).flatten(1)
        delta_pred = (logits_next - logits_prev).pow(2).mean(dim=(1, 2, 3)).sqrt().unsqueeze(1)
        stats = torch.cat(
            [
                pooled,
                state_delta.unsqueeze(1),
                uncertainty.unsqueeze(1),
                uncertainty_drop.unsqueeze(1),
                delta_pred,
            ],
            dim=1,
        )
        return torch.sigmoid(self.head(stats)).squeeze(1)


class LightweightRDTTracePack:
    @staticmethod
    def build(
        step_id: int,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state_delta: torch.Tensor,
        uncertainty: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        halt_prob: torch.Tensor,
        halt_reason: str,
        executed_steps: int,
        keep_gate: Optional[torch.Tensor] = None,
        inject_gate: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        return {
            "step_id": int(step_id),
            "executed_steps": int(executed_steps),
            "alpha_mean": float(alpha.mean().detach().item()),
            "beta_mean": float(beta.mean().detach().item()),
            "state_delta": float(state_delta.mean().detach().item()),
            "uncertainty": float(uncertainty.mean().detach().item()),
            "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
            "halt_metric_value": float(halt_prob.mean().detach().item()),
            "halt_reason": str(halt_reason),
            "halt_metric": "halt_probability",
            "keep_gate": float(keep_gate.mean().detach().item()) if isinstance(keep_gate, torch.Tensor) else 0.0,
            "inject_gate": float(inject_gate.mean().detach().item()) if isinstance(inject_gate, torch.Tensor) else 0.0,
        }



from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fg_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] > 1:
        return torch.softmax(logits, dim=1)[:, 1:2, ...]
    return torch.sigmoid(logits)


def boundary_strength(prob: torch.Tensor) -> torch.Tensor:
    smooth = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
    return torch.abs(prob - smooth)


def gather_anchor_stats(logits_t: torch.Tensor, base_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    eps = 1e-6
    prob_t = fg_prob(logits_t)
    prob_0 = fg_prob(base_logits)
    delta_map = torch.abs(prob_t - prob_0)
    uncertainty_map = prob_t * (1.0 - prob_t)
    boundary_map = torch.maximum(boundary_strength(prob_t), boundary_strength(prob_0))
    boundary_var = boundary_map.var(dim=(1, 2, 3), unbiased=False)
    overlap_ub = (uncertainty_map * boundary_map).mean(dim=(1, 2, 3))
    overlap_db = (delta_map * boundary_map).mean(dim=(1, 2, 3))
    boundary_confidence = (overlap_ub / (boundary_map.mean(dim=(1, 2, 3)) + eps)).clamp(0.0, 1.0)
    uncertainty_boundary_ratio = (overlap_ub / (uncertainty_map.mean(dim=(1, 2, 3)) + eps)).clamp(0.0, 1.5)
    delta_boundary_ratio = (overlap_db / (delta_map.mean(dim=(1, 2, 3)) + eps)).clamp(0.0, 1.5)
    return {
        "prob_t": prob_t,
        "prob_0": prob_0,
        "delta_map": delta_map,
        "uncertainty_map": uncertainty_map,
        "boundary_map": boundary_map,
        "delta_mean": delta_map.mean(dim=(1, 2, 3)),
        "uncertainty_mean": uncertainty_map.mean(dim=(1, 2, 3)),
        "boundary_mean": boundary_map.mean(dim=(1, 2, 3)),
        "boundary_var": boundary_var,
        "uncertainty_boundary_overlap": overlap_ub,
        "delta_boundary_overlap": overlap_db,
        "boundary_confidence": boundary_confidence,
        "uncertainty_boundary_ratio": uncertainty_boundary_ratio,
        "delta_boundary_ratio": delta_boundary_ratio,
    }


class MicroStateUpdate(nn.Module):
    def __init__(
        self,
        micro_state_dim: int = 4,
        rank: int = 2,
        gain_enhanced: bool = False,
        gain_decomposition: bool = False,
    ):
        super().__init__()
        self.micro_state_dim = max(2, int(micro_state_dim))
        self.rank = max(1, min(int(rank), self.micro_state_dim))
        self.gain_enhanced = bool(gain_enhanced)
        self.gain_decomposition = bool(gain_decomposition)
        extra_gain_dim = 2 if self.gain_enhanced else 0
        extra_gain_dim += 2 if self.gain_decomposition else 0
        in_dim = self.micro_state_dim + 4 + extra_gain_dim
        self.in_proj = nn.Linear(in_dim, self.rank, bias=False)
        self.out_proj = nn.Linear(self.rank, self.micro_state_dim, bias=True)
        self.keep_proj = nn.Linear(in_dim, 1, bias=True)
        self.inject_proj = nn.Linear(in_dim, 1, bias=True)
        self.gain_proj = nn.Linear(in_dim, self.micro_state_dim, bias=True) if self.gain_enhanced else None

    def forward(
        self,
        micro_state: torch.Tensor,
        delta_mean: torch.Tensor,
        uncertainty_mean: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        step_ratio: torch.Tensor,
        gain_ratio: Optional[torch.Tensor] = None,
        gain_ema: Optional[torch.Tensor] = None,
        gain_delta: Optional[torch.Tensor] = None,
        budget_gain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        items = [
            micro_state,
            delta_mean.unsqueeze(1),
            uncertainty_mean.unsqueeze(1),
            uncertainty_drop.unsqueeze(1),
            step_ratio.unsqueeze(1),
        ]
        if self.gain_enhanced:
            if gain_ratio is None:
                gain_ratio = uncertainty_drop / uncertainty_mean.clamp_min(1e-6)
            if gain_ema is None:
                gain_ema = gain_ratio
            items.extend([gain_ratio.unsqueeze(1), gain_ema.unsqueeze(1)])
        if self.gain_decomposition:
            if gain_ratio is None:
                gain_ratio = uncertainty_drop / uncertainty_mean.clamp_min(1e-6)
            if gain_ema is None:
                gain_ema = gain_ratio
            if gain_delta is None:
                gain_delta = gain_ratio - gain_ema
            if budget_gain is None:
                budget_gain = gain_ratio * (1.0 - step_ratio).clamp_min(0.0)
            items.extend([gain_delta.unsqueeze(1), budget_gain.unsqueeze(1)])
        stats = torch.cat(items, dim=1)
        update = self.out_proj(self.in_proj(stats))
        gain_feature = torch.zeros_like(micro_state)
        if self.gain_proj is not None:
            gain_feature = torch.tanh(self.gain_proj(stats))
            update = update + 0.25 * gain_feature
        update = torch.tanh(update)
        keep = 0.65 + 0.30 * torch.sigmoid(self.keep_proj(stats))
        inject = 0.05 + 0.20 * torch.sigmoid(self.inject_proj(stats))
        next_state = keep * micro_state + inject * update
        aux = {
            "keep_gate": keep.squeeze(1),
            "inject_gate": inject.squeeze(1),
            "gain_feature": gain_feature.abs().mean(dim=1),
            "gain_delta": torch.zeros_like(delta_mean) if gain_delta is None else gain_delta,
            "budget_gain": torch.zeros_like(delta_mean) if budget_gain is None else budget_gain,
        }
        return next_state, aux


class LowRankGlobalControl(nn.Module):
    def __init__(
        self,
        micro_state_dim: int = 4,
        rank: int = 2,
        num_classes: int = 2,
        mode: str = "scalar_lowrank",
        groups: int = 2,
        spatial_rank: int = 1,
    ):
        super().__init__()
        self.micro_state_dim = max(2, int(micro_state_dim))
        self.rank = max(1, min(int(rank), self.micro_state_dim))
        self.num_classes = int(num_classes)
        self.mode = str(mode)
        self.groups = max(1, int(groups))
        self.spatial_rank = max(1, int(spatial_rank))
        self.proj_in = nn.Linear(self.micro_state_dim, self.rank, bias=False)
        self.proj_scale = nn.Linear(self.rank, 2, bias=True)
        self.class_scale = nn.Parameter(torch.zeros(self.num_classes))
        self.group_scale = None
        self.group_u = None
        self.group_v = None
        if self.mode == "groupwise_lowrank":
            self.group_scale = nn.Linear(self.rank, 2 * self.groups * self.groups, bias=True)
        elif self.mode in {"groupwise_spatial_lowrank", "groupwise_rank1", "groupwise_rank2"}:
            if self.mode == "groupwise_rank1":
                self.spatial_rank = 1
            elif self.mode == "groupwise_rank2":
                self.spatial_rank = max(2, self.spatial_rank)
            self.group_u = nn.Linear(self.rank, 2 * self.spatial_rank * self.groups, bias=True)
            self.group_v = nn.Linear(self.rank, 2 * self.spatial_rank * self.groups, bias=True)

    def forward(
        self,
        logits_t: torch.Tensor,
        base_logits: torch.Tensor,
        micro_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        latent = self.proj_in(micro_state)
        gain_base, gain_center = self.proj_scale(latent).unbind(dim=1)
        gain_base = 0.02 + 0.25 * torch.sigmoid(gain_base)
        gain_center = 0.01 + 0.15 * torch.sigmoid(gain_center)
        centered = logits_t - logits_t.mean(dim=(2, 3), keepdim=True)
        class_gain = (0.5 + torch.tanh(self.class_scale)).view(1, -1, 1, 1)
        spatial_base = torch.ones_like(logits_t[:, :1, ...])
        spatial_center = torch.ones_like(logits_t[:, :1, ...])
        spatial_code_std = torch.zeros_like(gain_base)
        if self.group_scale is not None:
            group_logits = self.group_scale(latent).view(-1, 2, self.groups, self.groups)
            spatial_base = 0.8 + 0.4 * torch.sigmoid(
                F.interpolate(group_logits[:, 0:1, ...], size=logits_t.shape[-2:], mode="bilinear", align_corners=False)
            )
            spatial_center = 0.8 + 0.4 * torch.sigmoid(
                F.interpolate(group_logits[:, 1:2, ...], size=logits_t.shape[-2:], mode="bilinear", align_corners=False)
            )
            spatial_code_std = group_logits.flatten(2).std(dim=2, unbiased=False).mean(dim=1)
        elif self.group_u is not None and self.group_v is not None:
            left = self.group_u(latent).view(-1, 2, self.spatial_rank, self.groups, 1)
            right = self.group_v(latent).view(-1, 2, self.spatial_rank, 1, self.groups)
            group_logits = (left * right).sum(dim=2)
            spatial_base = 0.75 + 0.5 * torch.sigmoid(
                F.interpolate(group_logits[:, 0:1, ...], size=logits_t.shape[-2:], mode="bilinear", align_corners=False)
            )
            spatial_center = 0.75 + 0.5 * torch.sigmoid(
                F.interpolate(group_logits[:, 1:2, ...], size=logits_t.shape[-2:], mode="bilinear", align_corners=False)
            )
            spatial_code_std = group_logits.flatten(2).std(dim=2, unbiased=False).mean(dim=1)
        delta = (
            gain_base.view(-1, 1, 1, 1) * spatial_base * (base_logits - logits_t)
            + gain_center.view(-1, 1, 1, 1) * spatial_center * centered * class_gain
        )
        aux = {
            "global_gain_center": gain_center,
            "spatial_base_mean": spatial_base.mean(dim=(1, 2, 3)),
            "spatial_center_mean": spatial_center.mean(dim=(1, 2, 3)),
            "spatial_code_std": spatial_code_std,
        }
        return delta, gain_base, aux


class EdgeLocalGate(nn.Module):
    def __init__(
        self,
        micro_state_dim: int = 4,
        enhanced_stats: bool = False,
        confidence_controller: bool = False,
        local_op: str = "edge_restore",
    ):
        super().__init__()
        self.enhanced_stats = bool(enhanced_stats)
        self.confidence_controller = bool(confidence_controller)
        self.local_op = str(local_op)
        extra_dim = 6 if self.enhanced_stats else 3
        if self.confidence_controller:
            extra_dim += 3
        self.weight = nn.Linear(micro_state_dim + extra_dim, 1, bias=True)
        self.mix_head = nn.Linear(micro_state_dim + extra_dim, 1, bias=True) if self.confidence_controller else None
        self.restore_head = nn.Linear(micro_state_dim + extra_dim, 1, bias=True) if self.confidence_controller else None

    def forward(
        self,
        logits_t: torch.Tensor,
        base_logits: torch.Tensor,
        micro_state: torch.Tensor,
        boundary_mean: torch.Tensor,
        boundary_var: torch.Tensor,
        uncertainty_mean: torch.Tensor,
        delta_mean: torch.Tensor,
        uncertainty_boundary_overlap: torch.Tensor,
        delta_boundary_overlap: torch.Tensor,
        boundary_confidence: torch.Tensor,
        uncertainty_boundary_ratio: torch.Tensor,
        delta_boundary_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        stats = [micro_state, boundary_mean.unsqueeze(1), uncertainty_mean.unsqueeze(1), delta_mean.unsqueeze(1)]
        if self.enhanced_stats:
            stats.extend(
                [
                    boundary_var.unsqueeze(1),
                    uncertainty_boundary_overlap.unsqueeze(1),
                    delta_boundary_overlap.unsqueeze(1),
                ]
            )
        if self.confidence_controller:
            stats.extend(
                [
                    boundary_confidence.unsqueeze(1),
                    uncertainty_boundary_ratio.unsqueeze(1),
                    delta_boundary_ratio.unsqueeze(1),
                ]
            )
        local_stats = torch.cat(stats, dim=1)
        gate = 0.01 + 0.14 * torch.sigmoid(self.weight(local_stats)).squeeze(1)
        edge = logits_t - F.avg_pool2d(logits_t, kernel_size=3, stride=1, padding=1)
        if self.local_op == "axial_edge_restore":
            horiz = logits_t - F.avg_pool2d(logits_t, kernel_size=(1, 3), stride=1, padding=(0, 1))
            vert = logits_t - F.avg_pool2d(logits_t, kernel_size=(3, 1), stride=1, padding=(1, 0))
            edge = 0.5 * (horiz + vert)
        elif self.local_op == "cheap_residual":
            edge = logits_t - F.avg_pool2d(logits_t, kernel_size=5, stride=1, padding=2)
        restore = base_logits - logits_t
        edge_mix = torch.full_like(gate, 0.70)
        restore_gain = torch.ones_like(gate)
        if self.enhanced_stats:
            overlap_strength = (uncertainty_boundary_overlap + delta_boundary_overlap).clamp(0.0, 1.0)
            edge_mix = 0.55 + 0.25 * overlap_strength
        if self.mix_head is not None and self.restore_head is not None:
            edge_mix = 0.40 + 0.40 * torch.sigmoid(self.mix_head(local_stats)).squeeze(1)
            restore_gain = 0.75 + 0.35 * torch.sigmoid(self.restore_head(local_stats)).squeeze(1)
        local_delta = gate.view(-1, 1, 1, 1) * (
            edge_mix.view(-1, 1, 1, 1) * edge
            + (1.0 - edge_mix).view(-1, 1, 1, 1) * restore_gain.view(-1, 1, 1, 1) * restore
        )
        aux = {
            "edge_mix": edge_mix,
            "boundary_var": boundary_var,
            "restore_gain": restore_gain,
            "boundary_confidence": boundary_confidence,
            "local_op_code": torch.full_like(gate, 1.0 if self.local_op == "axial_edge_restore" else (2.0 if self.local_op == "cheap_residual" else 0.0)),
        }
        return local_delta, gate, aux


class _ScalarBlendController(nn.Module):
    def __init__(
        self,
        max_steps: int = 5,
        micro_state_dim: int = 4,
        mode: str = "step_scalar",
    ):
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.micro_state_dim = max(2, int(micro_state_dim))
        self.mode = str(mode)
        self.alpha = nn.Parameter(torch.zeros(self.max_steps))
        self.beta = nn.Parameter(torch.full((self.max_steps,), -2.0))
        self.state_head = None
        if self.mode == "state_aware":
            # Keep the fusion controller tiny: state + a handful of scalar evidence.
            self.state_head = nn.Linear(self.micro_state_dim + 5, 2, bias=True)

    def forward(
        self,
        batch_size: int,
        step_id: int,
        device: torch.device,
        micro_state: Optional[torch.Tensor] = None,
        step_ratio: Optional[torch.Tensor] = None,
        gain_ratio: Optional[torch.Tensor] = None,
        boundary_confidence: Optional[torch.Tensor] = None,
        global_gain: Optional[torch.Tensor] = None,
        local_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        idx = min(max(0, int(step_id) - 1), self.max_steps - 1)
        alpha = 0.10 + 0.35 * torch.sigmoid(self.alpha[idx]).expand(batch_size).to(device)
        beta = 0.00 + 0.18 * torch.sigmoid(self.beta[idx]).expand(batch_size).to(device)
        aux = {
            "fusion_policy_shift": torch.zeros(batch_size, device=device),
            "fusion_mode_code": torch.full(
                (batch_size,),
                1.0 if self.mode == "state_aware" else 0.0,
                device=device,
            ),
        }
        if self.state_head is None:
            return alpha, beta, aux

        if micro_state is None:
            micro_state = torch.zeros(batch_size, self.micro_state_dim, device=device)
        if step_ratio is None:
            step_ratio = torch.full((batch_size,), float(step_id) / float(max(1, self.max_steps)), device=device)
        if gain_ratio is None:
            gain_ratio = torch.zeros(batch_size, device=device)
        if boundary_confidence is None:
            boundary_confidence = torch.zeros(batch_size, device=device)
        if global_gain is None:
            global_gain = torch.zeros(batch_size, device=device)
        if local_gate is None:
            local_gate = torch.zeros(batch_size, device=device)

        stats = torch.cat(
            [
                micro_state,
                step_ratio.unsqueeze(1),
                gain_ratio.unsqueeze(1),
                boundary_confidence.unsqueeze(1),
                global_gain.unsqueeze(1),
                local_gate.unsqueeze(1),
            ],
            dim=1,
        )
        delta_alpha, delta_beta = self.state_head(stats).unbind(dim=1)
        delta_alpha = 0.08 * torch.tanh(delta_alpha)
        delta_beta = 0.05 * torch.tanh(delta_beta)
        alpha = (alpha + delta_alpha).clamp(0.08, 0.48)
        beta = (beta + delta_beta).clamp(0.0, 0.22)
        aux["fusion_policy_shift"] = (delta_alpha.abs() + delta_beta.abs()) * 0.5
        return alpha, beta, aux


class TinyHaltController(nn.Module):
    def __init__(
        self,
        micro_state_dim: int = 4,
        stabilized: bool = False,
        gain_ema_input: bool = False,
        budget_aware: bool = False,
    ):
        super().__init__()
        self.stabilized = bool(stabilized)
        self.gain_ema_input = bool(gain_ema_input)
        self.budget_aware = bool(budget_aware)
        extra_dim = 6 if self.stabilized else 4
        if self.gain_ema_input:
            extra_dim += 2
        if self.budget_aware:
            extra_dim += 2
        self.head = nn.Linear(micro_state_dim + extra_dim, 1, bias=True)

    def forward(
        self,
        micro_state: torch.Tensor,
        delta_mean: torch.Tensor,
        uncertainty_mean: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        step_ratio: torch.Tensor,
        stabilized_delta: Optional[torch.Tensor] = None,
        stabilized_drop: Optional[torch.Tensor] = None,
        gain_ema: Optional[torch.Tensor] = None,
        gain_ratio: Optional[torch.Tensor] = None,
        remaining_budget: Optional[torch.Tensor] = None,
        budget_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        stats = [
            micro_state,
            delta_mean.unsqueeze(1),
            uncertainty_mean.unsqueeze(1),
            uncertainty_drop.unsqueeze(1),
            step_ratio.unsqueeze(1),
        ]
        if self.stabilized:
            if stabilized_delta is None:
                stabilized_delta = delta_mean
            if stabilized_drop is None:
                stabilized_drop = uncertainty_drop
            stats.extend([stabilized_delta.unsqueeze(1), stabilized_drop.unsqueeze(1)])
        if self.gain_ema_input:
            if gain_ema is None:
                gain_ema = uncertainty_drop
            if gain_ratio is None:
                gain_ratio = uncertainty_drop / uncertainty_mean.clamp_min(1e-6)
            stats.extend([gain_ema.unsqueeze(1), gain_ratio.unsqueeze(1)])
        if self.budget_aware:
            if remaining_budget is None:
                remaining_budget = (1.0 - step_ratio).clamp_min(0.0)
            if budget_value is None:
                base_gain = gain_ratio if gain_ratio is not None else (uncertainty_drop / uncertainty_mean.clamp_min(1e-6))
                budget_value = remaining_budget * base_gain
            stats.extend([remaining_budget.unsqueeze(1), budget_value.unsqueeze(1)])
        stats = torch.cat(stats, dim=1)
        return torch.sigmoid(self.head(stats)).squeeze(1)


class MicroStateTracePack:
    @staticmethod
    def build(
        step_id: int,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state_delta: torch.Tensor,
        delta_mean: torch.Tensor,
        uncertainty_mean: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        halt_prob: torch.Tensor,
        global_gain: torch.Tensor,
        local_gate: torch.Tensor,
        gain_ratio: Optional[torch.Tensor] = None,
        gain_ema: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        gain_ratio = torch.zeros_like(delta_mean) if gain_ratio is None else gain_ratio
        gain_ema = torch.zeros_like(delta_mean) if gain_ema is None else gain_ema
        return {
            "step_id": int(step_id),
            "alpha_mean": float(alpha.mean().detach().item()),
            "beta_mean": float(beta.mean().detach().item()),
            "state_delta": float(state_delta.mean().detach().item()),
            "delta_mean": float(delta_mean.mean().detach().item()),
            "uncertainty": float(uncertainty_mean.mean().detach().item()),
            "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
            "halt_metric_value": float(halt_prob.mean().detach().item()),
            "global_gain": float(global_gain.mean().detach().item()),
            "local_gate": float(local_gate.mean().detach().item()),
            "gain_ratio": float(gain_ratio.mean().detach().item()),
            "gain_ema": float(gain_ema.mean().detach().item()),
        }



import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter



TensorLike = Optional[Union[torch.Tensor, Dict[str, torch.Tensor], Sequence[torch.Tensor]]]


def _first_tensor(payload: TensorLike) -> Optional[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        return payload if payload.ndim == 4 else None
    if isinstance(payload, dict):
        for value in payload.values():
            found = _first_tensor(value)
            if found is not None:
                return found
        return None
    if isinstance(payload, (list, tuple)):
        for value in reversed(payload):
            found = _first_tensor(value)
            if found is not None:
                return found
    return None


def _collect_tensors(payload: TensorLike) -> Sequence[torch.Tensor]:
    items = []
    if isinstance(payload, torch.Tensor):
        if payload.ndim == 4:
            items.append(payload)
        return items
    if isinstance(payload, dict):
        for value in payload.values():
            items.extend(_collect_tensors(value))
        return items
    if isinstance(payload, (list, tuple)):
        for value in payload:
            items.extend(_collect_tensors(value))
    return items


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _safe_param_count(module: nn.Module) -> float:
    total = 0
    for p in module.parameters():
        if isinstance(p, UninitializedParameter):
            continue
        total += int(p.numel())
    return float(total) / 1e6


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=_norm_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LazyConvNormAct(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(num_groups=_norm_groups(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureAdapterContract(nn.Module):
    """
    Normalize heterogeneous backbone feature payloads into a stable evidence tensor.

    Contract modes:
    - none: ignore backbone features
    - last_feature: use only the last available 4D feature map
    - pyramid_feature: use all available 4D feature maps
    - auto: single feature -> last_feature, multi feature -> pyramid_feature
    """

    def __init__(self, hidden_dim: int, feature_contract: str = "auto", pyramid_merge: str = "mean"):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_contract = str(feature_contract)
        self.pyramid_merge = str(pyramid_merge)
        self.stats_proj = ConvNormAct(2, hidden_dim, kernel_size=3)
        self.merge = ConvNormAct(hidden_dim, hidden_dim, kernel_size=1)

    @staticmethod
    def _feature_stats(feature: torch.Tensor) -> torch.Tensor:
        mean = feature.mean(dim=1, keepdim=True)
        std = feature.var(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6).sqrt()
        return torch.cat([mean, std], dim=1)

    def _select_mode(self, tensors: Sequence[torch.Tensor]) -> str:
        mode = self.feature_contract.lower()
        if mode != "auto":
            return mode
        if len(tensors) <= 0:
            return "none"
        if len(tensors) == 1:
            return "last_feature"
        return "pyramid_feature"

    def forward(self, backbone_features: TensorLike, target_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        tensors = [t for t in _collect_tensors(backbone_features) if t.ndim == 4]
        mode = self._select_mode(tensors)
        if mode == "none" or len(tensors) == 0:
            return None
        if mode == "last_feature":
            tensors = [tensors[-1]]

        outputs = []
        for feat in tensors:
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            outputs.append(self.stats_proj(self._feature_stats(feat)))
        if len(outputs) == 1:
            merged = outputs[0]
        elif self.pyramid_merge == "max":
            merged = torch.stack(outputs, dim=0).amax(dim=0)
        else:
            merged = torch.stack(outputs, dim=0).mean(dim=0)
        return self.merge(merged)


class EvidenceEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        evidence_mode: str = "logits_plus_image",
        feature_contract: str = "auto",
        pyramid_merge: str = "mean",
    ):
        super().__init__()
        self.evidence_mode = str(evidence_mode)
        self.logits_proj = ConvNormAct(num_classes, hidden_dim, kernel_size=1)
        self.image_proj = LazyConvNormAct(hidden_dim, kernel_size=3)
        self.feature_adapter = FeatureAdapterContract(hidden_dim, feature_contract=feature_contract, pyramid_merge=pyramid_merge)
        self.merge = LazyConvNormAct(hidden_dim, kernel_size=1)

    def forward(
        self,
        base_logits: torch.Tensor,
        input_ref: Optional[torch.Tensor] = None,
        backbone_features: TensorLike = None,
    ) -> torch.Tensor:
        parts = [self.logits_proj(base_logits)]
        h, w = base_logits.shape[-2:]

        use_image = self.evidence_mode in {"logits_plus_image", "logits_plus_feature_image"}
        if use_image and input_ref is not None:
            image_feat = input_ref
            if image_feat.shape[-2:] != (h, w):
                image_feat = F.interpolate(image_feat, size=(h, w), mode="bilinear", align_corners=False)
            parts.append(self.image_proj(image_feat))

        use_feature = self.evidence_mode in {"logits_plus_feature", "logits_plus_feature_image"}
        if use_feature:
            feature_tensor = self.feature_adapter(backbone_features=backbone_features, target_hw=(h, w))
            if feature_tensor is not None:
                parts.append(feature_tensor)

        return self.merge(torch.cat(parts, dim=1))


class RecurrentStateInitializer(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int):
        super().__init__()
        self.init = nn.Sequential(
            ConvNormAct(num_classes + hidden_dim, hidden_dim, kernel_size=3),
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
        )

    def forward(self, base_logits: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        return self.init(torch.cat([base_logits, evidence], dim=1))


class LoopIndexEmbedding2D(nn.Module):
    def __init__(self, step_dim: int):
        super().__init__()
        self.step_dim = max(1, int(step_dim))
        self.project = nn.Sequential(
            nn.Linear(2, self.step_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.step_dim, self.step_dim),
        )

    def forward(self, ref: torch.Tensor, step_id: int, max_steps: int) -> torch.Tensor:
        b, _, h, w = ref.shape
        step_ratio = float(step_id) / float(max(1, max_steps))
        progress = ref.new_full((b, 1), step_ratio)
        periodic = ref.new_full((b, 1), math.sin(step_ratio * math.pi))
        embed = self.project(torch.cat([progress, periodic], dim=1))
        return embed.view(b, self.step_dim, 1, 1).expand(-1, -1, h, w)


class StableEvidenceInjection(nn.Module):
    """
    Stable recurrent update inspired by the evidence reinjection pattern in OpenMythos.
    The module keeps state/evidence/core on explicit branches so the recurrent update
    remains interpretable and easy to audit.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_norm = nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim)
        self.evidence_norm = nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim)
        self.core_norm = nn.GroupNorm(_norm_groups(hidden_dim), hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.state_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.evidence_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.core_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)

    @staticmethod
    def _scale_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(x)

    def forward(
        self,
        state: torch.Tensor,
        evidence: torch.Tensor,
        core_update: torch.Tensor,
        step_id: int,
        max_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled_state = F.adaptive_avg_pool2d(self.state_norm(state), output_size=1).flatten(1)
        pooled_evidence = F.adaptive_avg_pool2d(self.evidence_norm(evidence), output_size=1).flatten(1)
        step_ratio = state.new_full((state.shape[0], 1), float(step_id) / float(max(1, max_steps)))
        periodic = state.new_full((state.shape[0], 1), math.sin(float(step_id) / float(max(1, max_steps)) * math.pi))
        stats = torch.cat([pooled_state, pooled_evidence, step_ratio, periodic], dim=1)
        state_logit, evidence_logit, core_logit = self.gate_mlp(stats).unbind(dim=1)

        state_gate = self._scale_range(state_logit, 0.55, 0.995).view(-1, 1, 1, 1)
        evidence_gate = self._scale_range(evidence_logit, 0.02, 0.35).view(-1, 1, 1, 1)
        core_gate = self._scale_range(core_logit, 0.05, 0.80).view(-1, 1, 1, 1)

        next_state = (
            state_gate * self.state_proj(self.state_norm(state))
            + evidence_gate * self.evidence_proj(self.evidence_norm(evidence))
            + core_gate * self.core_proj(self.core_norm(core_update))
        )
        aux = {
            "state_gate": state_gate.squeeze(-1).squeeze(-1).squeeze(-1),
            "evidence_gate": evidence_gate.squeeze(-1).squeeze(-1).squeeze(-1),
            "core_gate": core_gate.squeeze(-1).squeeze(-1).squeeze(-1),
        }
        return next_state, aux


class RecurrentGlobalCore(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pre = ConvNormAct(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.mix = ConvNormAct(hidden_dim, hidden_dim, kernel_size=1)
        self.stable_injection = StableEvidenceInjection(hidden_dim)
        self.last_injection_aux: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        state: torch.Tensor,
        evidence: torch.Tensor,
        step_id: int,
        max_steps: int = 11,
    ) -> torch.Tensor:
        stacked = torch.cat([state, evidence], dim=1)
        x = self.pre(stacked)
        x = self.dw(x)
        core_update = self.mix(x)
        next_state, aux = self.stable_injection(
            state=state,
            evidence=evidence,
            core_update=core_update,
            step_id=step_id,
            max_steps=max_steps,
        )
        self.last_injection_aux = aux
        return next_state


class GlobalProposalHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(state)


class LocalResidualAdapter(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(num_classes + hidden_dim + 1, hidden_dim, kernel_size=3),
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1, bias=True),
        )

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1:2, ...]
        return torch.sigmoid(logits)

    @staticmethod
    def _boundary(prob: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
        return torch.abs(prob - smooth)

    def forward(self, logits_t: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        fg = self._fg_prob(logits_t)
        boundary = self._boundary(fg)
        return self.block(torch.cat([logits_t, evidence, boundary], dim=1))


class _ConfidenceBlend(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        fusion_mode: str = "confidence_scalar",
        alpha_range: Tuple[float, float] = (0.2, 0.8),
        beta_range: Tuple[float, float] = (0.0, 0.5),
        max_steps: int = 11,
    ):
        super().__init__()
        self.fusion_mode = str(fusion_mode)
        self.alpha_range = tuple(alpha_range)
        self.beta_range = tuple(beta_range)
        self.max_steps = max(1, int(max_steps))
        self.mlp = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, 2))
        self.step_alpha = nn.Parameter(torch.zeros(self.max_steps))
        self.step_beta = nn.Parameter(torch.full((self.max_steps,), -0.5))

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1:2, ...]
        return torch.sigmoid(logits)

    def _scale_range(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(x)

    def forward(
        self,
        logits_t: torch.Tensor,
        global_delta: torch.Tensor,
        local_delta: torch.Tensor,
        state: torch.Tensor,
        step_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "fixed_scalar":
            alpha = logits_t.new_full((logits_t.shape[0],), 0.6)
            beta = logits_t.new_full((logits_t.shape[0],), 0.2 if local_delta.abs().sum().item() > 0 else 0.0)
            return alpha, beta

        if self.fusion_mode == "step_scalar":
            idx = min(max(0, int(step_id) - 1), self.max_steps - 1)
            alpha = self._scale_range(self.step_alpha[idx].expand(logits_t.shape[0]), *self.alpha_range)
            beta = self._scale_range(self.step_beta[idx].expand(logits_t.shape[0]), *self.beta_range)
            return alpha, beta

        fg = self._fg_prob(logits_t)
        uncertainty = (fg * (1.0 - fg)).mean(dim=(1, 2, 3))
        stats = torch.cat(
            [
                F.adaptive_avg_pool2d(state, output_size=1).flatten(1),
                uncertainty.unsqueeze(1),
                global_delta.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                local_delta.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                logits_t.new_full((logits_t.shape[0], 1), float(step_id) / float(self.max_steps)),
            ],
            dim=1,
        )
        alpha_logit, beta_logit = self.mlp(stats).unbind(dim=1)
        alpha = self._scale_range(alpha_logit, *self.alpha_range)
        beta = self._scale_range(beta_logit, *self.beta_range)
        return alpha, beta


class GainAwareACTHalt(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        halt_mode: str = "gain_aware_act",
        threshold: float = 0.99,
        patience: int = 2,
        state_eps: float = 1e-3,
        uncertainty_eps: float = 1e-3,
    ):
        super().__init__()
        self.halt_mode = str(halt_mode)
        self.threshold = float(threshold)
        self.patience = max(1, int(patience))
        self.state_eps = float(state_eps)
        self.uncertainty_eps = float(uncertainty_eps)
        self.halt_head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.SiLU(inplace=True), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        state: torch.Tensor,
        logits_prev: torch.Tensor,
        logits_next: torch.Tensor,
        state_delta: torch.Tensor,
        uncertainty: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        step_id: int,
        max_steps: int,
    ) -> torch.Tensor:
        _ = logits_next
        if self.halt_mode == "double_evidence_patience":
            cond = ((state_delta < self.state_eps) & (uncertainty_drop < self.uncertainty_eps)).float()
            return cond
        pooled = F.adaptive_avg_pool2d(state, output_size=1).flatten(1)
        stats = torch.cat(
            [
                pooled,
                state_delta.unsqueeze(1),
                uncertainty.unsqueeze(1),
                uncertainty_drop.unsqueeze(1),
                logits_prev.new_full((logits_prev.shape[0], 1), float(step_id) / float(max_steps)),
            ],
            dim=1,
        )
        return torch.sigmoid(self.halt_head(stats)).squeeze(1)


class ACTWeightedReadout(nn.Module):
    """
    Aggregate multi-step predictions with ACT-style ponder weights.
    This keeps the final output aligned with compute allocation, not only hard stop.
    """

    def __init__(self, threshold: float = 0.99):
        super().__init__()
        self.threshold = float(threshold)

    def forward(
        self,
        logits_seq: Sequence[torch.Tensor],
        state_seq: Sequence[torch.Tensor],
        halt_prob_seq: Sequence[torch.Tensor],
        active_mask_seq: Sequence[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
        if len(logits_seq) == 0:
            raise ValueError("ACTWeightedReadout expects at least one recursive step.")

        device = logits_seq[0].device
        batch = logits_seq[0].shape[0]
        remaining = torch.ones(batch, device=device)
        agg_logits = torch.zeros_like(logits_seq[0])
        agg_state = torch.zeros_like(state_seq[0])
        ponder_weights = []

        for logits_t, state_t, halt_t, active_t in zip(logits_seq, state_seq, halt_prob_seq, active_mask_seq):
            active = active_t.float()
            raw_weight = halt_t.clamp(min=0.0, max=1.0) * active
            weight = torch.minimum(remaining, raw_weight)
            agg_logits = agg_logits + weight.view(-1, 1, 1, 1) * logits_t
            agg_state = agg_state + weight.view(-1, 1, 1, 1) * state_t
            remaining = remaining - weight
            ponder_weights.append(weight)

        if float(remaining.max().detach().item()) > 0.0:
            agg_logits = agg_logits + remaining.view(-1, 1, 1, 1) * logits_seq[-1]
            agg_state = agg_state + remaining.view(-1, 1, 1, 1) * state_seq[-1]
            ponder_weights[-1] = ponder_weights[-1] + remaining
            remaining = torch.zeros_like(remaining)

        ponder_mass = torch.stack(ponder_weights, dim=0).sum(dim=0) if len(ponder_weights) > 0 else torch.zeros(batch, device=device)
        return {
            "logits": agg_logits,
            "state": agg_state,
            "ponder_weights": ponder_weights,
            "ponder_mass": ponder_mass,
        }


class InternalRecursiveController(nn.Module):
    """
    Portable logits-level recursive corrector with a shared recurrent core.

    Design goals:
    - single-file, torch-only implementation
    - pluggable after any segmentation backbone that emits logits
    - recurrent hidden state + repeated evidence injection
    - lightweight fusion and halt policies
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 64,
        max_steps: int = 5,
        min_steps: int = 2,
        evidence_mode: str = "logits_plus_image",
        feature_contract: str = "auto",
        pyramid_merge: str = "mean",
        enable_global_branch: bool = True,
        enable_local_branch: bool = False,
        fusion_mode: str = "confidence_scalar",
        halt_mode: str = "gain_aware_act",
        halt_threshold: float = 0.99,
        halt_patience: int = 2,
        readout_mode: str = "act_weighted",
        budget_profile: str = "lightweight_v1",
        cue_mode: str = "single_scale_compressed",
        state_stride: int = 8,
        transition_groups: int = 4,
        transition_cheap_core: str = "grouped_hybrid",
        micro_state_dim: int = 4,
        micro_state_rank: int = 2,
        micro_global_mode: str = "scalar_lowrank",
        micro_global_groups: int = 2,
        micro_global_spatial_rank: int = 1,
        micro_edge_stats_enhanced: bool = False,
        micro_local_confidence: bool = False,
        micro_local_op: str = "edge_restore",
        micro_state_gain_enhanced: bool = False,
        micro_state_gain_decomposition: bool = False,
        micro_fusion_mode: str = "step_scalar",
        micro_halt_stabilized: bool = False,
        micro_halt_gain_ema: bool = False,
        micro_halt_budget_aware: bool = False,
        micro_halt_momentum: float = 0.7,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.max_steps = max(1, int(max_steps))
        self.min_steps = max(1, int(min_steps))
        self.evidence_mode = str(evidence_mode)
        self.feature_contract = str(feature_contract)
        self.pyramid_merge = str(pyramid_merge)
        self.enable_global_branch = bool(enable_global_branch)
        self.enable_local_branch = bool(enable_local_branch)
        self.fusion_mode = str(fusion_mode)
        self.halt_mode = str(halt_mode)
        self.halt_threshold = float(halt_threshold)
        self.halt_patience = max(1, int(halt_patience))
        self.readout_mode = str(readout_mode)
        self.budget_profile = str(budget_profile).lower()
        self.cue_mode = str(cue_mode)
        self.state_stride = max(2, int(state_stride))
        self.transition_groups = max(1, int(transition_groups))
        self.transition_cheap_core = str(transition_cheap_core)
        self.micro_state_dim = max(2, int(micro_state_dim))
        self.micro_state_rank = max(1, int(micro_state_rank))
        self.micro_global_mode = str(micro_global_mode)
        self.micro_global_groups = max(1, int(micro_global_groups))
        self.micro_global_spatial_rank = max(1, int(micro_global_spatial_rank))
        self.micro_edge_stats_enhanced = bool(micro_edge_stats_enhanced)
        self.micro_local_confidence = bool(micro_local_confidence)
        self.micro_local_op = str(micro_local_op)
        self.micro_state_gain_enhanced = bool(micro_state_gain_enhanced)
        self.micro_state_gain_decomposition = bool(micro_state_gain_decomposition)
        self.micro_fusion_mode = str(micro_fusion_mode)
        self.micro_halt_stabilized = bool(micro_halt_stabilized)
        self.micro_halt_gain_ema = bool(micro_halt_gain_ema)
        self.micro_halt_budget_aware = bool(micro_halt_budget_aware)
        self.micro_halt_momentum = float(micro_halt_momentum)
        profile = self.budget_profile
        if profile == "lightweight_v1":
            profile = "micro_state_v1"
        self.budget_profile = profile
        if self.budget_profile != "micro_state_v1":
            raise ValueError("Unsupported controller profile.")
        self.is_micro_state = self.budget_profile == "micro_state_v1"
        self.is_legacy_lightweight = self.budget_profile == "legacy_lightweight"

        self.evidence_encoder = None
        self.state_initializer = None
        self.global_core = None
        self.global_head = None
        self.local_adapter = None
        self.fusion = None
        self.halt = None
        self.lightweight_cue = None
        self.lightweight_state_initializer = None
        self.lightweight_transition = None
        self.lightweight_global_head = None
        self.lightweight_local_adapter = None
        self.lightweight_fusion = None
        self.lightweight_halt = None
        self.micro_state_update = None
        self.micro_global = None
        self.micro_local = None
        self.micro_fusion = None
        self.micro_halt = None
        self.readout = ACTWeightedReadout(threshold=self.halt_threshold)
        if self.is_micro_state:
            self.micro_state_update = MicroStateUpdate(
                micro_state_dim=self.micro_state_dim,
                rank=self.micro_state_rank,
                gain_enhanced=self.micro_state_gain_enhanced,
                gain_decomposition=self.micro_state_gain_decomposition,
            )
            self.micro_global = LowRankGlobalControl(
                micro_state_dim=self.micro_state_dim,
                rank=self.micro_state_rank,
                num_classes=self.num_classes,
                mode=self.micro_global_mode,
                groups=self.micro_global_groups,
                spatial_rank=self.micro_global_spatial_rank,
            )
            self.micro_local = EdgeLocalGate(
                micro_state_dim=self.micro_state_dim,
                enhanced_stats=self.micro_edge_stats_enhanced,
                confidence_controller=self.micro_local_confidence,
                local_op=self.micro_local_op,
            )
            self.micro_fusion = _ScalarBlendController(
                max_steps=self.max_steps,
                micro_state_dim=self.micro_state_dim,
                mode=self.micro_fusion_mode,
            )
            self.micro_halt = TinyHaltController(
                micro_state_dim=self.micro_state_dim,
                stabilized=self.micro_halt_stabilized,
                gain_ema_input=self.micro_halt_gain_ema,
                budget_aware=self.micro_halt_budget_aware,
            )
        elif self.is_legacy_lightweight:
            use_image_context = "image" in self.evidence_mode
            self.lightweight_cue = LightweightEvidenceCueEncoder(
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                cue_mode=self.cue_mode,
                state_stride=self.state_stride,
                use_image_context=use_image_context,
            )
            self.lightweight_state_initializer = CompressedStateInitializer(self.num_classes, self.hidden_dim)
            self.lightweight_transition = GroupedHybridTransitionCore(
                hidden_dim=self.hidden_dim,
                groups=self.transition_groups,
                cheap_core=self.transition_cheap_core,
            )
            self.lightweight_global_head = GlobalCorrectionHeadLite(self.hidden_dim, self.num_classes)
            self.lightweight_local_adapter = LocalResidualAdapterLite(self.num_classes, self.hidden_dim)
            self.lightweight_fusion = _BudgetController(
                hidden_dim=self.hidden_dim,
                fusion_mode=self.fusion_mode,
                max_steps=self.max_steps,
            )
            self.lightweight_halt = GainAwareLiteHalt(
                hidden_dim=self.hidden_dim,
                threshold=min(self.halt_threshold, 0.85),
                patience=self.halt_patience,
                halt_mode=self.halt_mode,
            )
        else:
            self.evidence_encoder = EvidenceEncoder(
                self.num_classes,
                self.hidden_dim,
                evidence_mode=self.evidence_mode,
                feature_contract=self.feature_contract,
                pyramid_merge=self.pyramid_merge,
            )
            self.state_initializer = RecurrentStateInitializer(self.num_classes, self.hidden_dim)
            self.global_core = RecurrentGlobalCore(self.hidden_dim)
            self.global_head = GlobalProposalHead(self.hidden_dim, self.num_classes)
            self.local_adapter = LocalResidualAdapter(self.num_classes, self.hidden_dim)
            self.fusion = _ConfidenceBlend(
                hidden_dim=self.hidden_dim,
                fusion_mode=self.fusion_mode,
                alpha_range=(0.2, 0.8),
                beta_range=(0.0, 0.45),
                max_steps=self.max_steps,
            )
            self.halt = GainAwareACTHalt(
                hidden_dim=self.hidden_dim,
                halt_mode=self.halt_mode,
                threshold=self.halt_threshold,
                patience=self.halt_patience,
            )
        self.corrector_param_m = _safe_param_count(self)

        self.last_pred_seq = []
        self.last_aux: Dict[str, float] = {}
        self.last_trace = []

    @staticmethod
    def _fg_prob(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)[:, 1:2, ...]
        return torch.sigmoid(logits)

    def _compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        fg = self._fg_prob(logits)
        return (fg * (1.0 - fg)).mean(dim=(1, 2, 3))

    def _compute_state_delta(self, prev_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return (next_state - prev_state).pow(2).mean(dim=(1, 2, 3)).sqrt()

    def _build_trace_item(
        self,
        step_id: int,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        state_delta: torch.Tensor,
        uncertainty: torch.Tensor,
        uncertainty_drop: torch.Tensor,
        halt_prob: torch.Tensor,
        halt_reason: str,
        executed_steps: int,
    ) -> Dict[str, float]:
        return {
            "step_id": int(step_id),
            "executed_steps": int(executed_steps),
            "alpha_mean": float(alpha.mean().detach().item()),
            "beta_mean": float(beta.mean().detach().item()),
            "alpha_beta_delta": float((alpha - beta).abs().mean().detach().item()),
            "state_delta": float(state_delta.mean().detach().item()),
            "delta_pred_norm": float(self.last_aux.get("delta_pred_norm", 0.0)),
            "uncertainty": float(uncertainty.mean().detach().item()),
            "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
            "halt_metric_value": float(halt_prob.mean().detach().item()),
            "halt_reason": str(halt_reason),
            "halt_metric": "halt_probability",
            "state_gate": float(self.last_aux.get("state_gate", 0.0)),
            "evidence_gate": float(self.last_aux.get("evidence_gate", 0.0)),
            "core_gate": float(self.last_aux.get("core_gate", 0.0)),
        }

    def _record_pred(self, logits: torch.Tensor, keep_full_seq: bool):
        if keep_full_seq:
            self.last_pred_seq.append(logits)
            return
        if len(self.last_pred_seq) <= 1:
            self.last_pred_seq.append(logits)
        else:
            self.last_pred_seq[-1] = logits

    def _micro_state_forward(
        self,
        base_logits: torch.Tensor,
        input_ref: Optional[torch.Tensor] = None,
        backbone_features: TensorLike = None,
    ) -> torch.Tensor:
        _ = input_ref
        _ = backbone_features
        keep_full_seq = False
        self.last_pred_seq = [base_logits]
        self.last_trace = []
        batch_size = base_logits.shape[0]
        logits = base_logits
        anchor = gather_anchor_stats(logits_t=logits, base_logits=base_logits)
        micro_state = base_logits.new_zeros((batch_size, self.micro_state_dim))
        cumulative_halt = base_logits.new_zeros(batch_size)
        running = torch.ones(batch_size, dtype=torch.bool, device=base_logits.device)
        stall_counter = torch.zeros(batch_size, dtype=torch.long, device=base_logits.device)
        ema_delta = anchor["delta_mean"]
        ema_drop = torch.zeros_like(anchor["delta_mean"])
        ema_gain = torch.zeros_like(anchor["delta_mean"])
        prev_budget_gain = torch.zeros_like(anchor["delta_mean"])
        self.last_aux = {
            "budget_profile_code": 2.0,
            "corrector_param_m": float(_safe_param_count(self)),
            "micro_state_dim": float(self.micro_state_dim),
            "micro_state_rank": float(self.micro_state_rank),
            "micro_global_mode_code": 2.0 if self.micro_global_mode in {"groupwise_spatial_lowrank", "groupwise_rank1", "groupwise_rank2"} else (1.0 if self.micro_global_mode == "groupwise_lowrank" else 0.0),
            "micro_global_spatial_rank": float(self.micro_global_spatial_rank),
            "micro_edge_stats_enhanced": 1.0 if self.micro_edge_stats_enhanced else 0.0,
            "micro_local_confidence": 1.0 if self.micro_local_confidence else 0.0,
            "micro_local_op_code": 1.0 if self.micro_local_op == "axial_edge_restore" else (2.0 if self.micro_local_op == "cheap_residual" else 0.0),
            "micro_state_gain_enhanced": 1.0 if self.micro_state_gain_enhanced else 0.0,
            "micro_state_gain_decomposition": 1.0 if self.micro_state_gain_decomposition else 0.0,
            "micro_fusion_mode_code": 1.0 if self.micro_fusion_mode == "state_aware" else 0.0,
            "micro_halt_stabilized": 1.0 if self.micro_halt_stabilized else 0.0,
            "micro_halt_gain_ema": 1.0 if self.micro_halt_gain_ema else 0.0,
            "micro_halt_budget_aware": 1.0 if self.micro_halt_budget_aware else 0.0,
            "local_enabled": 1.0 if self.enable_local_branch else 0.0,
            "global_enabled": 1.0 if self.enable_global_branch else 0.0,
        }
        if self.max_steps <= 1 or (not self.enable_global_branch and not self.enable_local_branch):
            return base_logits

        for step_id in range(1, self.max_steps + 1):
            if not bool(running.any().item()):
                break
            step_ratio = logits.new_full((batch_size,), float(step_id) / float(max(1, self.max_steps)))
            prev_drop = self.last_aux.get("_uncertainty_drop_tensor", torch.zeros_like(anchor["delta_mean"]))
            prev_gain_ratio = self.last_aux.get("_gain_ratio_tensor", torch.zeros_like(anchor["delta_mean"]))
            prev_gain_delta = self.last_aux.get("_gain_delta_tensor", torch.zeros_like(anchor["delta_mean"]))
            prev_budget_gain = self.last_aux.get("_budget_gain_tensor", torch.zeros_like(anchor["delta_mean"]))
            micro_next, micro_aux = self.micro_state_update(
                micro_state=micro_state,
                delta_mean=anchor["delta_mean"],
                uncertainty_mean=anchor["uncertainty_mean"],
                uncertainty_drop=torch.zeros_like(anchor["delta_mean"]) if step_id == 1 else prev_drop,
                step_ratio=step_ratio,
                gain_ratio=prev_gain_ratio,
                gain_ema=ema_gain,
                gain_delta=prev_gain_delta,
                budget_gain=prev_budget_gain,
            )
            state_delta = (micro_next - micro_state).pow(2).mean(dim=1).sqrt()
            global_delta, global_gain, global_aux = self.micro_global(
                logits_t=logits,
                base_logits=base_logits,
                micro_state=micro_next,
            )
            if not self.enable_global_branch:
                global_delta = torch.zeros_like(logits)
                global_gain = torch.zeros_like(global_gain)
                global_aux = {
                    "global_gain_center": torch.zeros_like(global_gain),
                    "spatial_base_mean": torch.zeros_like(global_gain),
                    "spatial_center_mean": torch.zeros_like(global_gain),
                    "spatial_code_std": torch.zeros_like(global_gain),
                }
            local_delta, local_gate, local_aux = self.micro_local(
                logits_t=logits,
                base_logits=base_logits,
                micro_state=micro_next,
                boundary_mean=anchor["boundary_mean"],
                boundary_var=anchor["boundary_var"],
                uncertainty_mean=anchor["uncertainty_mean"],
                delta_mean=anchor["delta_mean"],
                uncertainty_boundary_overlap=anchor["uncertainty_boundary_overlap"],
                delta_boundary_overlap=anchor["delta_boundary_overlap"],
                boundary_confidence=anchor["boundary_confidence"],
                uncertainty_boundary_ratio=anchor["uncertainty_boundary_ratio"],
                delta_boundary_ratio=anchor["delta_boundary_ratio"],
            )
            if not self.enable_local_branch:
                local_delta = torch.zeros_like(logits)
                local_gate = torch.zeros_like(local_gate)
                local_aux = {
                    "edge_mix": torch.zeros_like(local_gate),
                    "boundary_var": torch.zeros_like(local_gate),
                    "restore_gain": torch.zeros_like(local_gate),
                    "boundary_confidence": torch.zeros_like(local_gate),
                    "local_op_code": torch.zeros_like(local_gate),
                }
            alpha, beta, fusion_aux = self.micro_fusion(
                batch_size=batch_size,
                step_id=step_id,
                device=logits.device,
                micro_state=micro_next,
                step_ratio=step_ratio,
                gain_ratio=prev_gain_ratio,
                boundary_confidence=anchor["boundary_confidence"],
                global_gain=global_gain,
                local_gate=local_gate,
            )
            logits_next = logits + alpha.view(-1, 1, 1, 1) * global_delta + beta.view(-1, 1, 1, 1) * local_delta
            next_anchor = gather_anchor_stats(logits_t=logits_next, base_logits=base_logits)
            uncertainty_drop = (anchor["uncertainty_mean"] - next_anchor["uncertainty_mean"]).clamp_min(0.0)
            gain_ratio = uncertainty_drop / anchor["uncertainty_mean"].clamp_min(1e-6)
            gain_delta = gain_ratio - ema_gain
            remaining_budget = (1.0 - step_ratio).clamp_min(0.0)
            budget_value = remaining_budget * gain_ratio
            if self.micro_halt_stabilized:
                ema_delta = self.micro_halt_momentum * ema_delta + (1.0 - self.micro_halt_momentum) * next_anchor["delta_mean"]
                ema_drop = self.micro_halt_momentum * ema_drop + (1.0 - self.micro_halt_momentum) * uncertainty_drop
            if self.micro_halt_gain_ema or self.micro_state_gain_enhanced:
                ema_gain = self.micro_halt_momentum * ema_gain + (1.0 - self.micro_halt_momentum) * gain_ratio
            halt_prob = self.micro_halt(
                micro_state=micro_next,
                delta_mean=next_anchor["delta_mean"],
                uncertainty_mean=next_anchor["uncertainty_mean"],
                uncertainty_drop=uncertainty_drop,
                step_ratio=step_ratio,
                stabilized_delta=ema_delta if self.micro_halt_stabilized else None,
                stabilized_drop=ema_drop if self.micro_halt_stabilized else None,
                gain_ema=ema_gain if self.micro_halt_gain_ema else None,
                gain_ratio=gain_ratio if self.micro_halt_gain_ema else None,
                remaining_budget=remaining_budget if self.micro_halt_budget_aware else None,
                budget_value=budget_value if self.micro_halt_budget_aware else None,
            )
            halt_delta = ema_delta if self.micro_halt_stabilized else next_anchor["delta_mean"]
            halt_drop = ema_drop if self.micro_halt_stabilized else uncertainty_drop
            gain_small = (halt_delta < 1e-3) & (halt_drop < 1e-3)
            stall_counter = torch.where(gain_small, stall_counter + 1, torch.zeros_like(stall_counter))
            cumulative_halt = cumulative_halt + halt_prob * running.float()
            should_halt = (cumulative_halt >= self.halt_threshold) | (stall_counter >= self.halt_patience)
            should_halt = should_halt & running & (step_id >= self.min_steps)
            logits = torch.where(running.view(-1, 1, 1, 1), logits_next, logits)
            micro_state = torch.where(running.view(-1, 1), micro_next, micro_state)
            anchor = next_anchor
            self._record_pred(logits, keep_full_seq=keep_full_seq)
            self.last_aux = {
                "budget_profile_code": 2.0,
                "corrector_param_m": float(_safe_param_count(self)),
                "micro_state_dim": float(self.micro_state_dim),
                "micro_state_rank": float(self.micro_state_rank),
                "micro_global_mode_code": 2.0 if self.micro_global_mode in {"groupwise_spatial_lowrank", "groupwise_rank1", "groupwise_rank2"} else (1.0 if self.micro_global_mode == "groupwise_lowrank" else 0.0),
                "micro_global_spatial_rank": float(self.micro_global_spatial_rank),
                "micro_edge_stats_enhanced": 1.0 if self.micro_edge_stats_enhanced else 0.0,
                "micro_local_confidence": 1.0 if self.micro_local_confidence else 0.0,
                "micro_local_op_code": 1.0 if self.micro_local_op == "axial_edge_restore" else (2.0 if self.micro_local_op == "cheap_residual" else 0.0),
                "micro_state_gain_enhanced": 1.0 if self.micro_state_gain_enhanced else 0.0,
                "micro_state_gain_decomposition": 1.0 if self.micro_state_gain_decomposition else 0.0,
                "micro_fusion_mode_code": 1.0 if self.micro_fusion_mode == "state_aware" else 0.0,
                "micro_halt_stabilized": 1.0 if self.micro_halt_stabilized else 0.0,
                "micro_halt_gain_ema": 1.0 if self.micro_halt_gain_ema else 0.0,
                "micro_halt_budget_aware": 1.0 if self.micro_halt_budget_aware else 0.0,
                "state_delta": float(state_delta.mean().detach().item()),
                "delta_pred_norm": float(anchor["delta_mean"].mean().detach().item()),
                "uncertainty": float(anchor["uncertainty_mean"].mean().detach().item()),
                "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
                "gain_ratio": float(gain_ratio.mean().detach().item()),
                "gain_ema": float(ema_gain.mean().detach().item()),
                "gain_delta": float(gain_delta.mean().detach().item()),
                "budget_value": float(budget_value.mean().detach().item()),
                "ema_delta": float(ema_delta.mean().detach().item()),
                "ema_uncertainty_drop": float(ema_drop.mean().detach().item()),
                "active_steps": float(step_id),
                "halt_metric_value": float(halt_prob.mean().detach().item()),
                "fuse_alpha_eff": float(alpha.mean().detach().item()),
                "fuse_beta_eff": float(beta.mean().detach().item()),
                "fusion_policy_shift": float(fusion_aux["fusion_policy_shift"].mean().detach().item()),
                "fusion_mode_code": float(fusion_aux["fusion_mode_code"].mean().detach().item()),
                "global_gain": float(global_gain.mean().detach().item()),
                "global_gain_center": float(global_aux["global_gain_center"].mean().detach().item()),
                "global_spatial_base_mean": float(global_aux["spatial_base_mean"].mean().detach().item()),
                "global_spatial_center_mean": float(global_aux["spatial_center_mean"].mean().detach().item()),
                "global_spatial_code_std": float(global_aux["spatial_code_std"].mean().detach().item()),
                "local_gate": float(local_gate.mean().detach().item()),
                "local_edge_mix": float(local_aux["edge_mix"].mean().detach().item()),
                "local_boundary_var": float(local_aux["boundary_var"].mean().detach().item()),
                "local_restore_gain": float(local_aux["restore_gain"].mean().detach().item()),
                "local_boundary_confidence": float(local_aux["boundary_confidence"].mean().detach().item()),
                "local_op_code": float(local_aux["local_op_code"].mean().detach().item()),
                "keep_gate": float(micro_aux["keep_gate"].mean().detach().item()),
                "inject_gate": float(micro_aux["inject_gate"].mean().detach().item()),
                "state_gain_feature": float(micro_aux["gain_feature"].mean().detach().item()),
                "state_gain_delta": float(micro_aux["gain_delta"].mean().detach().item()),
                "state_budget_gain": float(micro_aux["budget_gain"].mean().detach().item()),
                "local_enabled": 1.0 if self.enable_local_branch else 0.0,
                "global_enabled": 1.0 if self.enable_global_branch else 0.0,
                "_uncertainty_drop_tensor": uncertainty_drop.detach(),
                "_gain_ratio_tensor": gain_ratio.detach(),
                "_gain_delta_tensor": gain_delta.detach(),
                "_budget_gain_tensor": budget_value.detach(),
            }
            self.last_trace.append(
                MicroStateTracePack.build(
                    step_id=step_id,
                    alpha=alpha,
                    beta=beta,
                    state_delta=state_delta,
                    delta_mean=anchor["delta_mean"],
                    uncertainty_mean=anchor["uncertainty_mean"],
                    uncertainty_drop=uncertainty_drop,
                    halt_prob=halt_prob,
                    global_gain=global_gain,
                    local_gate=local_gate,
                    gain_ratio=gain_ratio,
                    gain_ema=ema_gain,
                )
            )
            running = running & (~should_halt)

        self.last_aux.pop("_uncertainty_drop_tensor", None)
        self.last_aux.pop("_gain_ratio_tensor", None)
        self.last_aux.pop("_gain_delta_tensor", None)
        self.last_aux.pop("_budget_gain_tensor", None)
        self.last_aux["ponder_mass"] = 1.0 if len(self.last_pred_seq) > 1 else 0.0
        return logits

    def _legacy_lightweight_forward(
        self,
        base_logits: torch.Tensor,
        input_ref: Optional[torch.Tensor] = None,
        backbone_features: TensorLike = None,
    ) -> torch.Tensor:
        keep_full_seq = False
        self.last_pred_seq = [base_logits]
        self.last_aux = {
            "budget_profile_code": 1.0,
            "corrector_param_m": float(_safe_param_count(self)),
            "state_stride": float(self.state_stride),
            "local_enabled": 1.0 if self.enable_local_branch else 0.0,
            "global_enabled": 1.0 if self.enable_global_branch else 0.0,
        }
        self.last_trace = []
        if self.max_steps <= 1 or (not self.enable_global_branch and not self.enable_local_branch):
            return base_logits

        cue = self.lightweight_cue(base_logits, input_ref=input_ref, backbone_features=backbone_features)
        cue_hw = cue.shape[-2:]
        cue_up = F.interpolate(cue, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        base_small = F.adaptive_avg_pool2d(base_logits, output_size=cue_hw)
        state = self.lightweight_state_initializer(base_small, cue)
        logits = base_logits
        cumulative_halt = logits.new_zeros(logits.shape[0])
        stall_counter = logits.new_zeros(logits.shape[0], dtype=torch.long)
        running = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)

        use_act_readout = self.readout_mode == "act_weighted"
        step_logits_seq = [] if use_act_readout else None
        step_state_seq = [] if use_act_readout else None
        step_halt_seq = [] if use_act_readout else None
        step_active_mask_seq = [] if use_act_readout else None

        for step_id in range(1, self.max_steps + 1):
            if not bool(running.any().item()):
                break

            active_mask = running.clone()
            next_state, transition_aux = self.lightweight_transition(
                state=state,
                cue=cue,
                step_id=step_id,
                max_steps=self.max_steps,
            )
            state_delta = self._compute_state_delta(state, next_state)

            global_delta = (
                self.lightweight_global_head(next_state, target_hw=base_logits.shape[-2:])
                if self.enable_global_branch
                else torch.zeros_like(logits)
            )
            local_delta = (
                self.lightweight_local_adapter(logits, cue_up)
                if self.enable_local_branch
                else torch.zeros_like(logits)
            )
            alpha, beta = self.lightweight_fusion(logits, global_delta, local_delta, next_state, step_id=step_id)
            merged = alpha.view(-1, 1, 1, 1) * global_delta + beta.view(-1, 1, 1, 1) * local_delta
            logits_next = logits + merged

            uncertainty_prev = self._compute_uncertainty(logits)
            uncertainty_next = self._compute_uncertainty(logits_next)
            uncertainty_drop = (uncertainty_prev - uncertainty_next).clamp_min(0.0)
            delta_pred = (logits_next - logits).pow(2).mean(dim=(1, 2, 3)).sqrt()
            halt_prob = self.lightweight_halt(
                state=next_state,
                logits_prev=logits,
                logits_next=logits_next,
                state_delta=state_delta,
                uncertainty=uncertainty_next,
                uncertainty_drop=uncertainty_drop,
                step_id=step_id,
                max_steps=self.max_steps,
            )

            gain_small = (state_delta < 1e-3) & (uncertainty_drop < 1e-3)
            stall_counter = torch.where(gain_small, stall_counter + 1, torch.zeros_like(stall_counter))
            cumulative_halt = cumulative_halt + halt_prob * running.float()
            should_halt = (cumulative_halt >= self.lightweight_halt.threshold) | (stall_counter >= self.halt_patience)
            should_halt = should_halt & running & (step_id >= self.min_steps)

            logits = torch.where(running.view(-1, 1, 1, 1), logits_next, logits)
            state = torch.where(running.view(-1, 1, 1, 1), next_state, state)

            if use_act_readout:
                step_logits_seq.append(logits)
                step_state_seq.append(state)
                step_halt_seq.append(halt_prob)
                step_active_mask_seq.append(active_mask)

            executed_steps = step_id
            self._record_pred(logits, keep_full_seq=keep_full_seq)
            self.last_aux = {
                "budget_profile_code": 1.0,
                "corrector_param_m": float(_safe_param_count(self)),
                "state_stride": float(self.state_stride),
                "state_hw_tokens": float(state.shape[-2] * state.shape[-1]),
                "state_delta": float(state_delta.mean().detach().item()),
                "uncertainty": float(uncertainty_next.mean().detach().item()),
                "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
                "delta_pred_norm": float(delta_pred.mean().detach().item()),
                "fuse_alpha_eff": float(alpha.mean().detach().item()),
                "fuse_beta_eff": float(beta.mean().detach().item()),
                "active_steps": float(executed_steps),
                "halt_metric_value": float(halt_prob.mean().detach().item()),
                "halt_reason_code": float(should_halt.float().mean().detach().item()),
                "keep_gate": float(transition_aux.get("keep_gate", alpha).mean().detach().item()),
                "inject_gate": float(transition_aux.get("inject_gate", beta).mean().detach().item()),
                "cue_mode_code": 0.0 if self.cue_mode == "none" else (2.0 if "pyramid" in self.cue_mode else 1.0),
                "local_enabled": 1.0 if self.enable_local_branch else 0.0,
                "global_enabled": 1.0 if self.enable_global_branch else 0.0,
            }
            self.last_trace.append(
                LightweightRDTTracePack.build(
                    step_id=step_id,
                    alpha=alpha,
                    beta=beta,
                    state_delta=state_delta,
                    uncertainty=uncertainty_next,
                    uncertainty_drop=uncertainty_drop,
                    halt_prob=halt_prob,
                    halt_reason="gain_aware_lite_halt",
                    executed_steps=executed_steps,
                    keep_gate=transition_aux.get("keep_gate"),
                    inject_gate=transition_aux.get("inject_gate"),
                )
            )
            running = running & (~should_halt)

        if use_act_readout and len(step_logits_seq) > 0:
            readout = self.readout(
                logits_seq=step_logits_seq,
                state_seq=step_state_seq,
                halt_prob_seq=step_halt_seq,
                active_mask_seq=step_active_mask_seq,
            )
            logits = readout["logits"]
            self.last_aux["ponder_mass"] = float(readout["ponder_mass"].mean().detach().item())
        else:
            self.last_aux["ponder_mass"] = 1.0 if len(self.last_pred_seq) > 1 else 0.0
        return logits

    def forward(
        self,
        base_logits: torch.Tensor,
        input_ref: Optional[torch.Tensor] = None,
        backbone_features: TensorLike = None,
    ) -> torch.Tensor:
        if self.is_micro_state:
            return self._micro_state_forward(
                base_logits=base_logits,
                input_ref=input_ref,
                backbone_features=backbone_features,
            )
        if self.is_legacy_lightweight:
            return self._legacy_lightweight_forward(
                base_logits=base_logits,
                input_ref=input_ref,
                backbone_features=backbone_features,
            )
        self.last_pred_seq = [base_logits]
        self.last_aux = {
            "budget_profile_code": 0.0,
            "corrector_param_m": float(_safe_param_count(self)),
            "state_stride": 1.0,
            "local_enabled": 1.0 if self.enable_local_branch else 0.0,
            "global_enabled": 1.0 if self.enable_global_branch else 0.0,
        }
        self.last_trace = []
        if self.max_steps <= 1 or (not self.enable_global_branch and not self.enable_local_branch):
            return base_logits

        evidence = self.evidence_encoder(base_logits, input_ref=input_ref, backbone_features=backbone_features)
        state = self.state_initializer(base_logits, evidence)
        logits = base_logits
        cumulative_halt = logits.new_zeros(logits.shape[0])
        stall_counter = logits.new_zeros(logits.shape[0], dtype=torch.long)
        running = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        use_act_readout = self.readout_mode == "act_weighted"
        step_logits_seq = [] if use_act_readout else None
        step_state_seq = [] if use_act_readout else None
        step_halt_seq = [] if use_act_readout else None
        step_active_mask_seq = [] if use_act_readout else None

        for step_id in range(1, self.max_steps + 1):
            if not bool(running.any().item()):
                break

            active_mask = running.clone()
            next_state = self.global_core(state, evidence, step_id=step_id, max_steps=self.max_steps)
            state_delta = self._compute_state_delta(state, next_state)

            global_delta = self.global_head(next_state) if self.enable_global_branch else torch.zeros_like(logits)
            local_delta = self.local_adapter(logits, evidence) if self.enable_local_branch else torch.zeros_like(logits)
            alpha, beta = self.fusion(logits, global_delta, local_delta, next_state, step_id=step_id)
            merged = alpha.view(-1, 1, 1, 1) * global_delta + beta.view(-1, 1, 1, 1) * local_delta
            logits_next = logits + merged

            uncertainty_prev = self._compute_uncertainty(logits)
            uncertainty_next = self._compute_uncertainty(logits_next)
            uncertainty_drop = (uncertainty_prev - uncertainty_next).clamp_min(0.0)
            delta_pred = (logits_next - logits).pow(2).mean(dim=(1, 2, 3)).sqrt()
            halt_prob = self.halt(
                state=next_state,
                logits_prev=logits,
                logits_next=logits_next,
                state_delta=state_delta,
                uncertainty=uncertainty_next,
                uncertainty_drop=uncertainty_drop,
                step_id=step_id,
                max_steps=self.max_steps,
            )

            gain_small = (state_delta < 1e-3) & (uncertainty_drop < 1e-3)
            stall_counter = torch.where(gain_small, stall_counter + 1, torch.zeros_like(stall_counter))
            cumulative_halt = cumulative_halt + halt_prob * running.float()
            should_halt = (cumulative_halt >= self.halt_threshold) | (stall_counter >= self.halt_patience)
            should_halt = should_halt & running & (step_id >= self.min_steps)

            logits = torch.where(running.view(-1, 1, 1, 1), logits_next, logits)
            state = torch.where(running.view(-1, 1, 1, 1), next_state, state)
            if use_act_readout:
                step_logits_seq.append(logits)
                step_state_seq.append(state)
                step_halt_seq.append(halt_prob)
                step_active_mask_seq.append(active_mask)

            executed_steps = step_id
            self.last_pred_seq.append(logits)
            self.last_aux = {
                "budget_profile_code": 0.0,
                "corrector_param_m": float(_safe_param_count(self)),
                "state_stride": 1.0,
                "delta_pred_norm": float(delta_pred.mean().detach().item()),
                "state_delta": float(state_delta.mean().detach().item()),
                "uncertainty": float(uncertainty_next.mean().detach().item()),
                "uncertainty_drop": float(uncertainty_drop.mean().detach().item()),
                "state_gate": float(self.global_core.last_injection_aux.get("state_gate", alpha).mean().detach().item()),
                "evidence_gate": float(self.global_core.last_injection_aux.get("evidence_gate", beta).mean().detach().item()),
                "core_gate": float(self.global_core.last_injection_aux.get("core_gate", halt_prob).mean().detach().item()),
                "fuse_alpha_eff": float(alpha.mean().detach().item()),
                "fuse_beta_eff": float(beta.mean().detach().item()),
                "alpha_beta_delta": float((alpha - beta).abs().mean().detach().item()),
                "executed_steps": float(executed_steps),
                "halt_metric_value": float(halt_prob.mean().detach().item()),
                "halt_reason_code": float(should_halt.float().mean().detach().item()),
                "local_enabled": 1.0 if self.enable_local_branch else 0.0,
                "global_enabled": 1.0 if self.enable_global_branch else 0.0,
            }
            self.last_trace.append(
                self._build_trace_item(
                    step_id=step_id,
                    alpha=alpha,
                    beta=beta,
                    state_delta=state_delta,
                    uncertainty=uncertainty_next,
                    uncertainty_drop=uncertainty_drop,
                    halt_prob=halt_prob,
                    halt_reason="gain_aware_act" if self.halt_mode != "double_evidence_patience" else "double_evidence_patience",
                    executed_steps=executed_steps,
                )
            )

            running = running & (~should_halt)

        if use_act_readout and len(step_logits_seq) > 0:
            readout = self.readout(
                logits_seq=step_logits_seq,
                state_seq=step_state_seq,
                halt_prob_seq=step_halt_seq,
                active_mask_seq=step_active_mask_seq,
            )
            logits = readout["logits"]
            state = readout["state"]
            self.last_aux["ponder_mass"] = float(readout["ponder_mass"].mean().detach().item())
            for item, ponder_weight in zip(self.last_trace, readout["ponder_weights"]):
                item["ponder_weight"] = float(ponder_weight.mean().detach().item())
        else:
            self.last_aux["ponder_mass"] = 1.0 if len(self.last_pred_seq) > 1 else 0.0

        return logits


class ModelWithController(nn.Module):
    def __init__(self, backbone: nn.Module, corrector: InternalRecursiveController):
        super().__init__()
        self.backbone = backbone
        self.corrector = corrector
        self.last_aux = {}
        self.last_trace = []
        self.last_pred_seq = []

    @staticmethod
    def _split_backbone_output(output):
        if isinstance(output, torch.Tensor):
            return output, None
        if isinstance(output, dict):
            logits = output.get("logits")
            features = output.get("features")
            if logits is None:
                logits = _first_tensor(output)
            return logits, features
        if isinstance(output, (list, tuple)):
            if len(output) == 2 and isinstance(output[0], torch.Tensor):
                return output[0], output[1]
            logits = _first_tensor(output)
            return logits, output
        raise TypeError(f"Unsupported backbone output type: {type(output)!r}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        logits, features = self._split_backbone_output(output)
        if logits is None:
            raise RuntimeError("Backbone output does not contain logits tensor.")
        refined = self.corrector(logits, input_ref=x, backbone_features=features)
        self.last_aux = dict(self.corrector.last_aux)
        self.last_trace = list(self.corrector.last_trace)
        self.last_pred_seq = list(self.corrector.last_pred_seq)
        return refined
