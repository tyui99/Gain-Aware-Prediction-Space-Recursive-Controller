from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import copy
import json
import random

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from main import PRESET_REGISTRY, _extract_state_dict
from models import get_model
from utils.dataloader import LightSegDataset, get_dataloaders
from utils.losses import calculate_hd_metrics, calculate_metrics, get_loss, prepare_binary_masks


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device('cpu')
    norm = str(device or 'cuda').lower()
    if norm in {'cuda', 'gpu', 'generic'} and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model_from_preset(preset: str) -> torch.nn.Module:
    preset_cfg = copy.deepcopy(PRESET_REGISTRY[preset])
    model_cfg = preset_cfg['model']
    return get_model(model_cfg['name'], **copy.deepcopy(model_cfg.get('params', {})))


def _collect_image_files(folder: Path) -> list[Path]:
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return sorted(files)


def _load_eval_only_arrays(data_root: Path, image_size: int, force_rgb: bool, prebinarize: bool) -> tuple[np.ndarray, np.ndarray]:
    target_size = (image_size, image_size)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    split_pairs = [
        (data_root / 'train' / 'images', data_root / 'train' / 'masks'),
        (data_root / 'val' / 'images', data_root / 'val' / 'masks'),
    ]
    flat_pair = (data_root / 'images', data_root / 'masks')
    if not any(img_dir.is_dir() and mask_dir.is_dir() for img_dir, mask_dir in split_pairs):
        split_pairs = [flat_pair]

    for img_dir, mask_dir in split_pairs:
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue
        for img_path in _collect_image_files(img_dir):
            mask_path = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                cand = mask_dir / f'{img_path.stem}{ext}'
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                continue
            if force_rgb:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            if mask.shape[:2] != target_size:
                mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if prebinarize:
                mask = (mask > 0).astype(np.uint8)
            samples.append((img, mask))
    if not samples:
        raise RuntimeError(f'No paired samples found under {data_root}')
    return np.array([img for img, _ in samples]), np.array([mask for _, mask in samples])


def build_validation_loader(args) -> DataLoader:
    data_root = Path(args.data_root).resolve()
    if args.eval_only:
        imgs, masks = _load_eval_only_arrays(data_root, int(args.image_size), bool(args.force_rgb_input), bool(args.light_mask_prebinarize))
        dataset = LightSegDataset(
            imgs,
            masks,
            train=False,
            normalize_mode=str(args.normalize_mode),
            mask_threshold=float(args.mask_threshold),
            aug_profile='none',
            label_mode='binary',
            force_rgb_input=bool(args.force_rgb_input),
        )
        return DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    data_cfg = {
        'root': str(data_root),
        'dataset': str(args.dataset).lower(),
        'batch_size': int(args.batch_size),
        'val_batch_size': int(args.batch_size),
        'num_workers': int(args.num_workers),
        'split_seed': int(args.split_seed),
        'loader_seed': int(args.seed),
        'target_size': (int(args.image_size), int(args.image_size)),
        'image_size': int(args.image_size),
        'direct_raw_rgb_loading': bool(args.force_rgb_input),
        'force_rgb_input': bool(args.force_rgb_input),
        'mask_threshold': float(args.mask_threshold),
        'aug_profile': 'superlight_default',
        'normalize_mode': str(args.normalize_mode),
        'light_mask_prebinarize': bool(args.light_mask_prebinarize),
    }
    return get_dataloaders(data_cfg)['val']


def _skeletonize_mask(mask_2d: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(mask_u8, dtype=np.uint8)
    work = mask_u8.copy()
    while True:
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skeleton


def _single_sample_cldice(pred_mask_2d: np.ndarray, target_mask_2d: np.ndarray) -> float:
    pred_u8 = (pred_mask_2d > 0).astype(np.uint8)
    target_u8 = (target_mask_2d > 0).astype(np.uint8)
    pred_has_fg = bool(pred_u8.any())
    target_has_fg = bool(target_u8.any())
    if not pred_has_fg and not target_has_fg:
        return 1.0
    if pred_has_fg != target_has_fg:
        return 0.0
    pred_skel = _skeletonize_mask(pred_u8)
    target_skel = _skeletonize_mask(target_u8)
    pred_skel_sum = float(pred_skel.sum())
    target_skel_sum = float(target_skel.sum())
    if pred_skel_sum <= 0.0 or target_skel_sum <= 0.0:
        return 0.0
    topology_precision = float(np.logical_and(pred_skel > 0, target_u8 > 0).sum()) / pred_skel_sum
    topology_sensitivity = float(np.logical_and(target_skel > 0, pred_u8 > 0).sum()) / target_skel_sum
    denom = topology_precision + topology_sensitivity
    if denom <= 0.0:
        return 0.0
    return float((2.0 * topology_precision * topology_sensitivity) / denom)


def calculate_extra_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float, fbeta_beta: float) -> dict[str, float]:
    pred_mask, target_proc = prepare_binary_masks(pred, target, threshold=threshold)
    p = (pred_mask == 1).float()
    t = (target_proc == 1).float()
    tp = float((p * t).sum().item())
    pred_sum = float(p.sum().item())
    target_sum = float(t.sum().item())
    precision = 1.0 if pred_sum <= 0.0 and target_sum <= 0.0 else (0.0 if pred_sum <= 0.0 else tp / max(pred_sum, 1e-8))
    recall = 1.0 if target_sum <= 0.0 and pred_sum <= 0.0 else (0.0 if target_sum <= 0.0 else tp / max(target_sum, 1e-8))
    beta_sq = float(fbeta_beta) ** 2
    denom = beta_sq * precision + recall
    fbeta = 0.0 if denom <= 0.0 else float(((1.0 + beta_sq) * precision * recall) / (denom + 1e-8))
    pred_np = pred_mask.detach().cpu().numpy()
    target_np = target_proc.detach().cpu().numpy()
    cldice_values = [_single_sample_cldice(sample_pred, sample_target) for sample_pred, sample_target in zip(pred_np, target_np)]
    cldice = float(np.mean(cldice_values)) if cldice_values else 0.0
    return {
        'precision': float(precision),
        'recall': float(recall),
        'fbeta': float(fbeta),
        'cldice': float(cldice),
    }


def evaluate(args) -> dict:
    set_global_seed(int(args.seed))
    device = resolve_device(args.device, force_cpu=bool(args.force_cpu))
    loader = build_validation_loader(args)
    model = build_model_from_preset(str(args.preset))
    state_dict = _extract_state_dict(torch.load(str(Path(args.checkpoint).resolve()), map_location='cpu'))
    load_result = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    criterion = get_loss(str(args.loss)) if str(args.loss) else None
    use_amp = bool(args.use_amp) and device.type == 'cuda'

    total_loss = 0.0
    inter_sum = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    total_pixels = 0.0
    hd95_sum = 0.0
    hd_sum = 0.0
    hd_cases = 0
    dice_sum = 0.0
    iou_sum = 0.0
    recall_sum = 0.0
    precision_sum = 0.0
    fbeta_sum = 0.0
    cldice_sum = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, masks = batch
            elif len(batch) == 3:
                images, masks, _ = batch
            else:
                raise ValueError(f'Unexpected validation batch format: {len(batch)}')
            if images.ndim == 3:
                images = images.unsqueeze(1)
            images = images.to(device).float()
            masks = masks.to(device)
            with autocast(enabled=use_amp):
                outputs = model(images)
                outputs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                loss = criterion(outputs, masks) if criterion is not None else None
            if loss is not None:
                total_loss += float(loss.item())
            metrics = calculate_metrics(outputs, masks, threshold=float(args.metric_threshold))
            hd_metrics = calculate_hd_metrics(outputs, masks, threshold=float(args.metric_threshold))
            extra = calculate_extra_metrics(outputs, masks, threshold=float(args.metric_threshold), fbeta_beta=float(args.fbeta_beta))
            inter_sum += float(metrics.get('inter_sum', 0.0))
            pred_sum += float(metrics.get('pred_sum', 0.0))
            target_sum += float(metrics.get('target_sum', 0.0))
            total_pixels += float(outputs.shape[0] * outputs.shape[2] * outputs.shape[3])
            dice_sum += float(metrics.get('dice', 0.0))
            iou_sum += float(metrics.get('iou', 0.0))
            recall_sum += float(metrics.get('recall', 0.0))
            precision_sum += float(extra.get('precision', 0.0))
            fbeta_sum += float(extra.get('fbeta', 0.0))
            cldice_sum += float(extra.get('cldice', 0.0))
            hd95_sum += float(hd_metrics.get('hd95_sum', 0.0))
            hd_sum += float(hd_metrics.get('hd_sum', 0.0))
            hd_cases += int(hd_metrics.get('sample_count', 0))
            batches += 1

    union = pred_sum + target_sum
    val_dice_global = -1.0 if union == 0.0 else float((2.0 * inter_sum) / (union + 1e-8))
    val_iou_global = -1.0 if val_dice_global < 0 else float(val_dice_global / (2.0 - val_dice_global + 1e-8))
    val_recall_global = -1.0 if target_sum <= 0.0 else float(inter_sum / (target_sum + 1e-8))

    return {
        'preset': str(args.preset),
        'checkpoint': str(Path(args.checkpoint)),
        'data_root': str(Path(args.data_root)),
        'dataset': str(args.dataset),
        'eval_only': bool(args.eval_only),
        'missing_keys': len(load_result.missing_keys),
        'unexpected_keys': len(load_result.unexpected_keys),
        'val_loss': float('nan') if criterion is None or batches == 0 else float(total_loss / max(1, batches)),
        'val_dice_global': float(val_dice_global),
        'val_dice_batch_mean': float(dice_sum / max(1, batches)),
        'val_iou_global': float(val_iou_global),
        'val_iou_batch_mean': float(iou_sum / max(1, batches)),
        'val_recall_global': float(val_recall_global),
        'val_recall_batch_mean': float(recall_sum / max(1, batches)),
        'val_precision_batch_mean': float(precision_sum / max(1, batches)),
        'val_fbeta_batch_mean': float(fbeta_sum / max(1, batches)),
        'val_cldice_batch_mean': float(cldice_sum / max(1, batches)),
        'val_fg_ratio': float(pred_sum / max(1.0, total_pixels)),
        'val_hd95': -1.0 if hd_cases <= 0 else float(hd95_sum / hd_cases),
        'val_hd': -1.0 if hd_cases <= 0 else float(hd_sum / hd_cases),
        'val_hd_case_count': int(hd_cases),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate one checkpoint')
    parser.add_argument('--preset', required=True, choices=sorted(PRESET_REGISTRY.keys()))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dataset', default='kvasirseg')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--loss', default='dice')
    parser.add_argument('--metric-threshold', type=float, default=0.1)
    parser.add_argument('--mask-threshold', type=float, default=0.1)
    parser.add_argument('--normalize-mode', default='instance_norm')
    parser.add_argument('--force-rgb-input', action='store_true')
    parser.add_argument('--light-mask-prebinarize', action='store_true')
    parser.add_argument('--fbeta-beta', type=float, default=0.3)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == '__main__':
    main()
