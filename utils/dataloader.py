from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _collect_image_files(folder: Path) -> list[Path]:
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return sorted(files)



def _find_mask_path(mask_dir: Path, stem: str) -> Path | None:
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        candidate = mask_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None



def _read_image(path: Path, read_rgb: bool) -> np.ndarray | None:
    if read_rgb:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)



def _resize_image_and_mask(image: np.ndarray, mask: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != target_size:
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask



def _normalize_tensor(image: np.ndarray, normalize_mode: str) -> torch.Tensor:
    if image.ndim == 2:
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)

    mode = str(normalize_mode or 'instance_norm').lower()
    if mode == 'imagenet' and tensor.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
    if mode in {'instance_norm', 'instance', 'per_image'}:
        dims = tuple(range(1, tensor.ndim))
        mean = tensor.mean(dim=dims, keepdim=True)
        std = tensor.std(dim=dims, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return (tensor - mean) / std
    return tensor



def _apply_train_aug(image: np.ndarray, mask: np.ndarray, aug_profile: str) -> tuple[np.ndarray, np.ndarray]:
    profile = str(aug_profile or 'superlight_default').lower()
    if profile in {'none', 'off'}:
        return image, mask
    if random.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    if random.random() < 0.3:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    return image, mask


class LightSegDataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        train: bool = False,
        normalize_mode: str = 'instance_norm',
        mask_threshold: float = 0.1,
        aug_profile: str = 'superlight_default',
        label_mode: str = 'binary',
        force_rgb_input: bool = True,
        **_: object,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.train = bool(train)
        self.normalize_mode = str(normalize_mode)
        self.mask_threshold = float(mask_threshold)
        self.aug_profile = str(aug_profile)
        self.label_mode = str(label_mode).lower()
        self.force_rgb_input = bool(force_rgb_input)

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int):
        image = np.asarray(self.x_data[idx]).copy()
        mask = np.asarray(self.y_data[idx]).copy()

        if self.train:
            image, mask = _apply_train_aug(image, mask, self.aug_profile)

        if image.ndim == 2 and self.force_rgb_input:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3 and not self.force_rgb_input:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image_tensor = _normalize_tensor(image, self.normalize_mode).float()

        if self.label_mode == 'binary':
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy((mask > self.mask_threshold).astype(np.uint8)).long()
        else:
            mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()
        return image_tensor, mask_tensor



def _resolve_split_root(root: Path, split_seed: int) -> Path:
    exact_seed_root = root / f'seed_{split_seed}'
    if exact_seed_root.is_dir():
        return exact_seed_root
    if (root / 'train' / 'images').is_dir() and (root / 'val' / 'images').is_dir():
        return root
    seed_roots = sorted([p for p in root.glob('seed_*') if p.is_dir()])
    if len(seed_roots) == 1:
        return seed_roots[0]
    return root



def _load_paired_split(split_root: Path, target_size: tuple[int, int], read_rgb: bool, prebinarize: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_dir = split_root / 'images'
    mask_dir = split_root / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        return images, masks

    for image_path in _collect_image_files(image_dir):
        mask_path = _find_mask_path(mask_dir, image_path.stem)
        if mask_path is None:
            continue
        image = _read_image(image_path, read_rgb=read_rgb)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image, mask = _resize_image_and_mask(image, mask, target_size)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if prebinarize:
            mask = (mask > 0).astype(np.uint8)
        images.append(image)
        masks.append(mask)
    return images, masks



def _split_arrays(images: list[np.ndarray], masks: list[np.ndarray], split_seed: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not images:
        raise RuntimeError('No paired samples were loaded from the dataset root.')
    indices = np.arange(len(images))
    rs = np.random.RandomState(int(split_seed))
    rs.shuffle(indices)
    val_count = max(1, int(len(indices) * float(val_ratio)))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx
    image_array = np.array(images)
    mask_array = np.array(masks)
    return image_array[train_idx], mask_array[train_idx], image_array[val_idx], mask_array[val_idx]



def get_dataloaders(config):
    data_config = config.get('data', config)
    root = Path(data_config.get('root', config.get('root', '.'))).resolve()
    batch_size = int(data_config.get('batch_size', config.get('batch_size', 4)))
    val_batch_size = int(data_config.get('val_batch_size', data_config.get('batch_size', batch_size)))
    num_workers = int(data_config.get('num_workers', config.get('num_workers', 0)))
    loader_seed = int(data_config.get('loader_seed', config.get('loader_seed', config.get('seed', 42))))
    split_seed = int(data_config.get('split_seed', config.get('split_seed', 42)))
    target_size = tuple(data_config.get('target_size', config.get('target_size', (352, 352))))
    mask_threshold = float(data_config.get('mask_threshold', config.get('mask_threshold', 0.1)))
    aug_profile = str(data_config.get('aug_profile', config.get('aug_profile', 'superlight_default')))
    normalize_mode = str(data_config.get('normalize_mode', config.get('normalize_mode', 'instance_norm')))
    label_mode = str(data_config.get('label_mode', config.get('label_mode', 'binary')))
    force_rgb_input = bool(data_config.get('force_rgb_input', config.get('force_rgb_input', True)))
    direct_raw_rgb_loading = bool(data_config.get('direct_raw_rgb_loading', config.get('direct_raw_rgb_loading', force_rgb_input)))
    light_mask_prebinarize = bool(data_config.get('light_mask_prebinarize', config.get('light_mask_prebinarize', True)))
    val_ratio = float(data_config.get('val_ratio', config.get('val_ratio', 0.1)))
    drop_last_train = bool(config.get('drop_last_train', data_config.get('drop_last_train', True)))

    def _seed_worker(worker_id: int):
        wseed = (loader_seed + worker_id) % (2**32)
        random.seed(wseed)
        np.random.seed(wseed)
        torch.manual_seed(wseed)

    data_gen = torch.Generator()
    data_gen.manual_seed(loader_seed)

    split_root = _resolve_split_root(root, split_seed)
    train_images, train_masks = _load_paired_split(split_root / 'train', target_size, direct_raw_rgb_loading, light_mask_prebinarize)
    val_images, val_masks = _load_paired_split(split_root / 'val', target_size, direct_raw_rgb_loading, light_mask_prebinarize)

    if train_images and val_images:
        x_train = np.array(train_images)
        y_train = np.array(train_masks)
        x_val = np.array(val_images)
        y_val = np.array(val_masks)
    else:
        all_images, all_masks = _load_paired_split(root, target_size, direct_raw_rgb_loading, light_mask_prebinarize)
        x_train, y_train, x_val, y_val = _split_arrays(all_images, all_masks, split_seed, val_ratio)

    train_ds = LightSegDataset(
        x_train,
        y_train,
        train=True,
        normalize_mode=normalize_mode,
        mask_threshold=mask_threshold,
        aug_profile=aug_profile,
        label_mode=label_mode,
        force_rgb_input=force_rgb_input,
    )
    val_ds = LightSegDataset(
        x_val,
        y_val,
        train=False,
        normalize_mode=normalize_mode,
        mask_threshold=mask_threshold,
        aug_profile='none',
        label_mode=label_mode,
        force_rgb_input=force_rgb_input,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=bool(drop_last_train and len(train_ds) >= batch_size),
        worker_init_fn=_seed_worker,
        generator=data_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=data_gen,
    )
    return {'train': train_loader, 'val': val_loader}



def postprocess_feasible_mask(
    mask: np.ndarray,
    min_area: int = 200,
    remove_border: bool = True,
    border_strip: int = 4,
    keep_global_largest_component: bool = False,
) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError('postprocess_feasible_mask expects a 2D binary mask.')

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    kept = np.zeros_like(binary, dtype=np.uint8)
    candidates: list[tuple[int, int]] = []
    h, w = binary.shape
    strip = max(0, int(border_strip))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        ww = int(stats[label_id, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        touches_border = x <= strip or y <= strip or (x + ww) >= (w - strip) or (y + hh) >= (h - strip)
        if remove_border and touches_border:
            continue
        candidates.append((label_id, area))

    if not candidates:
        return kept

    if keep_global_largest_component:
        label_id = max(candidates, key=lambda item: item[1])[0]
        kept[labels == label_id] = 1
        return kept

    for label_id, _ in candidates:
        kept[labels == label_id] = 1
    return kept
