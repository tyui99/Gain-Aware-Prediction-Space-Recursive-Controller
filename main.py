from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from models import get_model
from trainer import train_model
from utils.dataloader import get_dataloaders

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

PRESET_REGISTRY = {
    'paper_main': {
        'runtime': {'variant': 'primary', 'recursion': 'fixed_t5'},
        'trainer': {
            'epochs': 200, 'lr': 1e-3, 'weight_decay': 1e-2, 'optimizer': 'adamw', 'scheduler': 'cosine',
            't_max': 200, 'eta_min': 1e-6, 'loss': 'dice_focal_06_04', 'use_amp': True,
        },
    },
    'paper_variant_global': {
        'runtime': {'variant': 'secondary', 'recursion': 'adaptive'},
        'trainer': {
            'epochs': 200, 'lr': 1e-3, 'weight_decay': 1e-2, 'optimizer': 'adamw', 'scheduler': 'cosine',
            't_max': 200, 'eta_min': 1e-6, 'loss': 'dice_focal_06_04', 'use_amp': True,
        },
    },
}


def _add_bool_optional_flag(parser: argparse.ArgumentParser, name: str, default=None):
    if hasattr(argparse, 'BooleanOptionalAction'):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        return
    dest = str(name).lstrip('-').replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action='store_true')
    group.add_argument(f"--no-{str(name).lstrip('-')}", dest=dest, action='store_false')
    parser.set_defaults(**{dest: default})


def _expand_model_entry(runtime_cfg: dict) -> dict:
    from models import build_public_runtime_params

    variant = str(runtime_cfg.get('variant', 'primary'))
    return {
        'name': 'refinement_primary' if variant == 'primary' else 'refinement_secondary',
        'params': build_public_runtime_params(
            variant=variant,
            recursion=str(runtime_cfg.get('recursion', 'adaptive')),
        ),
    }


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_device_string(args) -> str:
    if args.force_cpu:
        return 'cpu'
    if args.device:
        return str(args.device)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _build_runtime_config(args) -> dict:
    preset = copy.deepcopy(PRESET_REGISTRY[args.preset])
    seed = int(args.seed)
    split_seed = int(args.split_seed if args.split_seed is not None else seed)
    image_size = int(args.image_size)
    output_root = Path(args.output_dir) if args.output_dir else Path('runs')
    run_name = str(args.run_name or f'{args.preset}_seed{seed}')
    log_dir = output_root / 'logs' / run_name
    ckpt_dir = output_root / 'checkpoints'

    data_cfg = {
        'root': str(Path(args.data_root)),
        'dataset': str(args.dataset).lower(),
        'batch_size': int(args.batch_size),
        'val_batch_size': int(args.val_batch_size or args.batch_size),
        'num_workers': int(args.num_workers),
        'split_seed': split_seed,
        'loader_seed': seed,
        'target_size': (image_size, image_size),
        'image_size': image_size,
        'direct_raw_rgb_loading': False,
        'force_rgb_input': False,
        'mask_threshold': float(args.mask_threshold),
        'aug_profile': str(args.aug_profile),
        'normalize_mode': str(args.normalize_mode),
        'light_mask_prebinarize': False,
        'label_mode': 'binary',
    }

    trainer_cfg = preset['trainer']
    trainer_cfg.update(
        {
            'epochs': int(args.epochs or trainer_cfg.get('epochs', 1)),
            'lr': float(args.lr or trainer_cfg.get('lr', 1e-3)),
            'weight_decay': float(args.weight_decay if args.weight_decay is not None else trainer_cfg.get('weight_decay', 1e-2)),
            'optimizer': str(args.optimizer or trainer_cfg.get('optimizer', 'adamw')),
            'scheduler': str(args.scheduler or trainer_cfg.get('scheduler', 'cosine')),
            'loss': str(args.loss or trainer_cfg.get('loss', 'dice_focal_06_04')),
            'device': _resolve_device_string(args),
            'run_name': run_name,
            'log_dir': str(log_dir),
            'best_path': str(ckpt_dir / f'{run_name}.pth'),
            'last_path': str(ckpt_dir / f'{run_name}.last.pth'),
            'disable_progress': bool(args.no_progress),
            'metric_threshold': float(args.metric_threshold),
            'checkpoint_metric': 'val_dice',
            'checkpoint_mode': 'max',
            'use_amp': trainer_cfg.get('use_amp', False) if args.use_amp is None else bool(args.use_amp),
            'seed': seed,
        }
    )

    resolved = {
        'preset': args.preset,
        'data': data_cfg,
        'model': _expand_model_entry(preset['runtime']),
        'trainer': trainer_cfg,
    }
    return resolved


def _instantiate_model(resolved_cfg: dict):
    model_cfg = resolved_cfg['model']
    return get_model(model_cfg['name'], **copy.deepcopy(model_cfg.get('params', {})))


def _write_resolved_run_spec(resolved_cfg: dict):
    log_dir = Path(resolved_cfg['trainer']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_path = log_dir / 'run_spec.json'
    spec_path.write_text(json.dumps(resolved_cfg, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='ICONIP recursive controller minimal training entry')
    parser.add_argument('--preset', required=True, choices=sorted(PRESET_REGISTRY.keys()))
    parser.add_argument('--data-root', required=True, help='Dataset root. Expected layout: train|val/{images,masks}.')
    parser.add_argument('--dataset', default='kvasirseg')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'runs'))
    parser.add_argument('--run-name')
    parser.add_argument('--device')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--optimizer')
    parser.add_argument('--scheduler')
    parser.add_argument('--loss')
    _add_bool_optional_flag(parser, '--use-amp', default=None)
    parser.add_argument('--metric-threshold', type=float, default=0.5)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--normalize-mode', default='instance_norm')
    parser.add_argument('--aug-profile', default='superlight_default')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    _set_global_seed(int(args.seed))
    resolved_cfg = _build_runtime_config(args)
    if args.dry_run:
        print(json.dumps(resolved_cfg, indent=2))
        return

    _write_resolved_run_spec(resolved_cfg)

    dataloaders = get_dataloaders(resolved_cfg['data'])
    model = _instantiate_model(resolved_cfg)
    train_model(model, dataloaders, resolved_cfg['trainer'], task='seg')


if __name__ == '__main__':
    main()
