from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.losses import calculate_hd_metrics, calculate_metrics, get_loss

logger = logging.getLogger(__name__)


def _move_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(v, device) for v in value)
    return value


def _extract_segfix_target_bundle(batch_aux, device):
    if not isinstance(batch_aux, dict):
        return None
    bundle = batch_aux.get('segfix_target_bundle')
    if isinstance(bundle, dict):
        return _move_value_to_device(bundle, device)
    return None


def _main_outputs(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def _model_accepts_arg(model, arg_name: str) -> bool:
    code = getattr(getattr(model, 'forward', None), '__code__', None)
    if code is None:
        return False
    return arg_name in code.co_varnames


def _forward_model(model, images, masks=None, batch_aux=None, use_amp=False):
    segfix_target_bundle = _extract_segfix_target_bundle(batch_aux, images.device)
    with autocast(enabled=use_amp):
        if masks is not None and _model_accepts_arg(model, 'segfix_target_bundle') and _model_accepts_arg(model, 'target'):
            return model(images, target=masks, segfix_target_bundle=segfix_target_bundle)
        if masks is not None and _model_accepts_arg(model, 'target'):
            return model(images, target=masks)
        return model(images)


def _compute_total_loss(model, criterion, outputs, masks, align_weight=1.0):
    main_outputs = _main_outputs(outputs)
    loss_inputs = outputs if criterion.__class__.__name__ == 'DeepSupervisionDiceFocal' else main_outputs
    total_loss = criterion(loss_inputs, masks)
    aux_value = None
    _ = align_weight

    return total_loss, main_outputs, aux_value


def _unpack_batch(batch):
    if len(batch) == 2:
        images, masks = batch
        batch_aux = None
    elif len(batch) == 3:
        images, masks, batch_aux = batch
    elif len(batch) == 4:
        _, _, images, masks = batch
        batch_aux = None
    else:
        raise ValueError(f'Unexpected batch format: {len(batch)}')
    if images.ndim == 3:
        images = images.unsqueeze(1)
    return images, masks, batch_aux


def validate(model, dataloader, criterion, device, metric_threshold=0.5, use_amp=False):
    model.eval()
    total_loss = 0.0
    inter_sum = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    total_pixels = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            images, masks, batch_aux = _unpack_batch(batch)
            images = images.to(device).float()
            masks = masks.to(device)

            outputs = _forward_model(model, images, masks=masks, batch_aux=batch_aux, use_amp=use_amp)
            loss, main_outputs, _ = _compute_total_loss(
                model,
                criterion,
                outputs,
                masks,
            )
            total_loss += float(loss.item())

            metrics = calculate_metrics(main_outputs, masks, threshold=float(metric_threshold))
            inter_sum += float(metrics.get('inter_sum', 0.0))
            pred_sum += float(metrics.get('pred_sum', 0.0))
            target_sum += float(metrics.get('target_sum', 0.0))
            total_pixels += float(main_outputs.shape[0] * main_outputs.shape[2] * main_outputs.shape[3])

    avg_loss = total_loss / max(1, len(dataloader))
    union = pred_sum + target_sum
    val_dice = -1.0 if union == 0.0 else float((2.0 * inter_sum) / (union + 1e-8))
    val_fg_ratio = float(pred_sum) / max(1.0, total_pixels)
    return avg_loss, val_dice, val_fg_ratio


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=None,
        log_dir='runs/logs/default',
        best_path=None,
        last_path=None,
        disable_progress=False,
        align_weight=1.0,
        metric_threshold=0.5,
        checkpoint_metric='val_dice',
        checkpoint_mode='max',
        use_amp=True,
        max_train_batches=0,
        max_val_batches=0,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.disable_progress = bool(disable_progress)
        self.align_weight = float(align_weight)
        self.metric_threshold = float(metric_threshold)
        self.checkpoint_metric = str(checkpoint_metric)
        self.checkpoint_mode = str(checkpoint_mode).lower()
        self.use_amp = bool(use_amp) and str(self.device).startswith('cuda') and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_metric = None
        self.best_epoch = -1
        self.max_train_batches = max(0, int(max_train_batches))
        self.max_val_batches = max(0, int(max_val_batches))
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = Path(best_path) if best_path else None
        self.last_path = Path(last_path) if last_path else None
        self.history: list[dict[str, float | int]] = []

    def _save_last_checkpoint(self, epoch, val_metrics):
        if self.last_path is None:
            return
        self.last_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': int(epoch),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
                'val_metrics': dict(val_metrics),
            },
            self.last_path,
        )

    def _maybe_save_best(self, epoch, val_metrics):
        metric_value = float(val_metrics.get(self.checkpoint_metric, val_metrics.get('val_dice', -1.0)))
        if self.best_metric is None:
            improved = True
        elif self.checkpoint_mode == 'min':
            improved = metric_value < float(self.best_metric)
        else:
            improved = metric_value > float(self.best_metric)
        if not improved:
            return
        self.best_metric = metric_value
        self.best_epoch = int(epoch)
        if self.best_path is not None:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.best_path)

    def _write_history(self):
        history_path = self.log_dir / 'history.json'
        history_path.write_text(json.dumps(self.history, indent=2), encoding='utf-8')

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        processed_batches = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Train', disable=self.disable_progress, leave=False)

        for batch in pbar:
            images, masks, batch_aux = _unpack_batch(batch)
            images = images.to(self.device).float()
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = _forward_model(self.model, images, masks=masks, batch_aux=batch_aux, use_amp=self.use_amp)
            loss, _, _ = _compute_total_loss(
                self.model,
                self.criterion,
                outputs,
                masks,
                align_weight=self.align_weight,
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())
            processed_batches += 1
            pbar.set_postfix(loss=f'{float(loss.item()):.4f}')

            if self.max_train_batches > 0 and processed_batches >= self.max_train_batches:
                break

        return running_loss / max(1, processed_batches)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        inter_sum = 0.0
        pred_sum = 0.0
        target_sum = 0.0
        total_pixels = 0.0
        hd95_sum = 0.0
        hd_sum = 0.0
        hd_cases = 0
        processed_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} Val', disable=self.disable_progress, leave=False)
            for batch in pbar:
                images, masks, batch_aux = _unpack_batch(batch)
                images = images.to(self.device).float()
                masks = masks.to(self.device)

                outputs = _forward_model(self.model, images, masks=masks, batch_aux=batch_aux, use_amp=self.use_amp)
                loss, main_outputs, _ = _compute_total_loss(
                    self.model,
                    self.criterion,
                    outputs,
                    masks,
                    align_weight=self.align_weight,
                )
                total_loss += float(loss.item())

                metrics = calculate_metrics(main_outputs, masks, threshold=self.metric_threshold)
                hd_metrics = calculate_hd_metrics(main_outputs, masks, threshold=self.metric_threshold)
                inter_sum += float(metrics.get('inter_sum', 0.0))
                pred_sum += float(metrics.get('pred_sum', 0.0))
                target_sum += float(metrics.get('target_sum', 0.0))
                total_pixels += float(main_outputs.shape[0] * main_outputs.shape[2] * main_outputs.shape[3])
                hd95_sum += float(hd_metrics.get('hd95_sum', 0.0))
                hd_sum += float(hd_metrics.get('hd_sum', 0.0))
                hd_cases += int(hd_metrics.get('sample_count', 0))
                processed_batches += 1

                pbar.set_postfix(loss=f'{float(loss.item()):.4f}', dice=f"{float(metrics.get('dice', 0.0)):.4f}")
                if self.max_val_batches > 0 and processed_batches >= self.max_val_batches:
                    break

        avg_loss = total_loss / max(1, processed_batches)
        union = pred_sum + target_sum
        val_dice = -1.0 if union == 0.0 else float((2.0 * inter_sum) / (union + 1e-8))
        val_fg_ratio = float(pred_sum) / max(1.0, total_pixels)
        val_hd95 = -1.0 if hd_cases <= 0 else float(hd95_sum / hd_cases)
        val_hd = -1.0 if hd_cases <= 0 else float(hd_sum / hd_cases)
        return {
            'val_loss': avg_loss,
            'val_dice': val_dice,
            'val_fg_ratio': val_fg_ratio,
            'val_hd95': val_hd95,
            'val_hd': val_hd,
        }

    def train(self, epochs):
        for epoch in range(int(epochs)):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = float(self.optimizer.param_groups[0]['lr'])

            record = {
                'epoch': int(epoch),
                'train_loss': float(train_loss),
                'val_loss': float(val_metrics['val_loss']),
                'val_dice': float(val_metrics['val_dice']),
                'val_hd95': float(val_metrics['val_hd95']),
                'lr': current_lr,
            }
            self.history.append(record)
            self._write_history()
            self._maybe_save_best(epoch, val_metrics)
            self._save_last_checkpoint(epoch, val_metrics)

            logger.info(
                'Epoch %d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | val_hd95=%.4f | lr=%.6f',
                epoch,
                record['train_loss'],
                record['val_loss'],
                record['val_dice'],
                record['val_hd95'],
                current_lr,
            )


def _build_optimizer(model, cfg):
    optimizer_name = str(cfg.get('optimizer', 'adamw')).lower()
    lr = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'sgd':
        momentum = float(cfg.get('momentum', 0.9))
        return torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == 'adam':
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, cfg):
    scheduler_name = str(cfg.get('scheduler', 'none')).lower()
    if scheduler_name == 'cosine':
        t_max = max(1, int(cfg.get('t_max', cfg.get('epochs', 1))))
        eta_min = float(cfg.get('eta_min', 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if scheduler_name == 'step':
        step_size = max(1, int(cfg.get('step_size', 20)))
        gamma = float(cfg.get('gamma', 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def train_model(model, dataloaders, config, task='seg'):
    if str(task).lower() != 'seg':
        raise RuntimeError('This open-source package supports segmentation only.')

    device = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device in {'cirrus', 'topcon', 'spectralis', 'generic'}:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = get_loss(str(config.get('loss', 'dice_focal_06_04')))
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=str(config.get('log_dir', 'runs/logs/default')),
        best_path=config.get('best_path'),
        last_path=config.get('last_path'),
        disable_progress=bool(config.get('disable_progress', False)),
        align_weight=float(config.get('align_weight', 1.0)),
        metric_threshold=float(config.get('metric_threshold', 0.5)),
        checkpoint_metric=str(config.get('checkpoint_metric', 'val_dice')),
        checkpoint_mode=str(config.get('checkpoint_mode', 'max')),
        use_amp=bool(config.get('use_amp', True)),
        max_train_batches=int(config.get('max_train_batches', 0)),
        max_val_batches=int(config.get('max_val_batches', 0)),
    )
    trainer.train(int(config.get('epochs', 1)))
    return trainer
