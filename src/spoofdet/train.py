from __future__ import annotations

import copy
import time
from collections.abc import Sized
from typing import cast

import torch
import torch.nn as nn
from spoofdet.data_processing import get_data_for_training
from spoofdet.data_processing import get_transform_pipeline
from spoofdet.dataset import CelebASpoofDataset
from spoofdet.efficient_net.model_utils import get_model
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall
from torchvision.transforms import v2


def train_model(
    model,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    profiler_log_name: str,
    train_transforms: v2.Compose,
    val_transforms: v2.Compose,
    early_stopping_limit: int = 3,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR |
    torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[torch.nn.Module, dict[str, list]]:
    """
    Trains the given model using the provided data loaders, criterion, and optimizer.

    outputs:
    - model: The trained model with the best validation loss weights.
    - history: A dictionary containing training and validation loss and metrics history.
        precision, accuracy, recall, f1 score
    """

    accuracy = Accuracy(task='binary').to(device)
    precision = Precision(task='binary').to(device)
    recall = Recall(task='binary').to(device)
    f1 = F1Score(task='binary').to(device)
    history: dict[str, list[float]] = {
        'train_loss': [],
        'val_loss': [],
        'val_precision': [],
        'val_accuracy': [],
        'val_recall': [],
        'val_f1': [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    early_stopping_counter = 0
    scaler = torch.GradScaler(str(device))

    has_mixup_cutmix_in_transforms = _has_mixup_cutmix(train_transforms)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/{profiler_log_name}",
        ),
        with_stack=True,
    ) as prof:
        for epoch in range(epochs):

            model.train()
            # for module in model.modules():
            #     if isinstance(module, torch.nn.BatchNorm2d):
            #         module.eval()
            train_loss = 0.0
            time_started = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                with record_function('data_transfer'):
                    images, labels = images.to(device, non_blocking=True), labels.to(
                        device,
                        non_blocking=True,
                    )
                with record_function('gpu_transforms'):

                    if has_mixup_cutmix_in_transforms:
                        images, labels = train_transforms(images, labels)
                        # print(" Applied MixUp/CutMix Transform")
                    else:
                        images = train_transforms(images)

                optimizer.zero_grad()
                with record_function('forward_pass'):

                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(images)
                loss = criterion(outputs.float(), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)
                prof.step()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    with record_function('data_transfer_val'):
                        images, labels = images.to(device), labels.to(device)
                    with record_function('gpu_transforms_val'):
                        images = val_transforms(images)
                    with record_function('forward_pass_val'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    with record_function('loss_accumulation'):
                        val_loss += loss.item() * images.size(0)
                        # print("Sample Spoof Probabilities:", spoof_probs[:10])

                    with record_function('precision_calculation'):
                        predicted = torch.argmax(outputs, dim=1)
                        precision.update(predicted, labels)
                        accuracy.update(predicted, labels)
                        recall.update(predicted, labels)
                        f1.update(predicted, labels)

            acc_val = accuracy.compute().item()
            prec_val = precision.compute().item()
            rec_val = recall.compute().item()
            f1_val = f1.compute().item()

            avg_train_loss = train_loss / \
                len(cast(Sized, train_loader.dataset))
            avg_val_loss = val_loss / len(cast(Sized, val_loader.dataset))

            time_ended = time.time()
            epoch_duration = time_ended - time_started
            mins = int(epoch_duration // 60)
            secs = int(epoch_duration % 60)

            print(
                f"Epoch [{epoch+1}/{epochs}] |"
                f"Time: {mins}m {secs}s "
                f"Train Loss: {avg_train_loss:.4f} |"
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Precision: {prec_val * 100:.2f}% | "
                f"Val Accuracy: {acc_val * 100:.2f}% | "
                f"Val Recall: {rec_val * 100:.2f}% | "
                f"Val F1: {f1_val * 100:.2f}%",
            )
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                print(f"Scheduler Step! New LR: {current_lr:.8f}", end='')

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_precision'].append(prec_val)
            history['val_accuracy'].append(acc_val)
            history['val_recall'].append(rec_val)
            history['val_f1'].append(f1_val)

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1.reset()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
                print('  -> New best model saved!')
            else:
                early_stopping_counter += 1
                print(
                    f"  -> No improvement. Counter: {
                        early_stopping_counter
                    }/{early_stopping_limit}",
                )

            if early_stopping_counter >= early_stopping_limit:
                print('Early stopping triggered.')
                break

    model.load_state_dict(best_model_wts)
    return model, history


def _has_mixup_cutmix(transform_pipeline):
    """
    Checks if MixUp or CutMix exists in a v2.Compose pipeline,
    even if nested inside RandomChoice.
    """
    if not isinstance(transform_pipeline, v2.Compose):
        return False

    # Define the types we are looking for
    target_types = (v2.MixUp, v2.CutMix)

    for t in transform_pipeline.transforms:
        # 1. Direct check: Is the transform itself MixUp/CutMix?
        if isinstance(t, target_types):
            return True

        # 2. Nested check: Is it a RandomChoice containing MixUp/CutMix?
        # (Common practice: v2.RandomChoice([v2.MixUp(...), v2.CutMix(...)])
        if isinstance(t, v2.RandomChoice):
            for sub_t in t.transforms:
                if isinstance(sub_t, target_types):
                    return True

    return False


if __name__ == '__main__':
    import spoofdet.config as config

    train_dict, val_dict = get_data_for_training(
        json_path=str(config.TRAIN_JSON),
        train_count=1000,
        val_count=200,
        spoof_percent=0.5,
    )
    train_ds = CelebASpoofDataset(
        root_dir=config.ROOT_DIR,
        json_label_path=train_dict,
        bbox_json_path=config.BBOX_LOOKUP,
        target_size=320,
        bbox_original_size=config.BBOX_ORGINAL_SIZE,
    )
    val_ds = CelebASpoofDataset(
        root_dir=config.ROOT_DIR,
        json_label_path=val_dict,
        bbox_json_path=config.BBOX_LOOKUP,
        target_size=320,
        bbox_original_size=config.BBOX_ORGINAL_SIZE,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_transforms_train, gpu_transforms_val = get_transform_pipeline(
        device=device,
        target_size=320,
    )

    model = get_model(with_weights=True)
    model = model.to(device)

    train_model(
        model=model,
        device=device,
        train_loader=DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=4,
        ),
        val_loader=DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=4,
        ),
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        epochs=10,
        profiler_log_name='test_profiler',
        early_stopping_limit=3,
        train_transforms=gpu_transforms_train,
        val_transforms=gpu_transforms_val,
    )
