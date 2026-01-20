import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    MulticlassConfusionMatrix,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2
import copy
import gc
import time


from spoofdet.config import mean, std


def train_model(
    model,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    profiler_log_name: str,
    early_stopping_limit: int = 3,
    train_transforms: v2.Compose | None = None,
    val_transforms: v2.Compose | None = None,
) -> tuple[torch.nn.Module, dict[str, list]]:
    """
    Trains the given model using the provided data loaders, criterion, and optimizer.

    outputs:
    - model: The trained model with the best validation loss weights.
    - history: A dictionary containing training and validation loss and metrics history.
        precision, accuracy, recall, f1 score
    """

    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_accuracy": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    early_stopping_counter = 0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./log/{profiler_log_name}"
        ),
        with_stack=True,
    ) as prof:
        for epoch in range(epochs):

            model.train()
            train_loss = 0.0
            time_started = time.time()

            for images, labels in train_loader:
                with record_function("data_transfer"):
                    images, labels = images.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )
                with record_function("gpu_transforms"):
                    images = train_transforms(images)

                optimizer.zero_grad()
                with record_function("forward_pass"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                prof.step()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    with record_function("data_transfer_val"):
                        images, labels = images.to(device), labels.to(device)
                    with record_function("gpu_transforms_val"):
                        images = val_transforms(images)

                    with record_function("forward_pass_val"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    with record_function("loss_accumulation"):
                        val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)

                    with record_function("precision_calculation"):
                        precision.update(predicted, labels)
                        accuracy.update(predicted, labels)
                        recall.update(predicted, labels)
                        f1.update(predicted, labels)

            acc_val = accuracy.compute().item()
            prec_val = precision.compute().item()
            rec_val = recall.compute().item()
            f1_val = f1.compute().item()

            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)

            time_ended = time.time()
            epoch_duration = time_ended - time_started
            mins = int(epoch_duration // 60)
            secs = int(epoch_duration % 60)

            print(
                f"Epoch [{epoch+1}/{epochs}] | Time: {mins}m {secs}s Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Precision: {prec_val:.2f}% | Val Accuracy: {acc_val:.2f}% | Val Recall: {rec_val:.2f}% | Val F1: {f1_val:.2f}%"
            )

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_precision"].append(prec_val)
            history["val_accuracy"].append(acc_val)
            history["val_recall"].append(rec_val)
            history["val_f1"].append(f1_val)

            accuracy.reset()
            precision.reset()
            recall.reset()
            f1.reset()

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
                print("  -> New best model saved!")
            else:
                early_stopping_counter += 1
                print(
                    f"  -> No improvement. Counter: {early_stopping_counter}/{early_stopping_limit}"
                )

            if early_stopping_counter >= early_stopping_limit:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    val_transforms: v2.Compose | None = None,
):
    confmat = MulticlassConfusionMatrix(num_classes=2).to(device)
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = val_transforms(images)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Update the metrics with this batch
            confmat.update(preds, labels)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

    # Compute the final results
    final_matrix = confmat.compute()
    print("\nConfusion Matrix:")
    print(f"         Predicted Live | Predicted Spoof")
    print(f"Live        {final_matrix[0,0]:>6}     |     {final_matrix[0,1]:>6}")
    print(f"Spoof       {final_matrix[1,0]:>6}     |     {final_matrix[1,1]:>6}")
    acc_val = accuracy.compute()
    prec_val = precision.compute()
    rec_val = recall.compute()
    f1_val = f1.compute()

    # Plot the matrix
    fig, ax = confmat.plot(labels=["Live", "Spoof"])
    ax.set_title("Confusion Matrix: Live vs Spoof")

    # Add metrics as text below the matrix
    metrics_text = (
        f"Accuracy: {acc_val:.4f}   "
        f"Precision: {prec_val:.4f}   "
        f"Recall: {rec_val:.4f}   "
        f"F1 Score: {f1_val:.4f}"
    )

    # Position the text at the bottom center of the figure
    fig.text(
        0.5,
        -0.05,
        metrics_text,
        ha="center",
        fontsize=10,
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"
        ),
    )

    plt.show()

    print(f"Accuracy: {acc_val:.4f}")
    print(f"Precision: {prec_val:.4f}")
    print(f"Recall:    {rec_val:.4f}")
    print(f"F1 Score:  {f1_val:.4f}")

    return fig, acc_val, prec_val, rec_val, f1_val


gpu_transforms_train = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
    ]
).to(torch.device)

gpu_transforms_val = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
).to(torch.device)


def checkImage(dataset, idx):
    sample_img, sample_label = dataset[idx]
    display_img = sample_img.permute(1, 2, 0).numpy() / 255.0
    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'}")
    plt.axis("off")
    plt.show()


def checkAugmentedImage(dataset: Dataset, idx, gpu_transforms: v2.Compose):
    sample_img, sample_label = dataset[idx]

    # Apply GPU transforms (same as training)
    sample_img = sample_img.unsqueeze(0).to(torch.device)  # Add batch dim
    augmented = gpu_transforms(sample_img).squeeze(0).cpu()  # Remove batch dim

    # Denormalize from ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_img = augmented * std + mean
    display_img = torch.clamp(display_img, 0, 1)

    display_img = display_img.permute(1, 2, 0).numpy()

    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'} (Augmented)")
    plt.axis("off")
    plt.show()


def create_subset(
    dataset: Dataset, total_size: int = 1000, spoof_percent: float = 0.5
) -> Subset:
    """
    Creates a Subset  by looking at
    the internal label dictionary instead of loading images.
    """
    num_spoof = int(total_size * spoof_percent)
    num_live = total_size - num_spoof
    live_indices = []
    spoof_indices = []

    print(" Scanning internal label dict for class balance...")

    # Fast Loop: Access RAM only, no File I/O
    for idx, key in enumerate(dataset.image_keys):
        # Your specific schema: label is at index 43
        # 0 = Live, 1 = Spoof
        label = dataset.label_dict[key][43]

        if label == 0:
            live_indices.append(idx)
        else:
            spoof_indices.append(idx)

    print(f" Found: {len(live_indices)} Live | {len(spoof_indices)} Spoof")

    # Check if we have enough data
    if len(live_indices) < num_live or len(spoof_indices) < num_spoof:
        raise ValueError(
            f"Not enough data to create a balanced set of {total_size}. Reduce total_size."
        )

    # Random Sampling
    selected_live = np.random.choice(live_indices, num_live, replace=False)
    selected_spoof = np.random.choice(spoof_indices, num_spoof, replace=False)

    # Combine and Shuffle
    # We shuffle indices so the DataLoader doesn't get [500 Live] then [500 Spoof]
    final_indices = np.concatenate([selected_live, selected_spoof])
    np.random.shuffle(final_indices)

    return Subset(dataset, final_indices)


def checkDatasetDistribution(dataset: Subset):
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
    live_count = 0
    spoof_count = 0
    for _, labels in loader:
        live_in_batch = (labels == 0).sum().item()
        live_count += live_in_batch
        spoof_count += labels.size(0) - live_in_batch
    print(f"Live count: {live_count}, Spoof count: {spoof_count}")


def display_train_result(history: dict[str, list]) -> tuple[plt.Figure, plt.Figure]:
    """
    Displays training and validation loss and metrics history.
    Outputs two figures: one for loss and one for precision, accuracy, recall, and F1 score.
    """
    fig_loss, ax1 = plt.subplots()
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Value")
    ax1.legend()
    fig_loss.show()

    fig_precision, ax2 = plt.subplots()
    ax2.plot(history["val_precision"], label="Val Precision")
    ax2.plot(history["val_accuracy"], label="Val Accuracy")
    ax2.plot(history["val_recall"], label="Val Recall")
    ax2.plot(history["val_f1"], label="Val F1")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Value")
    ax2.legend()
    fig_precision.show()

    return fig_loss, fig_precision
