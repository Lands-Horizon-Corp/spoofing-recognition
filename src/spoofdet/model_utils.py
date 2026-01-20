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
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torchvision.transforms import v2
import copy
import gc
import time


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
):
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
