import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
from pathlib import Path
import torch.nn.functional as F

import os
from typing import Literal


from spoofdet.config import mean, std
from spoofdet.spoofing_metric import SpoofingMetric
from spoofdet.dataset import CelebASpoofDataset


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    val_transforms: v2.Compose | None = None,
    threshold: float = 0.5,
    final_activation: Literal["softmax", "sigmoid", "argmax"] = "argmax",
) -> tuple[plt.Figure, float, float, float, float, dict]:
    """
    Evaluates the model on the given dataloader and computes various metrics.
    args:
    - model: The trained model to evaluate.
    - dataloader: DataLoader for the evaluation dataset.
    - device: The device to run the evaluation on.
    - val_transforms: Transformations to apply to the validation data.
    - threshold: Threshold for classifying spoof probabilities (used for sigmoid/softmax).
    - final_activation: The final activation function used in the model ("softmax", "sigmoid", "argmax", or None).
    outputs:
    - fig: Confusion matrix figure.
    - acc_val: Accuracy value.
    - prec_val: Precision value.
    - rec_val: Recall value.
    - f1_val: F1 score value.
    - spoof_metric_val: Dictionary containing APCER, BPCER, and ACER values.
    """
    confmat = MulticlassConfusionMatrix(num_classes=2).to(device)
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    spoof_metric = SpoofingMetric().to(device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            images = val_transforms(images)
            outputs = model(images)
            if final_activation == "argmax":
                preds = torch.argmax(outputs, dim=1)
            elif final_activation == "sigmoid":
                probs = torch.sigmoid(outputs)
                preds = (probs[:, 1] > threshold).long()
            elif final_activation == "softmax":
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = (probs[:, 1] > threshold).long()

            # Update the metrics with this batch
            confmat.update(preds, labels)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
            spoof_metric.update(preds, labels)

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
    spoof_metric_val = spoof_metric.compute()

    # Plot the matrix
    fig, ax = confmat.plot(labels=["Live", "Spoof"])
    ax.set_title("Confusion Matrix: Live vs Spoof")

    # Add metrics as text below the matrix
    metrics_text = (
        f"Accuracy: {acc_val:.4f}   "
        f"Precision: {prec_val:.4f}   "
        f"Recall: {rec_val:.4f}   "
        f"F1 Score: {f1_val:.4f}   "
        f"APCER: {spoof_metric_val['APCER']:.4f}   "
        f"BPCER: {spoof_metric_val['BPCER']:.4f}   "
        f"ACER: {spoof_metric_val['ACER']:.4f}"
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
    print(
        f"Spoofing Metrics: APCER: {spoof_metric_val['APCER']:.4f}, BPCER: {spoof_metric_val['BPCER']:.4f}, ACER: {spoof_metric_val['ACER']:.4f}"
    )

    return fig, acc_val, prec_val, rec_val, f1_val, spoof_metric_val
