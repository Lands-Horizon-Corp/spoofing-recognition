from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# Get predictions and labels on full test set
def calculate_roc_auc(model_quantized, small_test_loader):
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for img, label in small_test_loader:
            img = img.to(torch.device)

            outputs = model_quantized(img)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(label.numpy())

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    print(f"AUC-ROC Score: {roc_auc:.4f}")

    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random Classifier')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
                s=100, label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Quantized Model')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()
