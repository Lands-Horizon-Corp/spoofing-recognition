import json
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
from pathlib import Path
import torch.nn.functional as F

from spoofdet.config import mean, std


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
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
    best_val_f1 = 0.0

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

            for batch_idx, (images, labels) in enumerate(train_loader):
                with record_function("data_transfer"):
                    images, labels = images.to(device, non_blocking=True), labels.to(
                        device, non_blocking=True
                    )
                with record_function("gpu_transforms"):
                    labels_onehot = F.one_hot(labels, num_classes=2).float()
                    images, labels_onehot = train_transforms(images, labels_onehot)

                optimizer.zero_grad()
                with record_function("forward_pass"):
                    outputs = model(images)
                    loss = criterion(outputs, labels_onehot)
                    loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                prof.step()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    with record_function("data_transfer_val"):
                        images, labels = images.to(device), labels.to(device)
                    with record_function("gpu_transforms_val"):
                        images = val_transforms(images)
                    with record_function("forward_pass_val"):
                        outputs = model(images)
                        labels_one_hot = F.one_hot(labels, num_classes=2).float()
                        loss = criterion(outputs, labels_one_hot)

                    with record_function("loss_accumulation"):
                        val_loss += loss.item() * images.size(0)
                        predicted = torch.argmax(outputs, dim=1)
                        # print("Sample Spoof Probabilities:", spoof_probs[:10])

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
            avg_val_f1 = max(best_val_f1, f1_val)

            time_ended = time.time()
            epoch_duration = time_ended - time_started
            mins = int(epoch_duration // 60)
            secs = int(epoch_duration % 60)

            print(
                f"Epoch [{epoch+1}/{epochs}] | Time: {mins}m {secs}s Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Precision: {prec_val * 100:.2f}% | Val Accuracy: {acc_val * 100:.2f}% | Val Recall: {rec_val * 100:.2f}% | Val F1: {f1_val * 100:.2f}%"
            )
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Scheduler Step! New LR: {current_lr:.8f}", end="")

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

            if avg_val_f1 > best_val_f1:
                best_val_f1 = avg_val_f1
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
        for batch_idx, (images, labels) in enumerate(dataloader):
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


def get_transform_pipeline(
    device: torch.device, target_size: int
) -> tuple[v2.Compose, v2.Compose]:
    """
    Returns training and validation transform pipelines moved to the specified device.
    """

    gpu_transforms_train = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=30),
            # v2.RandomPerspective(distortion_scale=0.4, p=0.2),
            v2.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Shift left/right/up/down
                scale=(0.8, 1.2),  # Zoom In AND Zoom Out (crucial!)
            ),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            v2.RandomGrayscale(p=0.1),
            v2.GaussianNoise(sigma=0.03),
            v2.ToDtype(torch.float32, scale=True),
            v2.MixUp(num_classes=2, alpha=0.2),
            # v2.RandomErasing(p=0.2),
        ]
    ).to(device)

    gpu_transforms_val = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    ).to(device)
    return gpu_transforms_train, gpu_transforms_val


def checkImage(dataset, idx):
    sample_img, sample_label = dataset[idx]
    display_img = sample_img.permute(1, 2, 0).numpy() / 255.0
    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'}")
    plt.axis("off")
    plt.show()


def checkAugmentedImage(dataset: Dataset, idx, gpu_transforms: v2.Compose):
    sample_img, sample_label = dataset[idx]
    viz_transforms = v2.Compose(
        [
            t
            for t in gpu_transforms.transforms
            if not isinstance(t, (v2.MixUp, v2.CutMix))
        ]
    )

    # Apply GPU transforms (same as training)
    sample_img = sample_img.unsqueeze(0).to(device)  # Add batch dim
    augmented = viz_transforms(sample_img).squeeze(0).cpu()  # Remove batch dim
    display_img = torch.clamp(augmented, 0, 1)

    display_img = display_img.permute(1, 2, 0).numpy()

    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'} (Augmented)")
    plt.axis("off")
    plt.show()


def create_subset(
    dataset_or_subset: Subset | Dataset,
    total_size: int = 1000,
    spoof_percent: float = 0.5,
) -> Subset:
    """
    Creates a Subset  by looking at
    the internal label dictionary instead of loading images.
    """

    if isinstance(dataset_or_subset, Subset):
        source_dataset = dataset_or_subset.dataset
        valid_indices = (
            dataset_or_subset.indices
        )  # The specific indices allowed for this split
    else:
        source_dataset = dataset_or_subset
        valid_indices = range(len(dataset_or_subset))
    num_spoof = int(total_size * spoof_percent)
    num_live = total_size - num_spoof
    live_indices_relative = []
    spoof_indices_relative = []

    print(" Scanning specific indices for class balance...")

    # Iterate ONLY over the valid indices for this subset
    # relative_idx: 0, 1, 2... (index in the new subset)
    # real_idx: 45, 102, 3... (index in the main dataset)
    for relative_idx, real_idx in enumerate(valid_indices):
        key = source_dataset.image_keys[real_idx]
        # 0 = Live, 1 = Spoof (Index 43 in your schema)
        label = source_dataset.label_dict[key][43]

        if label == 0:
            live_indices_relative.append(relative_idx)
        else:
            spoof_indices_relative.append(relative_idx)

    print(
        f" Found in this split: {len(live_indices_relative)} Live | {len(spoof_indices_relative)} Spoof"
    )

    # Check if we have enough data
    if len(live_indices_relative) < num_live or len(spoof_indices_relative) < num_spoof:
        raise ValueError(
            f"Not enough data in this split to create size {total_size}. "
            f"Available: {len(live_indices_relative)} Live, {len(spoof_indices_relative)} Spoof."
        )

    # Random Sampling from relative indices
    selected_live = np.random.choice(live_indices_relative, num_live, replace=False)
    selected_spoof = np.random.choice(spoof_indices_relative, num_spoof, replace=False)

    # Combine and Shuffle
    final_indices = np.concatenate([selected_live, selected_spoof])
    np.random.shuffle(final_indices)

    # Return a Subset OF THE SUBSET
    # This keeps the chain valid (train_ds -> balanced_train_ds)
    return Subset(dataset_or_subset, final_indices)


def checkDatasetDistribution(dataset: Subset):
    live_count = 0
    for img, label in dataset:
        if label.item() == 0:
            live_count += 1
    spoof_count = len(dataset) - live_count
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


def _create_save_new_path(save_path: Path, path_name: str, num: int) -> Path:
    new_dir = save_path / f"{path_name}_{num}"
    if new_dir.exists():
        return _create_save_new_path(save_path, path_name, num + 1)
    else:
        new_dir.mkdir(parents=True)
        return new_dir


def save_results(
    model: torch.nn.Module,
    confusion_matrix_fig: plt.Figure,
    train_loss_fig: plt.Figure,
    precision_fig: plt.Figure,
    params: json,
):

    save_path = Path("train_results")
    path_name = "train"
    num = 0

    save_path.mkdir(parents=True, exist_ok=True)

    # create new dir if already exist
    new_path = _create_save_new_path(save_path, path_name, num)

    print(f"Saving results to: {new_path}")

    confusion_matrix_fig.savefig(new_path / "confusion_matrix.png", bbox_inches="tight")
    train_loss_fig.savefig(new_path / "train_loss.png")
    precision_fig.savefig(new_path / "precision.png")

    torch.save(model.state_dict(), new_path / "model.pt")
    with open(new_path / "params.json", "w", encoding="utf-8") as f:
        f.write(params)


def analyze_spoof_types(
    model: torch.nn.Module,
    dataset: Subset,
    device: torch.device,
    val_transforms: v2.Compose,
):
    """
    Analyzes model performance across different spoof types.
    Spoof type is at index 40 in the label array.
    """
    spoof_type_labels = {
        0: "Live",
        1: "Photo",
        2: "Poster",
        3: "A4",
        4: "Face Mask",
        5: "Upper Body Mask",
        6: "Region Mask",
        8: "Pad",
        7: "PC",
        9: "Phone",
        10: "3D Mask",
    }

    model.eval()

    # Dictionary to store results per spoof type
    spoof_type_results = {}

    with torch.no_grad():

        for idx in range(len(dataset)):
            img, label = dataset[idx]

            if hasattr(dataset, "dataset"):  # If it's a Subset
                actual_idx = dataset.indices[idx]
                image_key = dataset.dataset.image_keys[actual_idx]
                full_labels = dataset.dataset.label_dict[image_key]
            else:
                image_key = dataset.image_keys[idx]
                full_labels = dataset.label_dict[image_key]

            # Get the original key to access full label info

            spoof_type = full_labels[40]  # Spoof type at index 40
            live_spoof_label = full_labels[43]  # Live/Spoof at index 43

            # Only analyze spoof images (label = 1)
            if live_spoof_label != 1:
                continue

            # Prepare image for model
            img = img.unsqueeze(0).to(device)
            img = val_transforms(img)

            # Get prediction
            output = model(img)
            pred = torch.argmax(output, dim=1).item()

            # Initialize spoof type entry if needed
            if spoof_type not in spoof_type_results:
                spoof_type_results[spoof_type] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                }

            # Update statistics
            spoof_type_results[spoof_type]["total"] += 1
            if pred == 1:  # Correctly identified as spoof
                spoof_type_results[spoof_type]["correct"] += 1
            else:  # Incorrectly identified as live
                spoof_type_results[spoof_type]["incorrect"] += 1

    # Calculate accuracy per spoof type
    results_df = []
    for spoof_type, stats in sorted(spoof_type_results.items()):
        accuracy = (
            (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        )
        results_df.append(
            {
                "Spoof Type": spoof_type_labels.get(spoof_type, "Unknown"),
                "Type ID": spoof_type,
                "Total": stats["total"],
                "Correct": stats["correct"],
                "Incorrect": stats["incorrect"],
                "Accuracy (%)": accuracy,
            }
        )

    results_df = pd.DataFrame(results_df)
    results_df = results_df.sort_values("Accuracy (%)")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of accuracy per spoof type
    ax1.barh(
        results_df["Spoof Type"].astype(str),
        results_df["Accuracy (%)"],
        color=[
            "red" if x < 50 else "orange" if x < 80 else "green"
            for x in results_df["Accuracy (%)"]
        ],
    )
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_ylabel("Spoof Type")
    ax1.set_title("Model Accuracy by Spoof Type")
    ax1.axvline(x=50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    # Stacked bar chart showing correct vs incorrect
    ax2.barh(
        results_df["Spoof Type"].astype(str),
        results_df["Correct"],
        label="Correct (Detected as Spoof)",
        color="green",
        alpha=0.7,
    )
    ax2.barh(
        results_df["Spoof Type"].astype(str),
        results_df["Incorrect"],
        left=results_df["Correct"],
        label="Incorrect (Detected as Live)",
        color="red",
        alpha=0.7,
    )
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Spoof Type")
    ax2.set_title("Correct vs Incorrect Predictions by Spoof Type")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nSpoof Type Analysis Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"\nWorst Performing Spoof Types (Accuracy < 70%):")
    worst = results_df[results_df["Accuracy (%)"] < 70]
    if len(worst) > 0:
        print(worst.to_string(index=False))
    else:
        print("None - All spoof types have >70% accuracy!")

    return results_df, fig


def analyze_dataset_spoof_distribution(
    dataset: torch.utils.data.Subset,  # Added type hint for clarity
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Analyzes the distribution of spoof types in the dataset.
    """
    spoof_type_labels = {
        0: "Live",  # This is the key we want to see in the plot
        1: "Photo",
        2: "Poster",
        3: "A4",
        4: "Face Mask",
        5: "Upper Body Mask",
        6: "Region Mask",
        7: "PC",
        8: "Pad",
        9: "Phone",
        10: "3D Mask",
    }

    spoof_type_counts = {}
    live_count = 0

    print("Analyzing spoof type distribution...")

    if hasattr(dataset, "dataset"):  # It is a Subset
        parent_dataset = dataset.dataset
        indices = dataset.indices
    else:  # It is the original dataset
        parent_dataset = dataset
        indices = range(len(dataset))
    current_dataset = dataset
    final_indices = list(range(len(dataset)))

    # unwraps subset inside subset inside subset...
    while isinstance(current_dataset, torch.utils.data.Subset):
        # Map the current indices to the parent's indices
        final_indices = [current_dataset.indices[i] for i in final_indices]
        current_dataset = current_dataset.dataset

    # Now current_dataset is the root (CelebASpoofDataset)
    parent_dataset = current_dataset
    for idx in indices:

        image_key = parent_dataset.image_keys[idx]
        full_labels = parent_dataset.label_dict[image_key]

        live_spoof_label = full_labels[43]  # Live/Spoof at index 43

        if live_spoof_label == 0:  # Live
            live_count += 1
        else:  # Spoof
            spoof_type = full_labels[40]
            spoof_type_counts[spoof_type] = spoof_type_counts.get(spoof_type, 0) + 1

    results = []

    if live_count > 0:
        results.append({"Spoof Type": "Live", "Type ID": 0, "Count": live_count})

    for spoof_type, count in sorted(spoof_type_counts.items()):
        results.append(
            {
                "Spoof Type": spoof_type_labels.get(spoof_type, "Unknown"),
                "Type ID": spoof_type,
                "Count": count,
            }
        )

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Warning: Dataset is empty.")
        return results_df, plt.figure()

    results_df["Percentage"] = (results_df["Count"] / len(dataset) * 100).round(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = [
        "green" if row["Spoof Type"] == "Live" else "red"
        for _, row in results_df.iterrows()
    ]
    ax1.barh(results_df["Spoof Type"], results_df["Count"], color=colors, alpha=0.7)
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Class Type")
    ax1.set_title("Class Distribution in Dataset")
    ax1.grid(axis="x", alpha=0.3)

    ax2.pie(
        results_df["Count"],
        labels=results_df["Spoof Type"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    ax2.set_title("Distribution (Percentage)")

    plt.tight_layout()
    plt.show()

    print("\nClass Distribution:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    return results_df, fig


def display_params(
    lr, weight_decay, batch_size, epochs, early_stopping_limit, target_size
):
    print("Training Configuration:")
    print(f" Batch Size: {batch_size}")
    print(f" Learning Rate: {lr}")
    print(f" Weight Decay: {weight_decay}")
    print(f" Epochs: {epochs}")
    print(f" Early Stopping Limit: {early_stopping_limit}")
    print(f" Target Size: {target_size}")

    return json.dumps(
        {
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "early_stopping_limit": early_stopping_limit,
            "target_size": target_size,
        },
        indent=4,
    )


import os


def check_subject_leakage(root_dir):
    """
    Scans root/train and root/test to ensure no Subject IDs overlap.
    Assumes structure: root / split / subject_id / ...
    """
    splits = ["train", "test"]
    subject_sets = {}

    # 1. Collect Subject IDs for each split
    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            print(f"Error: Could not find folder {split_path}")
            return

        # Get all folder names (subject_ids) in this split
        subjects = set(os.listdir(split_path))
        subject_sets[split] = subjects
        print(f"Found {len(subjects)} subjects in '{split}'")

    # 2. Check for Overlap
    # intersection() finds items present in BOTH sets
    overlap = subject_sets["train"].intersection(subject_sets["test"])

    if len(overlap) > 0:
        print("\n🚨 CRITICAL FAILURE: DATA LEAK DETECTED! 🚨")
        print(f"Found {len(overlap)} subjects that are in BOTH Train and Test.")
        print(f"Example Leaked IDs: {list(overlap)[:5]}")
        print(
            "ACTION: You must remove these subjects from one of the sets or re-split."
        )
    else:
        print("\n✅ SUCCESS: No subject leakage detected.")
        print("Train and Test sets are completely independent.")


# Usage:
# check_subject_leakage("path/to/your/dataset_root")
