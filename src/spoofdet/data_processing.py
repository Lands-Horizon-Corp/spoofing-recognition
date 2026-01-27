import random
import json
import os
import copy
import gc
import time
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
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

import torch.nn.functional as F


from typing import Dict, List, Tuple


from spoofdet.config import mean, std
from spoofdet.spoofing_metric import SpoofingMetric
from spoofdet.dataset import CelebASpoofDataset


def get_transform_pipeline(
    device: torch.device, target_size: int
) -> tuple[v2.Compose, v2.Compose]:
    """
    Returns training and validation transform pipelines moved to the specified device.
    """

    gpu_transforms_train = v2.Compose(
        [
            v2.Resize((target_size, target_size), antialias=True),
            # v2.RandomResizedCrop(
            #     size=(target_size, target_size),
            #     scale=(0.7, 1.0),  # Zoom range
            #     ratio=(0.75, 1.33),
            #     antialias=True,
            # ),
            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomRotation(degrees=30),
            # v2.RandomPerspective(distortion_scale=0.3, p=0.2),
            # v2.RandomAffine(
            #     degrees=0,
            #     translate=(0.1, 0.1),  # Shift left/right/up/down
            #     scale=(0.8, 1.2),  # Zoom In AND Zoom Out (crucial!)
            # ),
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0),
            # v2.Grayscale(num_output_channels=3),
            # v2.RandomGrayscale(p=0.1),
            # v2.GaussianBlur(kernel_size=3, sigma=(0.3, 2.0)),
            # v2.GaussianNoise(sigma=0.02),
            # v2.RandomErasing(p=0.2),
            v2.Normalize(mean=mean, std=std),
            # v2.RandomChoice(
            #     [
            #         v2.MixUp(num_classes=2, alpha=0.2),
            #         v2.CutMix(num_classes=2, alpha=1.0),
            #     ]
            # ),
        ]
    ).to(device)

    gpu_transforms_val = v2.Compose(
        [
            v2.Resize((target_size, target_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    ).to(device)
    return gpu_transforms_train, gpu_transforms_val


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
        label = int(source_dataset.label_dict[key][43])

        if label == 0:
            live_indices_relative.append(relative_idx)
        elif label == 1:
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


def read_json_data_path(json_path: str):
    """
    reading the CelebA-Spoof JSON
    """
    with open(json_path, "r") as f:
        celeba_data = json.load(f)
    if not isinstance(celeba_data, dict):
        raise ValueError("The JSON data is not in the expected dictionary format.")
    return celeba_data


def get_data_for_training(
    json_path: str,
    train_count: int,
    val_count: int,
    spoof_percent: float = 0.5,
    seed: int = 42,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Complete data processing with subject-disjoint splitting and label balancing.

    Returns:
    - train_dict: {image_path: label_array} for training
    - val_dict: {image_path: label_array} for validation
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # 1. Read JSON data
    celeba_data = read_json_data_path(json_path)

    # 2. Filter ONLY training data (not test data)
    train_paths_only = {
        path: labels
        for path, labels in celeba_data.items()
        if path.startswith("Data/train/")  # Only training set
    }

    print(f"Total training images: {len(train_paths_only)}")

    # 3. Split by subject (subject-disjoint)
    train_subject_paths, val_subject_paths = split_json_by_subject(train_paths_only)

    print(
        f"Subject-split - Train: {len(train_subject_paths)}, Val: {len(val_subject_paths)}"
    )

    # 4. Balance by labels using JSON labels (not folder names)
    train_balanced_paths = balance_by_labels(
        path_list=train_subject_paths,
        target_count=train_count,
        spoof_percent=spoof_percent,
        celeba_data=celeba_data,  # Pass the full data for label lookup
    )

    val_balanced_paths = balance_by_labels(
        path_list=val_subject_paths,
        target_count=val_count,
        spoof_percent=spoof_percent,
        celeba_data=celeba_data,
    )

    # 5. Create final dictionaries
    train_dict = {path: celeba_data[path] for path in train_balanced_paths}
    val_dict = {path: celeba_data[path] for path in val_balanced_paths}

    # 6. Statistics
    print_stats(train_dict, "Training")
    print_stats(val_dict, "Validation")

    return train_dict, val_dict


def balance_by_labels(
    path_list: List[str], target_count: int, spoof_percent: float, celeba_data: Dict
) -> List[str]:
    """
    Balance data by live/spoof labels using JSON labels (not folder names).
    """
    # Calculate required counts
    live_count = int(target_count * (1 - spoof_percent))
    spoof_count = target_count - live_count

    # Separate paths by actual JSON labels
    live_paths = []
    spoof_paths = []

    for path in path_list:
        # Get label from JSON (index 43 = live/spoof)
        label_array = celeba_data[path]
        if len(label_array) < 44:
            raise ValueError(f"Invalid label array for {path}: {label_array}")

        label = int(label_array[43])  # 0 = live, 1 = spoof

        if label == 0:
            live_paths.append(path)
        elif label == 1:
            spoof_paths.append(path)
        else:
            raise ValueError(f"Invalid label value {label} for {path}")

    # Check availability
    if len(live_paths) < live_count:
        raise ValueError(f"Insufficient live images: {len(live_paths)} < {live_count}")
    if len(spoof_paths) < spoof_count:
        raise ValueError(
            f"Insufficient spoof images: {len(spoof_paths)} < {spoof_count}"
        )

    # Random selection
    selected_live = np.random.choice(live_paths, live_count, replace=False)
    selected_spoof = np.random.choice(spoof_paths, spoof_count, replace=False)

    # Combine and shuffle
    selected_paths = np.concatenate([selected_live, selected_spoof])
    np.random.shuffle(selected_paths)

    return list(selected_paths)


def split_json_by_subject(
    celeba_data: Dict[str, List],
    val_split: float = 0.2,  # Changed from 0.5 - typical 80/20 split
) -> Tuple[List[str], List[str]]:
    """
    Create subject-disjoint splits for CelebA-Spoof.
    Only processes training data (paths starting with 'Data/train/').
    """
    # Group paths by subject ID
    subject_to_paths = {}

    for path in celeba_data.keys():
        parts = path.split("/")
        if len(parts) < 4:
            print(f"Warning: Unexpected path format: {path}")
            continue

        # Extract subject ID (e.g., "12345" from "Data/train/12345/live/001.jpg")
        subject_id = parts[2]  # Index 2 is subject ID

        if subject_id not in subject_to_paths:
            subject_to_paths[subject_id] = []
        subject_to_paths[subject_id].append(path)

    # Shuffle subjects
    subjects = list(subject_to_paths.keys())
    np.random.shuffle(subjects)

    # Split subjects (not images)
    split_idx = int(len(subjects) * (1 - val_split))
    train_subjects = subjects[:split_idx]
    val_subjects = subjects[split_idx:]

    # Collect all paths for each subject
    train_paths = []
    val_paths = []

    for subject in train_subjects:
        train_paths.extend(subject_to_paths[subject])

    for subject in val_subjects:
        val_paths.extend(subject_to_paths[subject])

    print(
        f"Subjects: {len(subjects)} total, {len(train_subjects)} train, {len(val_subjects)} val"
    )
    print(f"Images: {len(train_paths)} train, {len(val_paths)} val")

    return train_paths, val_paths


def print_stats(data_dict: Dict, name: str):
    """Print statistics about the dataset."""
    live_count = 0
    spoof_count = 0

    for path, labels in data_dict.items():
        label = int(labels[43])  # Live/spoof label
        if label == 0:
            live_count += 1
        else:
            spoof_count += 1

    total = live_count + spoof_count
    print(f"{name} set: {total} images")
    print(f"  Live: {live_count} ({live_count/total*100:.1f}%)")
    print(f"  Spoof: {spoof_count} ({spoof_count/total*100:.1f}%)")


if __name__ == "__main__":
    import spoofdet.config as config

    train_dict, val_dict = get_data_for_training(
        json_path=config.TRAIN_JSON, train_count=1000, val_count=200, spoof_percent=0.5
    )
    train_ds = CelebASpoofDataset(
        root_dir=config.ROOT_DIR,
        json_label_path=train_dict,
        bbox_json_path=config.BBOX_LOOKUP,
    )
