from __future__ import annotations

import json
import random
from collections.abc import Sized
from typing import Any
from typing import cast

import numpy as np
from spoofdet.data_processing.dataset import CelebASpoofDataset
from torch.utils.data import Dataset
from torch.utils.data import Subset


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
        valid_indices = dataset_or_subset.indices
    else:
        source_dataset = dataset_or_subset
        valid_indices = range(len(cast(Sized, dataset_or_subset)))

    # Get labels efficiently
    if hasattr(source_dataset, 'targets'):
        all_labels = cast(CelebASpoofDataset,
                          source_dataset).targets   # list of ints
    elif hasattr(source_dataset, 'samples'):
        all_labels = [label for _, label in cast(
            CelebASpoofDataset, source_dataset).samples]
    else:
        # Fallback – but warn user
        print('Warning: No fast label access, falling back to slow __getitem__')
        all_labels = []
        for i in range(len(cast(Sized, source_dataset))):
            _, label = source_dataset[i]   # still slow, but only once?
            all_labels.append(label)

    # Now build indices using the pre‑computed labels
    live_indices_relative = []
    spoof_indices_relative = []
    for relative_idx, real_idx in enumerate(valid_indices):
        label = all_labels[real_idx]
        if label == 0:
            live_indices_relative.append(relative_idx)
        else:
            spoof_indices_relative.append(relative_idx)

    print(
        f" Found in this split: {len(live_indices_relative)}"
        f" Live | {len(spoof_indices_relative)} Spoof",
    )
    num_live = int(total_size * (1 - spoof_percent))
    num_spoof = total_size - num_live

    # Check if we have enough data
    if len(live_indices_relative) < num_live or len(spoof_indices_relative) < num_spoof:
        raise ValueError(
            f"Not enough data in this split to create size {total_size}. "
            f"Available: {len(live_indices_relative)} "
            f"Live, {len(spoof_indices_relative)} Spoof."
        )

    # Random Sampling from relative indices
    selected_live = np.random.choice(
        live_indices_relative,
        num_live,
        replace=False,
    )
    selected_spoof = np.random.choice(
        spoof_indices_relative,
        num_spoof,
        replace=False,
    )

    # Combine and Shuffle
    final_indices = np.concatenate([selected_live, selected_spoof])
    np.random.shuffle(final_indices)

    # Return a Subset OF THE SUBSET
    # This keeps the chain valid (train_ds -> balanced_train_ds)
    return Subset(dataset_or_subset, cast(Any, final_indices))


def read_json_data_path(json_path: str):
    """
    reading the CelebA-Spoof JSON
    """
    with open(json_path) as f:
        celeba_data = json.load(f)
    if not isinstance(celeba_data, dict):
        raise ValueError(
            'The JSON data is not in the expected dictionary format.',
        )
    return celeba_data


def get_data_for_training(
    json_path: str,
    train_count: int,
    val_count: int,
    spoof_percent: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, list], dict[str, list]]:
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
        if path.startswith('Data/train/')  # Only training set
    }

    print(f"Total training images: {len(train_paths_only)}")

    # 3. Split by subject (subject-disjoint)
    train_subject_paths, val_subject_paths = split_json_by_subject(
        train_paths_only,
    )

    print(
        f"Subject-split -"
        f" Train: {len(train_subject_paths)}"
        f" Val: {len(val_subject_paths)}"
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
    print_stats(train_dict, 'Training')
    print_stats(val_dict, 'Validation')

    return train_dict, val_dict


def balance_by_labels(
    path_list: list[str],
    target_count: int,
    spoof_percent: float,
    celeba_data: dict,
) -> list[str]:
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
        raise ValueError(
            f"Insufficient live images:"
            f" {len(live_paths)}"
            f" < {live_count}",
        )
    if len(spoof_paths) < spoof_count:
        raise ValueError(
            f"Insufficient spoof images:"
            f" {len(spoof_paths)}"
            f" < {spoof_count}",
        )
    # Random selection
    selected_live = np.random.choice(live_paths, live_count, replace=False)
    selected_spoof = np.random.choice(spoof_paths, spoof_count, replace=False)

    # Combine and shuffle
    selected_paths = np.concatenate([selected_live, selected_spoof])
    np.random.shuffle(selected_paths)

    return list(selected_paths)


def split_json_by_subject(
    celeba_data: dict[str, list],
    val_split: float = 0.2,  # Changed from 0.5 - typical 80/20 split
) -> tuple[list[str], list[str]]:
    """
    Create subject-disjoint splits for CelebA-Spoof.
    Only processes training data (paths starting with 'Data/train/').
    """
    # Group paths by subject ID
    subject_to_paths: dict[str, list[str]] = {}

    for path in celeba_data.keys():
        parts = path.split('/')
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
        f"Subjects: {len(subjects)} "
        f"total, {len(train_subjects)} "
        f"train, {len(val_subjects)} val",
    )
    print(f"Images: {len(train_paths)} train, {len(val_paths)} val")

    return train_paths, val_paths


def print_stats(data_dict: dict, name: str):
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


# if __name__ == '__main__':
#     import spoofdet.config as config

#     train_dict, val_dict = get_data_for_training(
#         json_path=str(config.TRAIN_JSON),
#         train_count=1000,
#         val_count=200,
#         spoof_percent=0.5,
#     # )
#     # train_ds = CelebASpoofDataset(
#     #     root_dir=config.ROOT_DIR,
#     #     json_label_path=train_dict,
#     #     bbox_json_path=config.BBOX_LOOKUP,
#     #     target_size=320,
#     #     bbox_original_size=config.BBOX_ORGINAL_SIZE,
#     # )
