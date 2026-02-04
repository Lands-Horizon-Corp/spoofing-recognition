from __future__ import annotations

import json
import os
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from spoofdet.data_processing import get_data_for_training
from spoofdet.dataset import CelebASpoofDataset
from torch.utils.data import Subset
from torchvision import models
from torchvision.transforms import v2


def invert_label(y):
    return 1 - y


def check_dataset_distribution(dataset: Subset):
    live_count = 0
    for i in range(len(dataset)):
        data_item: Any = dataset[i]
        img, label = data_item  # Unpack the tuple explicitly
        if label.item() == 0:
            live_count += 1
    spoof_count = len(dataset) - live_count
    print(f"Live count: {live_count}, Spoof count: {spoof_count}")


def display_train_result(history: dict[str, list]) -> tuple[Figure, Figure]:
    """
    Displays training and validation loss and metrics history.
    Outputs two figures: one for loss and one for precision,
    Accuracy, recall, and F1 score.
    """
    fig_loss, ax1 = plt.subplots()
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Value')
    ax1.legend()
    fig_loss.show()

    fig_precision, ax2 = plt.subplots()
    ax2.plot(history['val_precision'], label='Val Precision')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.plot(history['val_recall'], label='Val Recall')
    ax2.plot(history['val_f1'], label='Val F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Value')
    ax2.legend()
    fig_precision.show()

    return fig_loss, fig_precision


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
        0: 'Live',
        1: 'Photo',
        2: 'Poster',
        3: 'A4',
        4: 'Face Mask',
        5: 'Upper Body Mask',
        6: 'Region Mask',
        8: 'Pad',
        7: 'PC',
        9: 'Phone',
        10: '3D Mask',
    }

    model.eval()

    # Dictionary to store results per spoof type
    spoof_type_results = {}

    with torch.no_grad():

        for idx in range(len(dataset)):
            data_item: Any = dataset[idx]
            img, label = data_item  # Unpack the tuple explicitly

            if hasattr(dataset, 'dataset'):  # If it's a Subset
                actual_idx = dataset.indices[idx]
                parent_dataset = cast(CelebASpoofDataset, dataset.dataset)
                image_key = parent_dataset.image_keys[actual_idx]
                full_labels = parent_dataset.label_dict[image_key]
            else:
                # Direct dataset access
                celeba_dataset = cast(CelebASpoofDataset, dataset)
                image_key = celeba_dataset.image_keys[idx]
                full_labels = celeba_dataset.label_dict[image_key]

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
            # pred = torch.argmax(output, dim=1).item()
            probs = torch.nn.functional.sigmoid(output)
            pred = (probs[:, 1] > 0.0143).long().item
            # Initialize spoof type entry if needed
            if spoof_type not in spoof_type_results:
                spoof_type_results[spoof_type] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                }

            # Update statistics
            spoof_type_results[spoof_type]['total'] += 1
            if pred == 1:  # Correctly identified as spoof
                spoof_type_results[spoof_type]['correct'] += 1
            else:  # Incorrectly identified as live
                spoof_type_results[spoof_type]['incorrect'] += 1

    # Calculate accuracy per spoof type
    results_list = []
    for spoof_type, stats in sorted(spoof_type_results.items()):
        accuracy = (
            (
                stats['correct'] / stats['total']
                * 100
            ) if stats['total'] > 0 else 0
        )
        results_list.append(
            {
                'Spoof Type': spoof_type_labels.get(spoof_type, 'Unknown'),
                'Type ID': spoof_type,
                'Total': stats['total'],
                'Correct': stats['correct'],
                'Incorrect': stats['incorrect'],
                'Accuracy (%)': accuracy,
            },
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('Accuracy (%)')

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of accuracy per spoof type
    ax1.barh(
        results_df['Spoof Type'].astype(str),
        results_df['Accuracy (%)'],
        color=[
            'red' if x < 50 else 'orange' if x < 80 else 'green'
            for x in results_df['Accuracy (%)']
        ],
    )
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_ylabel('Spoof Type')
    ax1.set_title('Model Accuracy by Spoof Type')
    ax1.axvline(
        x=50,
        color='red',
        linestyle='--',
        alpha=0.5,
        label='50% threshold',
    )
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Stacked bar chart showing correct vs incorrect
    ax2.barh(
        results_df['Spoof Type'].astype(str),
        results_df['Correct'],
        label='Correct (Detected as Spoof)',
        color='green',
        alpha=0.7,
    )
    ax2.barh(
        results_df['Spoof Type'].astype(str),
        results_df['Incorrect'],
        left=results_df['Correct'],
        label='Incorrect (Detected as Live)',
        color='red',
        alpha=0.7,
    )
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Spoof Type')
    ax2.set_title('Correct vs Incorrect Predictions by Spoof Type')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print('\nSpoof Type Analysis Results:')
    print('=' * 80)
    print(results_df.to_string(index=False))
    print('\n' + '=' * 80)
    print('\nWorst Performing Spoof Types (Accuracy < 70%):')
    worst = results_df[results_df['Accuracy (%)'] < 70]
    if len(worst) > 0:
        print(worst.to_string(index=False))
    else:
        print('None - All spoof types have >70% accuracy!')

    return results_df, fig


def analyze_dataset_spoof_distribution(
    dataset: torch.utils.data.Subset,  # Added type hint for clarity
) -> tuple[pd.DataFrame, Figure]:
    """
    Analyzes the distribution of spoof types in the dataset.
    """
    spoof_type_labels = {
        1: 'Photo',
        2: 'Poster',
        3: 'A4',
        4: 'Face Mask',
        5: 'Upper Body Mask',
        6: 'Region Mask',
        7: 'PC',
        8: 'Pad',
        9: 'Phone',
        10: '3D Mask',
    }

    spoof_type_counts: dict[str, int] = {}
    live_count = 0

    print('Analyzing spoof type distribution...')

    if hasattr(dataset, 'dataset'):  # It is a Subset
        parent_dataset = cast(CelebASpoofDataset, dataset.dataset)
        indices = dataset.indices
    else:  # It is the original dataset
        parent_dataset = cast(CelebASpoofDataset, dataset)
        indices = range(len(dataset))
    current_dataset = dataset
    final_indices = list(range(len(dataset)))

    # unwraps subset inside subset inside subset...
    while isinstance(current_dataset, torch.utils.data.Subset):
        # Map the current indices to the parent's indices
        final_indices = [current_dataset.indices[i] for i in final_indices]
        current_dataset = current_dataset.dataset

    # Now current_dataset is the root (CelebASpoofDataset)
    parent_dataset = cast(CelebASpoofDataset, current_dataset)
    for idx in indices:

        image_key = parent_dataset.image_keys[idx]
        full_labels = parent_dataset.label_dict[image_key]

        live_spoof_label = full_labels[43]  # Live/Spoof at index 43

        if live_spoof_label == 0:  # Live
            live_count += 1
        else:  # Spoof
            spoof_type = full_labels[40]
            spoof_type_counts[spoof_type] = (
                spoof_type_counts.get(
                    spoof_type,
                    0,
                )
                + 1
            )

    results = []

    for spoof_type, count in sorted(spoof_type_counts.items()):
        results.append(
            {
                'Spoof Type': spoof_type_labels.get(int(spoof_type), 'Unknown'),
                'Type ID': spoof_type,
                'Count': count,
            },
        )

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print('Warning: Dataset is empty.')
        return results_df, plt.figure()

    results_df['Percentage'] = (
        results_df['Count'] / len(dataset) * 100
    ).round(2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = [
        'green' if row['Spoof Type'] == 'Live' else 'red'
        for _, row in results_df.iterrows()
    ]
    ax1.barh(
        results_df['Spoof Type'],
        results_df['Count'],
        color=colors,
        alpha=0.7,
    )
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Class Type')
    ax1.set_title('Class Distribution in Dataset')
    ax1.grid(axis='x', alpha=0.3)

    ax2.pie(
        results_df['Count'],
        labels=results_df['Spoof Type'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
    )
    ax2.set_title('Distribution (Percentage)')

    plt.tight_layout()
    plt.show()

    print('\nClass Distribution:')
    print('=' * 60)
    print(results_df.to_string(index=False))
    print('=' * 60)

    return results_df, fig


def display_params(
    model_name,
    lr,
    weight_decay,
    batch_size,
    epochs,
    early_stopping_limit,
    target_size,
    train_size,
    val_size,
    num_unfrozen_layers=0,
    backbone_lr=0.0,
    head_lr=0.0,
):
    print('Training Configuration:')
    print(f" Model Name: {model_name}")
    print(f" Batch Size: {batch_size}")
    print(f" Learning Rate: {lr}")
    print(f" Weight Decay: {weight_decay}")
    print(f" Epochs: {epochs}")
    print(f" Early Stopping Limit: {early_stopping_limit}")
    print(f" Target Size: {target_size}")
    print(f" Train Size: {train_size}")
    print(f" Validation Size: {val_size}")
    print(f" Unfrozen Layers: {num_unfrozen_layers}")
    print(f" Backbone LR: {backbone_lr}")
    print(f" Head LR: {head_lr}")

    return json.dumps(
        {
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'early_stopping_limit': early_stopping_limit,
            'target_size': target_size,
            'train_size': train_size,
            'validation size': val_size,
            'unfrozen layers': num_unfrozen_layers,
            'backbone lr': backbone_lr,
            'head lr': head_lr,
        },
        indent=4,
    )


def check_subject_leakage(root_dir):
    """
    Scans root/train and root/test to ensure no Subject IDs overlap.
    Assumes structure: root / split / subject_id / ...
    """
    splits = ['train', 'test']
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
    overlap = subject_sets['train'].intersection(subject_sets['test'])

    if len(overlap) > 0:
        print('\n CRITICAL FAILURE: DATA LEAK DETECTED! ')
        print(f"Found {len(overlap)} subjects that are in BOTH Train and Test.")
        print(f"Example Leaked IDs: {list(overlap)[:5]}")
        print(
            'ACTION: You must remove these subjects from one of the sets or re-split.',
        )
    else:
        print('\n SUCCESS: No subject leakage detected.')
        print('Train and Test sets are completely independent.')


def verify_subject_split(dataset, train_indices, val_indices):
    """
    Verifies that train and validation splits have no overlapping subject IDs.

    Args:
        dataset: The CelebASpoofDataset instance
        train_indices: List of training indices
        val_indices: List of validation indices

    Returns:
        bool: True if no overlap, False if overlap detected
    """
    # Extract subject IDs from train indices
    train_subjects = set()
    for idx in train_indices:
        path = dataset.image_keys[idx]
        parts = path.split('/')
        if len(parts) > 2:
            subj_id = parts[-3]
        else:
            subj_id = 'unknown'
        train_subjects.add(subj_id)

    # Extract subject IDs from val indices
    val_subjects = set()
    for idx in val_indices:
        path = dataset.image_keys[idx]
        parts = path.split('/')
        if len(parts) > 2:
            subj_id = parts[-3]
        else:
            subj_id = 'unknown'
        val_subjects.add(subj_id)

    # Check for overlap
    overlap = train_subjects.intersection(val_subjects)

    print('Subject Split Verification:')

    print(f"Training Subjects:   {len(train_subjects)}")
    print(f"Validation Subjects: {len(val_subjects)}")
    print(f"Overlapping Subjects: {len(overlap)}")

    if len(overlap) > 0:
        print('\nWARNING: SUBJECT LEAKAGE DETECTED!')
        print(f"Found {len(overlap)} subjects in BOTH train and val")
        print(f"Example leaked subjects: {list(overlap)[:10]}")
        print(f"{'='*60}\n")
        return False
    else:
        print('\n SUCCESS: No subject leakage detected')
        print('Train and validation sets are completely disjoint')
        print(f"{'='*60}\n")
        return True


def get_model(with_weights: bool = False) -> torch.nn.Module:
    """getting the model for either training or inference"""

    if with_weights:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT,
        )
    else:
        model = models.efficientnet_v2_s(weights=None)

    # Get the number of input features from the existing classifier
    classifier_layer = model.classifier[1]
    if isinstance(classifier_layer, nn.Linear):
        in_features = classifier_layer.in_features
    else:
        # Handle other module types
        in_features = 1280  # Default for EfficientNet-V2-S
    model.classifier[1] = nn.Linear(in_features, 2)

    return model


def diagnose_dataset_issue(json_path, root_dir, bbox_path):
    """
    Run comprehensive diagnostics on your data pipeline.
    """
    # 1. Load your current data pipeline
    train_dict, val_dict = get_data_for_training(
        json_path=json_path,
        train_count=100,
        val_count=20,
        spoof_percent=0.5,
    )

    # 2. Check for label mismatches between folder names and JSON
    mismatches = 0
    sample_size = min(200, len(train_dict))

    print('Checking label consistency...')
    for i, (path, label_data) in enumerate(list(train_dict.items())[:sample_size]):
        # Get label from JSON
        if isinstance(label_data, list):
            json_label = int(label_data[43])
        else:
            json_label = int(label_data)

        # Get label from folder name
        parts = path.split('/')
        folder_label_str = parts[-2]  # "live" or "spoof"
        folder_label = 0 if folder_label_str == 'live' else 1

        if json_label != folder_label:
            mismatches += 1
            if mismatches <= 5:  # Show first 5 mismatches
                print(f"  MISMATCH: {path}")
                print(f"    JSON says: {
                      'Live' if json_label == 0 else 'Spoof'
                      } {json_label}")
                print(f"    Folder says: {folder_label_str} ({folder_label})")

    print(f"\nFound {mismatches} mismatches in {
          sample_size
          } samples ({mismatches/sample_size*100:.1f}%)")

    # 3. Check data leakage
    print('\nChecking data leakage...')
    train_subjects = set()
    val_subjects = set()

    for path in train_dict.keys():
        parts = path.split('/')
        if len(parts) >= 3:
            train_subjects.add(parts[2])  # Subject ID

    for path in val_dict.keys():
        parts = path.split('/')
        if len(parts) >= 3:
            val_subjects.add(parts[2])  # Subject ID

    overlap = train_subjects.intersection(val_subjects)
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Overlapping subjects: {len(overlap)}")

    # 4. Check class balance
    print('\nChecking class balance...')
    for name, data_dict in [('Train', train_dict), ('Val', val_dict)]:
        live_count = 0
        for path, label_data in data_dict.items():
            if isinstance(label_data, list):
                label = int(label_data[43])
            else:
                label = int(label_data)
            if label == 0:
                live_count += 1

        total = len(data_dict)
        print(f"{name}: {total} images")
        print(f"  Live: {live_count} ({live_count/total*100:.1f}%)")
        spoof_counts = total - live_count
        percentage_spoof = spoof_counts / total * 100
        print(f"  Spoof: {spoof_counts} ({percentage_spoof:.1f}%)")

    return mismatches, len(overlap)


def check_spoof_type_distribution(data_dict):
    """Check which spoof types are actually in your set."""
    spoof_type_counts = {}

    for path, labels in data_dict.items():
        if isinstance(labels, list) and len(labels) > 40:
            spoof_type = labels[40]  # Index 40 = spoof type
            spoof_type_counts[spoof_type] = (
                spoof_type_counts.get(
                    spoof_type,
                    0,
                )
                + 1
            )

    spoof_type_names = {
        1: 'Photo',
        2: 'Poster',
        3: 'A4',
        4: 'Face Mask',
        5: 'Upper Body Mask',
        6: 'Region Mask',
        7: 'PC',
        8: 'Pad',
        9: 'Phone',
        10: '3D Mask',
    }

    print('\nCurrent Spoof Type Distribution:')
    print('=' * 50)
    total = sum(spoof_type_counts.values())

    for spoof_type, count in sorted(spoof_type_counts.items()):
        name = spoof_type_names.get(spoof_type, f"Unknown({spoof_type})")
        print(f"{name:20} {count:5} ({count/total*100:5.1f}%)")

    return spoof_type_counts


if __name__ == '__main__':
    import spoofdet.config as config

    train_dict, val_dict = get_data_for_training(
        json_path=str(config.TRAIN_JSON),
        train_count=1000,
        val_count=200,
        spoof_percent=0.5,
    )
    spoof_counts = check_spoof_type_distribution(val_dict)

    diagnose_dataset_issue(
        json_path=config.TRAIN_JSON,
        root_dir=config.ROOT_DIR,
        bbox_path=config.BBOX_LOOKUP,
    )
