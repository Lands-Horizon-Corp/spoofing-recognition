from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset
from torchvision.transforms import v2


def check_image(dataset, idx):
    sample_img, sample_label = dataset[idx]

    # 1. Convert to Numpy (H, W, C)
    display_img = sample_img.permute(1, 2, 0).numpy()

    # 2. Dynamic Un-Normalization
    # Case A: ImageNet Normalized (contains negative values)
    if display_img.min() < 0:
        print('Detected ImageNet Normalization. Un-normalizing...')
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_img = (display_img * std) + mean

    # Case B: Standard 0-255 range (e.g. uint8 or unscaled float)
    elif display_img.max() > 1.0:
        display_img = display_img / 255.0

    # Case C: Already 0-1 (Float)
    # do nothing

    # 3. Clip just in case (to avoid matplotlib warnings)
    display_img = np.clip(display_img, 0, 1)

    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'}")
    plt.axis('off')
    plt.show()


def check_augmented_image(device, dataset: Dataset, idx, gpu_transforms: v2.Compose):
    sample_img, sample_label = dataset[idx]
    viz_transforms = v2.Compose(
        [
            t
            for t in gpu_transforms.transforms
            if not isinstance(t, (v2.MixUp, v2.CutMix, v2.RandomChoice, v2.Normalize))
        ],
    )

    # Apply GPU transforms (same as training)
    sample_img = sample_img.unsqueeze(0).to(device)  # Add batch dim
    augmented = viz_transforms(sample_img).squeeze(0).cpu()  # Remove batch dim
    display_img = torch.clamp(augmented, 0, 1)

    display_img = display_img.permute(1, 2, 0).numpy()

    plt.imshow(display_img)
    plt.title(f"Label: {'Live' if sample_label == 0 else 'Spoof'} (Augmented)")
    plt.axis('off')
    plt.show()


def save_results(
    model: torch.nn.Module,
    confusion_matrix_fig: Figure,
    train_loss_fig: Figure,
    precision_fig: Figure,
    params: dict,
    spoof_fig: Figure,
):
    """
    save all training results to a new directory inside train_results/
    args:
    - model: trained model
    - confusion_matrix_fig: confusion matrix figure
    - train_loss_fig: training loss figure
    - precision_fig: precision figure
    - params: training parameters in json format
    - spoof_fig: spoof type analysis figure

    """

    save_path = Path('train_results')
    path_name = 'train'
    num = 0

    save_path.mkdir(parents=True, exist_ok=True)

    # create new dir if already exist
    new_path = _create_save_new_path(save_path, path_name, num)

    print(f"Saving results to: {new_path}")

    confusion_matrix_fig.savefig(
        new_path / 'confusion_matrix.png',
        bbox_inches='tight',
    )
    train_loss_fig.savefig(new_path / 'train_loss.png')
    precision_fig.savefig(new_path / 'precision.png')

    torch.save(model.state_dict(), new_path / 'model.pt')
    with open(new_path / 'params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)

    spoof_fig.savefig(
        new_path / 'spoof_type_analysis.png',
        bbox_inches='tight',
    )


def _create_save_new_path(save_path: Path, path_name: str, num: int) -> Path:
    new_dir = save_path / f"{path_name}_{num}"
    if new_dir.exists():
        return _create_save_new_path(save_path, path_name, num + 1)
    else:
        new_dir.mkdir(parents=True)
        return new_dir
