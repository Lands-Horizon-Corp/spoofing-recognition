from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F
from torchvision.transforms import v2


class CelebASpoofDataset(Dataset):
    def __init__(
        self,
        root_dir,
        json_label_path,
        bbox_json_path,
        target_size,
        bbox_original_size,
        transform=None,
        data_count=None,
    ):
        self.root_dir = root_dir
        self.json_label_path = json_label_path
        self.bbox_json_path = bbox_json_path
        self.target_size = target_size
        self.bbox_original_size = bbox_original_size

        if isinstance(json_label_path, dict):
            self.label_dict = json_label_path
        else:
            with open(json_label_path, encoding='utf-8') as f:
                self.label_dict = json.load(f)

        print('Loading BBox Cache into RAM...')
        with open(bbox_json_path, encoding='utf-8') as f:
            self.bbox_dict = json.load(f)

        self.image_keys = list(self.label_dict.keys())
        self.resize_op = v2.Resize((target_size, target_size), antialias=True)
        self.buffer_size = 600

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        rel_path = self.image_keys[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        img = read_image(full_path)
        _, real_h, real_w = img.shape
        bbox = self.bbox_dict.get(rel_path)

        if bbox is not None and len(bbox) >= 4:
            scale_x = real_w / self.bbox_original_size
            scale_y = real_h / self.bbox_original_size
            x, y, w, h = (
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y),
            )
            if w > 0 and h > 0:
                img = F.crop(img, top=y, left=x, height=h, width=w)
            else:
                c_h, c_w = img.shape[-2:]
                img = F.center_crop(
                    img, output_size=(
                        min(c_h, c_w), min(c_h, c_w),
                    ),
                )

        img = F.resize(img, size=self.buffer_size, antialias=True)
        img = F.center_crop(
            img, output_size=(
                self.buffer_size, self.buffer_size,
            ),
        )

        img = img.to(torch.uint8)
        label_data = self.label_dict[rel_path]

        # Get the raw label (likely Spoof Type: 0=Live, >0=Spoof)
        raw_label = label_data[43] if isinstance(
            label_data, list,
        ) else label_data

        assert isinstance(raw_label, int), f"Label must be int, got {
            type(raw_label)
        }"
        assert raw_label >= 0, f"Label must be non-negative, got {raw_label}"
        assert raw_label <= 1, f"Label must be in [0,1], got {raw_label}"

        # Convert to binary: 0 (Live) vs 1 (Spoof)
        label = torch.tensor(raw_label, dtype=torch.long)

        return img, label


if __name__ == '__main__':
    from spoofdet import config

    dataset = CelebASpoofDataset(
        root_dir=config.ROOT_DIR,
        json_label_path=config.TRAIN_JSON,
        bbox_json_path=config.BBOX_LOOKUP,
        target_size=224,
        bbox_original_size=224,
    )

    print(f"Dataset size: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
