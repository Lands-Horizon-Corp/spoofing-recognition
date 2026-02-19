from __future__ import annotations

import gc
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
        transform,
        data_count=None,
    ):
        self.root_dir = root_dir
        self.json_label_path = json_label_path
        self.bbox_json_path = bbox_json_path
        self.target_size = target_size
        self.bbox_original_size = bbox_original_size
        self.samples = []  # Will hold tuples of (rel_path, binary_label, bbox)
        self.targets = []  # Will hold just the binary labels for easy access
        # Assume folder structure: root_dir/class_name/*.jpg
        self.transform = transform

        print('Loading JSON metadata...')
        if isinstance(json_label_path, dict):
            raw_labels = json_label_path
        else:
            with open(json_label_path, encoding='utf-8') as f:
                raw_labels = json.load(f)

        with open(bbox_json_path, encoding='utf-8') as f:
            raw_bboxes = json.load(f)

        # 2. FLATTEN DATA: Convert Dicts to a simple List of Tuples
        # Format: [(rel_path, binary_label, bbox_list), ...]
        # This removes the overhead of dictionary hashmaps and duplicate keys
        self.samples = []

        print('Processing metadata into optimized list...')
        # Iterate once to build the clean list
        for rel_path, label_data in raw_labels.items():
            # Process label logic HERE, not in __getitem__ (saves CPU time later)
            raw_label_val = label_data[43] if isinstance(
                label_data, list) else label_data
            binary_label = 1.0 if raw_label_val > 0 else 0.0

            # Get bbox if exists
            bbox = raw_bboxes.get(rel_path, None)

            # Append tuple: (path, label, bbox)
            self.samples.append((rel_path, binary_label, bbox))
            # Keep a separate list of labels for fast access
            self.targets.append(binary_label)

        # 3. DELETE THE DICTIONARIES TO FREE RAM
        del raw_labels
        del raw_bboxes
        gc.collect()  # Force Python to release memory immediately
        print(f'Metadata processed. Dataset size: {len(self.samples)}')

        self.image_keys = [sample[0] for sample in self.samples]
        self.resize_op = v2.Resize((target_size, target_size), antialias=True)

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        rel_path, label_val, bbox = self.samples[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        img = read_image(full_path)
        _, real_h, real_w = img.shape

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
                    img, output_size=[
                        min(c_h, c_w), min(c_h, c_w),
                    ],
                )

        img = F.resize(
            img, size=[
                self.target_size,
                self.target_size,
            ], antialias=True,
        )
        img = F.center_crop(
            img, output_size=[
                self.target_size, self.target_size,
            ],
        )

        img = img.to(torch.uint8)
        if self.transform is not None:
            img = self.transform(img)

        #  Binary label: 0 = live, 1 = spoof

        binary_label = label_val  # Use the precomputed label value from samples list
        # BCEWithLogitsLoss expects float
        label = torch.tensor(binary_label, dtype=torch.float32)

        return img, label

    # dataset = CelebASpoofDataset(
    #     root_dir=config.ROOT_DIR,
    #     json_label_path=config.TRAIN_JSON,
    #     bbox_json_path=config.BBOX_LOOKUP,
    #     target_size=224,
    #     bbox_original_size=224,
    # # )
    # print(f"Dataset size: {len(dataset)}")
    # img, label = dataset[0]
    # print(f"Image shape: {img.shape}, Label: {label}")
