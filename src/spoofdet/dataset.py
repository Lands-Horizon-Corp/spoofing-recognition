import os
import json
import torch
from torch.utils.data import Dataset
import kornia.augmentation as K
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.transforms import functional as F


class CelebASpoofDataset(Dataset):
    def __init__(
        self,
        root_dir,
        json_label_path,
        bbox_json_path,
        target_size,
        bbox_original_size,
        transform=None,
    ):
        self.root_dir = root_dir
        self.json_label_path = json_label_path
        self.bbox_json_path = bbox_json_path
        self.target_size = target_size
        self.bbox_original_size = bbox_original_size

        with open(json_label_path, "r", encoding="utf-8") as f:
            self.label_dict = json.load(f)

        print("Loading BBox Cache into RAM...")
        with open(bbox_json_path, "r", encoding="utf-8") as f:
            self.bbox_dict = json.load(f)

        self.image_keys = list(self.label_dict.keys())
        self.resize_op = v2.Resize((target_size, target_size), antialias=True)

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

            img = F.crop(img, top=y, left=x, height=h, width=w)

        img = F.resize(img, size=(self.target_size, self.target_size), antialias=True)

        label_data = self.label_dict[rel_path]
        label = (
            torch.tensor(label_data[43])
            if isinstance(label_data, list)
            else torch.tensor(label_data)
        )

        return img, label
