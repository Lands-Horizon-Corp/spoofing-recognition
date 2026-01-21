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

        # img = F.resize(img, size=(self.target_size, self.target_size), antialias=True)

        # Get current dimensions of the crop
        curr_h, curr_w = img.shape[-2], img.shape[-1]  # Shape is (C, H, W)

        # Calculate Scale to fit longest edge into target_size
        scale = self.target_size / max(curr_h, curr_w)

        # Calculate new dimensions
        new_h = int(curr_h * scale)
        new_w = int(curr_w * scale)

        # Resize the image to these new dimensions (this preserves aspect ratio)
        img = F.resize(img, size=(new_h, new_w), antialias=True)

        # Calculate Padding required to reach target_size
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h

        # Split padding for centering (Left, Top, Right, Bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply Padding
        # fill=0 is Black, fill=114 is YOLO-style Grey.
        img = F.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

        label_data = self.label_dict[rel_path]
        label = (
            torch.tensor(label_data[43])
            if isinstance(label_data, list)
            else torch.tensor(label_data)
        )

        return img, label
