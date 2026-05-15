from __future__ import annotations

from pathlib import Path

from lightning import LightningDataModule
from spoofdet.data_processing.data_augmentation import get_transform_pipeline
from spoofdet.data_processing.dataset import CelebASpoofDataset
from spoofdet.data_processing.train_val_split import create_subset
from spoofdet.efficient_net.model_utils import get_data_for_training
from torch.utils.data import DataLoader


class SpoofDetDataModule(LightningDataModule):
    def __init__(self,
                 json_train_path: str,
                 json_test_path: Path,
                 root_dir: Path,
                 bbox_lookup_path: Path,
                 target_size,
                 bbox_original_size,
                 train_img_count=1000,
                 val_img_count=200,
                 test_img_count=200,
                 spoof_percent=0.5,
                 batch_size=32,
                 num_workers=2):
        super().__init__()
        self.json_train_path = json_train_path
        self.json_test_path = json_test_path
        self.target_size = target_size
        self.root_dir = root_dir
        self.bbox_lookup_path = bbox_lookup_path
        self.bbox_original_size = bbox_original_size
        self.train_img_count = train_img_count
        self.val_img_count = val_img_count
        self.test_img_count = test_img_count
        self.spoof_percent = spoof_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage) -> None:
        train_dict, val_dict = get_data_for_training(
            self.json_train_path,
            train_count=self.train_img_count,
            val_count=self.val_img_count,
            spoof_percent=self.spoof_percent)

        train_transform, val_transform = get_transform_pipeline(
            target_size=self.target_size)

        if stage == 'fit' or stage is None:
            self.train_ds = CelebASpoofDataset(
                root_dir=self.root_dir,
                json_label_path=train_dict,
                bbox_json_path=self.bbox_lookup_path,
                target_size=self.target_size,
                bbox_original_size=self.bbox_original_size,
                transform=train_transform
            )

            self.val_ds = CelebASpoofDataset(
                root_dir=self.root_dir,
                json_label_path=val_dict,
                bbox_json_path=self.bbox_lookup_path,
                target_size=self.target_size,
                bbox_original_size=self.bbox_original_size,
                transform=val_transform
            )
        if stage == 'test':
            self.test_ds = CelebASpoofDataset(
                root_dir=self.root_dir,
                json_label_path=self.json_test_path,
                bbox_json_path=self.bbox_lookup_path,
                target_size=self.target_size,
                bbox_original_size=self.bbox_original_size,
                transform=val_transform
            )

            self.small_test_ds = create_subset(
                self.test_ds, total_size=self.test_img_count, spoof_percent=self.spoof_percent)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.
                          batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.small_test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
