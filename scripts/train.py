from __future__ import annotations

import spoofdet.config as config
from spoofdet.data_module import SpoofDetDataModule
from spoofdet.models.sota_model import SpoofingDetection
from spoofdet.trainer import trainer


def main():
    celeba_spoof_data_module = SpoofDetDataModule(
        json_train_path=str(config.TRAIN_JSON),
        json_test_path=config.TEST_JSON,
        root_dir=config.ROOT_DIR,
        bbox_lookup_path=config.BBOX_LOOKUP,
        target_size=config.TARGET_SIZE,
        bbox_original_size=config.BBOX_ORIGINAL_SIZE,
        train_img_count=config.TRAIN_IMG_COUNT,
        val_img_count=config.VAL_IMG_COUNT,
        test_img_count=config.TEST_IMG_COUNT,
        spoof_percent=config.SPOOF_PERCENT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    model = SpoofingDetection(
        backbone_model=config.MODEL_NAME,
        backbone_lr=config.BACKBONE_LR,
        head_lr=config.HEAD_LR,
        target_size=config.TARGET_SIZE,
        frozen_stages=config.NUM_FROZEN_LAYERS
    )

    trainer.fit(model, datamodule=celeba_spoof_data_module)

    trainer.test(model, datamodule=celeba_spoof_data_module)


if __name__ == '__main__':
    main()
