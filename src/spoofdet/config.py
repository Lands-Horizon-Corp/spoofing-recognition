from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ROOT_DIR = BASE_DIR / 'dataset/CelebA_Spoof'
TRAIN_JSON = BASE_DIR / 'dataset/CelebA_Spoof/metas/intra_test/train_label.json'
TEST_JSON = BASE_DIR / 'dataset/CelebA_Spoof/metas/intra_test/test_label.json'

BBOX_LOOKUP = BASE_DIR / 'bbox_lookup.json'

REAL_VS_FAKE_PATH = BASE_DIR / 'dataset/archive/real_vs_fake/real-vs-fake'

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


MODEL_NAME = 'mobile_net_v4'

BATCH_SIZE = 32
EPOCHS = 3
EARLY_STOPPING_LIMIT = 30
BBOX_ORIGINAL_SIZE = 224
# TARGET_SIZE = 500
TARGET_SIZE = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5


BACKBONE_LR = 1e-6
HEAD_LR = 3e-4
NUM_FROZEN_LAYERS = 0
WORKERS = 4

SPOOF_PERCENT = 0.7
TRAIN_IMG_COUNT = 10000
VAL_IMG_COUNT = 2000
TEST_IMG_COUNT = 2000

NUM_WORKERS = 2


if __name__ == '__main__':
    print(f"Project Root is: {BASE_DIR}")
