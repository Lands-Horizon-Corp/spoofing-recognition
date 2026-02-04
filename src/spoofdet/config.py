from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ROOT_DIR = BASE_DIR / 'dataset/CelebA_Spoof'
TRAIN_JSON = BASE_DIR / 'dataset/CelebA_Spoof/metas/intra_test/train_label.json'
TEST_JSON = BASE_DIR / 'dataset/CelebA_Spoof/metas/intra_test/test_label.json'

BBOX_LOOKUP = BASE_DIR / 'bbox_lookup.json'

REAL_VS_FAKE_PATH = BASE_DIR / 'dataset/archive/real_vs_fake/real-vs-fake'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


BATCH_SIZE = 32
EPOCHS = 200
EARLY_STOPPING_LIMIT = 30
BBOX_ORGINAL_SIZE = 224
# TARGET_SIZE = 500
TARGET_SIZE = 224
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

WORKERS = 4

SPOOF_PERCENT = 0.7


if __name__ == '__main__':
    print(f"Project Root is: {BASE_DIR}")
