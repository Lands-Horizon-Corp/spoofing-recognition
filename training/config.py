from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ROOT_DIR = BASE_DIR / "dataset/CelebA_Spoof"
TRAIN_JSON = BASE_DIR / "dataset/CelebA_Spoof/metas/intra_test/train_label.json"
TEST_JSON = BASE_DIR / "dataset/CelebA_Spoof/metas/intra_test/test_label.json"


if __name__ == "__main__":
    print(f"Project Root is: {BASE_DIR}")
