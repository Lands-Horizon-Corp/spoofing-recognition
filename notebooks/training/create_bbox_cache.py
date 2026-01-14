import os
import json
import concurrent.futures
from tqdm import tqdm
import config

# CONFIG
JSON_LABEL_PATH = "path/to/train_label.json"
OUTPUT_BBOX_FILE = "bbox_lookup.json"


def process_single_file(rel_path):
    """
    Worker function to check one file.
    Returns (rel_path, coords) if found, else None.
    """
    full_path = os.path.join(config.ROOT_DIR, rel_path)
    base_path, _ = os.path.splitext(full_path)
    txt_path = base_path + "_BB.txt"

    try:
        with open(txt_path, "r") as f:
            coords = list(map(float, f.read().strip().split()))
            return rel_path, coords
    except (FileNotFoundError, ValueError, IndexError):
        return None


def generate_bbox_cache_threaded():
    print("Loading image list...")
    with open(config.TRAIN_JSON, "r") as f:
        label_dict = json.load(f)

    with open(config.TEST_JSON, "r") as f:
        test_label_dict = json.load(f)

    label_dict.update(test_label_dict)

    image_keys = list(label_dict.keys())
    bbox_cache = {}

    print(f"Scanning {len(image_keys)} files with multithreading...")

    # Adjust max_workers based on your disk speed.
    # For SSD: 16-32 is usually good. For HDD: 4-8.
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(
            tqdm(executor.map(process_single_file, image_keys), total=len(image_keys))
        )

    for res in results:
        if res is not None:
            bbox_cache[res[0]] = res[1]

    print(f"Found {len(bbox_cache)} bounding boxes.")
    print(f"Saving to {OUTPUT_BBOX_FILE}...")
    with open(OUTPUT_BBOX_FILE, "w") as f:
        json.dump(bbox_cache, f)
    print("Done!")


if __name__ == "__main__":
    generate_bbox_cache_threaded()
