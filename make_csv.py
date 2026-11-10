"""Merge dataset/potatoes, dataset/onions and dataset/both into one folder + labels.csv.

Result:
    dataset/all_images/image_0001.jpg, image_0002.jpg, ...
    dataset/labels.csv   (columns: filename, potato, onion)

Run from the folder that contains ``dataset/``:
    python prepare_dataset.py
"""

import csv
import shutil
from pathlib import Path

DATASET_DIR = Path("dataset")
OUTPUT_DIR = DATASET_DIR / "all_images"
CSV_PATH = DATASET_DIR / "labels.csv"

# source folder -> (potato label, onion label)
FOLDERS = {
    "potatoes": (1, 0),
    "onions": (0, 1),
    "both": (1, 1),
}


def main() -> None:
    # Make sure the three source folders exist.
    missing = [name for name in FOLDERS if not (DATASET_DIR / name).is_dir()]
    if missing:
        print(f"Error: missing folder(s) inside '{DATASET_DIR}': {', '.join(missing)}")
        return

    # Don't mix new copies with an earlier run.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Error: '{OUTPUT_DIR}' already contains files. Delete or rename it first, then run again.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created '{OUTPUT_DIR}'")

    counter = 0
    per_folder: dict[str, int] = {}

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "potato", "onion"])

        for folder_name, (potato, onion) in FOLDERS.items():
            images = sorted(
                p for p in (DATASET_DIR / folder_name).iterdir()
                if p.is_file() and p.suffix.lower() == ".jpg"
            )
            print(f"\nFound {len(images)} .jpg image(s) in '{folder_name}' -> labels potato={potato}, onion={onion}")

            for source in images:
                counter += 1
                new_name = f"image_{counter:04d}.jpg"
                shutil.copy2(source, OUTPUT_DIR / new_name)
                writer.writerow([new_name, potato, onion])

                if counter % 100 == 0:
                    print(f"  ...copied {counter} images so far")

            per_folder[folder_name] = len(images)

    print("\n" + "=" * 40)
    for folder_name, count in per_folder.items():
        print(f"  {folder_name:<10} {count:>6} images")
    print(f"  {'TOTAL':<10} {counter:>6} images")
    print(f"Images saved in : {OUTPUT_DIR}")
    print(f"Labels saved in : {CSV_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()