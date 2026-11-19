"""Download a small, attributed gallery from the Kaggle car-image dataset."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parent
GALLERY_DIR = ROOT / "static" / "images" / "car-gallery"
DATASET = "kshitij192/cars-image-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET}"


def class_name_for(image: Path, dataset_root: Path) -> str:
    """Get the category directory, preferring the training split."""
    relative_parts = image.relative_to(dataset_root).parts
    for index, part in enumerate(relative_parts[:-1]):
        if part.casefold() == "train":
            return relative_parts[index + 1]
    return image.parent.name


def main() -> None:
    dataset_root = Path(kagglehub.dataset_download(DATASET))
    images_by_class: dict[str, list[Path]] = {}

    for image in dataset_root.rglob("*"):
        if image.is_file() and image.suffix.casefold() in {".jpg", ".jpeg"}:
            category = class_name_for(image, dataset_root)
            images_by_class.setdefault(category, []).append(image)

    if not images_by_class:
        raise FileNotFoundError(
            f"No JPG images were found in the downloaded dataset at {dataset_root}."
        )

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    gallery = []
    for category, images in sorted(images_by_class.items()):
        source = sorted(images)[0]
        slug = re.sub(r"[^a-z0-9]+", "-", category.casefold()).strip("-")
        filename = f"{slug}.jpg"
        shutil.copyfile(source, GALLERY_DIR / filename)
        gallery.append(
            {
                "label": category,
                "filename": filename,
                "source_file": source.relative_to(dataset_root).as_posix(),
            }
        )

    (GALLERY_DIR / "gallery.json").write_text(
        json.dumps(gallery, indent=2),
        encoding="utf-8",
    )
    (GALLERY_DIR / "ATTRIBUTION.txt").write_text(
        "Car Images Dataset by Kshitij Kumar\n"
        f"Source: {DATASET_URL}\n"
        "Dataset license: GPL-2.0 (as listed on Kaggle).\n"
        "The site gallery contains one representative image per dataset class.\n"
        "Review the dataset page and license before redistributing these images.\n",
        encoding="utf-8",
    )
    print(f"Downloaded {len(gallery)} gallery images to {GALLERY_DIR}")


if __name__ == "__main__":
    main()
