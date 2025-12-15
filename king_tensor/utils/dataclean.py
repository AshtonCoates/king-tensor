from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
import csv
from pathlib import Path
from typing import Iterable, Any


def _iter_pngs(directory: Path) -> Iterable[Path]:
    """Yield .png files in a directory in sorted order (case-insensitive)."""
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: p.name.lower(),
    )


def rename_pngs_to_sequence(directory: Path, zero_pad: int = 4) -> int:
    """
    Rename all .png files in `directory` to a sequential list starting at 1.

    Files are renamed to zero-padded numbers (e.g., 0001.png, 0002.png, ...).
    Files already matching the convention are left in place, and remaining
    files fill in the next available numbers after those.
    Returns the number of files renamed.
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    pngs = list(_iter_pngs(directory))
    if not pngs:
        return 0

    pattern = re.compile(rf"^\d{{{zero_pad}}}\.png$")
    reserved: dict[int, Path] = {}
    to_rename: list[Path] = []
    for p in pngs:
        if pattern.match(p.name):
            reserved[int(p.stem)] = p
        else:
            to_rename.append(p)

    if not to_rename:
        return 0

    def number_generator() -> Iterable[int]:
        n = 1
        used = set(reserved.keys())
        while True:
            if n not in used:
                yield n
            n += 1

    num_iter = number_generator()
    assignments: list[tuple[Path, Path]] = []
    for src in to_rename:
        num = next(num_iter)
        final_name = f"{num:0{zero_pad}d}.png"
        assignments.append((src, directory / final_name))

    temp_suffix = f".dataclean-temp-{uuid.uuid4().hex}"
    temp_paths: list[tuple[Path, int]] = []

    # First pass: move everything to temporary names so we never clobber files mid-run.
    for idx, src in enumerate(to_rename, start=1):
        temp_target = src.with_name(f"{src.name}{temp_suffix}")
        src.rename(temp_target)
        temp_paths.append((temp_target, idx))

    # Second pass: rename to the final numbered filenames.
    for (temp_path, _), (_, final_path) in zip(temp_paths, assignments):
        if final_path.exists():
            raise FileExistsError(
                f"Destination already exists: {final_path}. Nothing was changed."
            )
        temp_path.rename(final_path)

    return len(temp_paths)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataclean",
        description="Utilities for cleaning raw data files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rename_parser = subparsers.add_parser(
        "clean-names",
        help="Rename all .png files in a directory to 0001.png, 0002.png, ...",
    )
    rename_parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing .png files to rename.",
    )
    rename_parser.set_defaults(func=_handle_rename_pngs)
    sort_parser = subparsers.add_parser(
        "sort",
        help="Copy images into label-named folders from a CSV export.",
    )
    sort_parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV export file.",
    )
    sort_parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing source images.",
    )
    sort_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write label-organized images into (will be cleared first).",
    )
    sort_parser.set_defaults(func=_handle_sort)

    return parser


def _handle_rename_pngs(args: argparse.Namespace) -> None:
    renamed = rename_pngs_to_sequence(args.directory)
    print(f"Renamed {renamed} file(s) in {args.directory}")


def _safe_label_name(raw_label: str) -> str:
    """Normalize a label for use as a directory name."""
    cleaned = str(raw_label).strip()
    if not cleaned:
        raise ValueError("Found an empty label; cannot create directory.")
    # Replace path separators to avoid accidental traversal.
    cleaned = cleaned.replace(os.sep, "_")
    if os.altsep:
        cleaned = cleaned.replace(os.altsep, "_")
    return cleaned


def _resolve_image_path(raw_path: str, export_path: Path) -> Path:
    """Resolve an image path relative to the export file if necessary."""
    img_path = Path(raw_path)
    if not img_path.is_absolute():
        img_path = (export_path.parent / img_path).resolve()
    return img_path


def _extract_label_from_annotations(annotations: Any) -> str | None:
    """
    Pull the first choice label from Label Studio annotation structures.

    This favors the first choice in the first annotation to keep behavior predictable.
    """
    if not isinstance(annotations, list):
        return None
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        results = annotation.get("result")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            value = result.get("value")
            if not isinstance(value, dict):
                continue
            choices = value.get("choices")
            if isinstance(choices, list) and choices:
                return str(choices[0])
            choice = value.get("choice")
            if isinstance(choice, str) and choice:
                return choice
    return None


def _extract_item(item: Any, export_path: Path) -> tuple[Path, str]:
    """
    Extract an (image_path, label) pair from a Label Studio export item.

    Supports both flat (image/label) records and task-style structures with
    a `data.image` entry and annotations carrying choices.
    """
    if not isinstance(item, dict):
        raise ValueError("Item is not an object")

    # Flat representation
    image_value = item.get("image")
    label_value = item.get("label")

    # Task representation
    data_section = item.get("data") if isinstance(item.get("data"), dict) else None
    if data_section and "image" in data_section:
        image_value = image_value or data_section.get("image")
    if label_value is None:
        label_value = _extract_label_from_annotations(item.get("annotations"))

    if image_value is None:
        raise ValueError("Missing image path")
    if label_value is None:
        raise ValueError("Missing label")

    image_path = _resolve_image_path(str(image_value), export_path)
    label = _safe_label_name(str(label_value))
    return image_path, label


def _load_export(export_path: Path) -> list[tuple[Path, str]]:
    if not export_path.exists():
        raise ValueError(f"Export file does not exist: {export_path}")
    with export_path.open("r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {exc}") from exc

    if isinstance(payload, dict):
        # Accept common wrapping keys (e.g., {"tasks": [...]}) if present.
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            payload = tasks
        else:
            raise ValueError("Export JSON must be a list of tasks")

    if not isinstance(payload, list):
        raise ValueError("Export JSON must be a list of tasks")

    items: list[tuple[Path, str]] = []
    for idx, item in enumerate(payload, start=1):
        try:
            parsed = _extract_item(item, export_path)
        except ValueError as exc:
            raise ValueError(f"Item #{idx}: {exc}") from exc
        items.append(parsed)
    return items


def sort_label_studio_export(export_path: Path, output_dir: Path) -> int:
    """
    Wipe the output directory, then copy each image into label-named folders.

    Returns the number of images copied.
    """
    pairs = _load_export(export_path)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path, label in pairs:
        if not image_path.exists():
            raise ValueError(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Image path is not a file: {image_path}")
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = label_dir / image_path.name
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}. Nothing was changed."
            )
        shutil.copy2(image_path, destination)
        copied += 1
    return copied


def _load_yolo_classes(yolo_dir: Path) -> dict[int, str]:
    classes_txt = yolo_dir / "classes.txt"
    if classes_txt.exists():
        names = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines()]
        return {idx: name for idx, name in enumerate(names) if name}

    notes_json = yolo_dir / "notes.json"
    if notes_json.exists():
        try:
            payload = json.loads(notes_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {notes_json}: {exc}") from exc
        categories = payload.get("categories") if isinstance(payload, dict) else None
        if isinstance(categories, list):
            mapping: dict[int, str] = {}
            for cat in categories:
                if isinstance(cat, dict) and "id" in cat and "name" in cat:
                    mapping[int(cat["id"])] = str(cat["name"])
            if mapping:
                return mapping
    raise ValueError(
        f"Could not find classes.txt or parse categories in notes.json under {yolo_dir}"
    )


def _parse_yolo_label_file(label_path: Path) -> int:
    """
    Read the first class id from a YOLO label file.

    We expect classification-style exports where the first token is the class id.
    """
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        try:
            return int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}: invalid class id '{parts[0]}'") from exc
    raise ValueError(f"{label_path}: file is empty; no class id found")


def _find_image_for_label(stem: str, images_dir: Path) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise ValueError(f"Image not found for '{stem}' in {images_dir}")


def sort_yolo_export(yolo_dir: Path, output_dir: Path) -> int:
    """
    Wipe the output directory, then copy each image into label-named folders
    based on a YOLO dataset structure (images/, labels/, classes.txt).
    """
    if not yolo_dir.exists() or not yolo_dir.is_dir():
        raise ValueError(f"YOLO directory does not exist: {yolo_dir}")

    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    if not images_dir.is_dir():
        raise ValueError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"Missing labels directory: {labels_dir}")

    class_map = _load_yolo_classes(yolo_dir)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label_file in sorted(labels_dir.glob("*.txt"), key=lambda p: p.name.lower()):
        class_id = _parse_yolo_label_file(label_file)
        if class_id not in class_map:
            raise ValueError(f"{label_file}: class id {class_id} not found in classes.txt")
        label_name = _safe_label_name(class_map[class_id])
        image_path = _find_image_for_label(label_file.stem, images_dir)

        label_dir = output_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = label_dir / image_path.name
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}. Nothing was changed."
            )
        shutil.copy2(image_path, destination)
        copied += 1
    return copied


def _resolve_csv_image_path(raw_image: str, image_dir: Path) -> Path:
    """
    Resolve an image reference from the CSV.

    We ignore any path prefix and use the filename, and also try the portion
    after the last hyphen (e.g., 0001.png) if needed.
    """
    name = Path(raw_image).name
    candidates = []

    # First try the full name.
    candidates.append(image_dir / name)

    # Then try the suffix after the last hyphen if present.
    if "-" in name:
        suffix = name.split("-")[-1]
        if suffix != name:
            candidates.append(image_dir / suffix)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise ValueError(
        f"Image not found for '{raw_image}'. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def sort_csv_export(csv_path: Path, image_dir: Path, output_dir: Path) -> int:
    """
    Wipe the output directory, then copy each image into label-named folders
    based on a CSV export containing `image` and `label` columns.
    """
    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'image' and 'label' columns")
        rows = list(reader)

    copied = 0
    for idx, row in enumerate(rows, start=1):
        image_value = row.get("image")
        label_value = row.get("label")
        if image_value is None or label_value is None:
            raise ValueError(f"Row {idx}: missing image or label")

        image_path = _resolve_csv_image_path(str(image_value), image_dir)
        if not image_path.is_file():
            raise ValueError(f"Row {idx}: image path is not a file: {image_path}")

        label_dir = output_dir / _safe_label_name(str(label_value))
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = label_dir / image_path.name
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}. Nothing was changed."
            )
        shutil.copy2(image_path, destination)
        copied += 1

    return copied


def _handle_sort(args: argparse.Namespace) -> None:
    copied = sort_csv_export(args.csv_path, args.image_dir, args.output_dir)
    print(f"Copied {copied} image(s) into {args.output_dir} by label.")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        parser.error(str(exc))
    except FileExistsError as exc:
        # Use a clearer exit code for collisions.
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
