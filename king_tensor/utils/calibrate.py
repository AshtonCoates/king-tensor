from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Mapping

import cv2
import mss
import numpy as np
import yaml
from PIL import Image

from king_tensor.globals import CONFIG_DIR


def _capture_screen() -> np.ndarray:
    """Grab a screenshot of the primary monitor and return it as an RGB array."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
    pil_image = Image.frombytes("RGB", raw.size, raw.rgb)
    return np.array(pil_image)


def _load_ui_entries(template_path: Path) -> Mapping[str, Mapping[str, int]]:
    """Load ui.yaml so we can iterate through known UI elements in order."""
    if not template_path.exists():
        raise FileNotFoundError(f"UI template not found: {template_path}")
    with template_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at {template_path}")
    return data


def _prompt_for_box(image_bgr: np.ndarray, label: str) -> dict[str, int]:
    prompt = f"Draw bounding box for '{label}'. Press ENTER/SPACE to confirm, C to cancel."
    print(prompt)
    while True:
        roi = cv2.selectROI(f"kt-calibrate-{label}", image_bgr, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        cv2.destroyWindow(f"kt-calibrate-{label}")
        if w == 0 or h == 0:
            print("No selection made. Try again or press Ctrl+C to abort.")
            continue
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _clip_to_screen(
    abs_box: dict[str, int],
    screen_origin: tuple[int, int],
    screen_size: tuple[int, int],
) -> dict[str, int] | None:
    screen_x1, screen_y1 = screen_origin
    screen_x2 = screen_x1 + screen_size[0]
    screen_y2 = screen_y1 + screen_size[1]

    box_x1 = abs_box["x"]
    box_y1 = abs_box["y"]
    box_x2 = box_x1 + abs_box["w"]
    box_y2 = box_y1 + abs_box["h"]

    inter_x1 = max(box_x1, screen_x1)
    inter_y1 = max(box_y1, screen_y1)
    inter_x2 = min(box_x2, screen_x2)
    inter_y2 = min(box_y2, screen_y2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return None

    rel_x = inter_x1 - screen_x1
    rel_y = inter_y1 - screen_y1
    return {"x": rel_x, "y": rel_y, "w": inter_w, "h": inter_h}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kt calibrate",
        description="Capture a screenshot and record bounding boxes for UI elements.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ui.yaml",
        help="Output filename placed inside the config directory (default: ui.yaml).",
    )
    parser.add_argument(
        "--template",
        default="ui.yaml",
        help="Template file under the config directory to determine element order (default: ui.yaml).",
    )
    parser.add_argument(
        "-f",
        "--full",
        action="store_true",
        help="Prompt for every UI element instead of only screen.",
    )
    parser.add_argument(
        "-t",
        "--timer",
        type=float,
        default=0.0,
        help="Delay before capturing the screenshot (seconds).",
    )
    return parser


def run(
    output_filename: str = "ui.yaml",
    template_filename: str = "ui.yaml",
    *,
    full: bool = False,
    timer: float = 0.0,
) -> None:
    template_path = CONFIG_DIR / template_filename
    elements = _load_ui_entries(template_path)

    if timer > 0:
        print(f"Waiting {timer:.1f} seconds before capturing...")
        time.sleep(timer)

    print("Capturing screenshot...")
    screenshot = _capture_screen()
    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cv2.imshow("kt calibrate screenshot", screenshot_bgr)
    cv2.waitKey(1)

    if full:
        results: dict[str, dict[str, int]] = {}
        screen_origin: tuple[int, int] | None = None
        screen_size: tuple[int, int] | None = None
        try:
            for name in elements:
                while True:
                    box = _prompt_for_box(screenshot_bgr, name)
                    if name == "screen":
                        screen_origin = (box["x"], box["y"])
                        screen_size = (box["w"], box["h"])
                        results[name] = box
                        break
                    if screen_origin is None or screen_size is None:
                        raise RuntimeError(
                            "Template must list 'screen' before other entries to compute relative coordinates."
                        )
                    clipped = _clip_to_screen(box, screen_origin, screen_size)
                    if clipped is None:
                        print(
                            f"Selection for '{name}' does not overlap the screen ROI. "
                            "Please reselect."
                        )
                        continue
                    results[name] = clipped
                    break
        finally:
            cv2.destroyAllWindows()
    else:
        results = dict(elements)
        try:
            screen_box = _prompt_for_box(screenshot_bgr, "screen")
            results["screen"] = screen_box
        finally:
            cv2.destroyAllWindows()

    output_name = Path(output_filename).name
    output_path = CONFIG_DIR / output_name
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(results, handle, sort_keys=False)
    print(f"Saved calibration to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args.output, args.template, full=args.full, timer=args.timer)


if __name__ == "__main__":
    main()
