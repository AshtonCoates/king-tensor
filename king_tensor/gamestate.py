from pathlib import Path
import cv2 as cv
from PIL import Image
import numpy as np
from king_tensor.screen import Screen
from enum import Enum
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import List, Tuple
from collections.abc import Callable

def predict_card(
    model: torch.nn.Module,
    img: np.ndarray,
    classes: List[str],
    tfm,
) -> Tuple[str, int, torch.Tensor]:
    """
    Run a single card image through the model and return:
      - predicted class name
      - predicted class index
      - raw logits tensor (1, num_classes)
    
    Args:
        model: Trained PyTorch model.
        img: Image as a NumPy array (H, W, 3). Assumed BGR if from OpenCV.
        classes: List of class names (e.g. train_ds.classes).

        tfm: Transform to apply to the image (e.g. val_tfm).
    """

    # Convert BGR (OpenCV) -> RGB
    if img.shape[-1] == 3:
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    else:
        img_rgb = img  # just in case it's already RGB/gray

    pil_img = Image.fromarray(img_rgb)

    # Apply transforms and add batch dimension
    x = tfm(pil_img).unsqueeze(0)  # shape: (1, C, H, W)

    # Move to same device as model
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)  # (1, num_classes)
        pred_idx = logits.argmax(dim=1).item()

    pred_class = classes[pred_idx]
    return pred_class, pred_idx, logits


class Gamestate:

    def __init__(
        self,
        screen: Screen,
        model: torch.nn.Module,
        tfm: Callable,
        config: dict[str, dict[str, np.ndarray]],
    ):
        self.config = config
        self.elixir = self._parse_elixir(screen.images['elixir_bar'], self.config['elixir_bar']['elixir_thresh'])
        self.hand = self._parse_hand()

    def _parse_elixir(self, img: np.ndarray, threshold: int, max_val: int = 255) -> float:
        gray_bar = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh_bar = cv.threshold(gray_bar, threshold, max_val, cv.THRESH_BINARY)

        H, W = thresh_bar.shape

        col_fill = (thresh_bar == max_val).sum(axis=0) / H
        filled_cols = col_fill > 0.5
        filled_len = filled_cols.sum()      # number of filled columns
        elixir = 10.0 * filled_len / W      # map [0, W] -> [0, 10]
        return elixir

    def _parse_hand(self, images: list[np.ndarray]):
        pass

    def _parse_next(self, img: np.ndarray):
        pass

if  __name__ == '__main__':
    from king_tensor.globals import CONFIG_DIR, DATA_DIR, PACKAGE_DIR
    from king_tensor.models import create_model

    img_path = DATA_DIR / 'card_classifier' / 'raw' / 'rg.png'
    img = cv.imread(img_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    classes = ['bandit', 'battle_ram', 'electro_wizard', 'minions', 'pekka', 'poison', 'royal_ghost', 'zap']

    num_classes = len(classes)
    model, card_tfm, model_path = create_model("v1", num_classes)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    card, idx, logits = predict_card(model, img_rgb, classes, card_tfm)
    print(card)
