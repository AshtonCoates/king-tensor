from pathlib import Path
import cv2 as cv
from PIL import Image
import numpy as np
from enum import Enum
import yaml

from king_tensor.screen import Screen

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import List, Tuple
from collections.abc import Callable

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

    img_path = DATA_DIR / 'card_classifier' / 'raw' / 'rg.png'
    img = cv.imread(img_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    classes = ['bandit', 'battle_ram', 'electro_wizard', 'minions', 'pekka', 'poison', 'royal_ghost', 'zap']

    num_classes = len(classes)
    card_predictor = CardPredictor('v1', classes)
    card, idx, logits = card_predictor.predict_card(img_rgb)
    print(card)
