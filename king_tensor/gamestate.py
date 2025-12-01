from pathlib import Path
import cv2 as cv
import numpy as np
from king_tensor.screen import Screen
from enum import Enum
import yaml

class Gamestate:

    def __init__(screen: Screen, config: Path):
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)

    def _parse_elixir(img: np.ndarray, threshold: int, max_val: int = 255) -> float:
        # thresh_bar: cropped binary image of JUST the bar (0 or max_val)
        H, W = thresh_bar.shape

        # fraction of bright pixels in each column
        col_fill = (thresh_bar == max_val).sum(axis=0) / H

        # treat column as filled if most of it is bright
        filled_cols = col_fill > 0.5

        filled_len = filled_cols.sum()      # number of filled columns
        elixir = 10.0 * filled_len / W      # map [0, W] -> [0, 10]
        return elixir

    def _parse_hand(self, images: list[np.ndarray]):
        pass

    def _parse_next(self, img: np.ndarray):
        pass

    def _parse_elixir

