from pathlib import Path
import cv2 as cv
import numpy as np
from king_tensor.screen import Screen
from enum import Enum
import yaml

class Gamestate:

    def __init__(self, screen: Screen, config: dict[str, dict[str, np.ndarray]]):
        self.config = config
        self.elixir = self._parse_elixir(screen.images['elixir_bar'], self.config['elixir_bar']['elixir_thresh'])

    def _parse_elixir(self, img: np.ndarray, threshold: int, max_val: int = 255) -> float:
        gray_bar = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, thresh_bar = cv.threshold(gray_bar, threshold, max_val, cv.THRESH_BINARY)

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

if  __name__ == '__main__':
    from king_tensor.globals import CONFIG_DIR, DATA_DIR

    with open(CONFIG_DIR / 'gamestate.yaml', 'r') as file:
        gamestate_config = yaml.safe_load(file)
    with open(CONFIG_DIR / 'ui.yaml', 'r') as file:
        ui_config = yaml.safe_load(file)

    img_path = DATA_DIR / 'IMG_1470.PNG'
    img = cv.imread(img_path)
    screen = Screen(img, ui_config)
    gs = Gamestate(screen, gamestate_config)
    print(gs.elixir)
