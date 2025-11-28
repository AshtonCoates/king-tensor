import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import yaml
import pathlib

@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

class Screen:
    """
    Class to parse the screen and store all objects
    """

    def __init__(self, img: np.ndarray, config: pathlib.Path):
        """
        Args:
            img (np.ndarray): The image of the screen
            config (pathlib.Path): Path to UI element config file
        Returns:
            Screen: Object containing images for each cropped component
        """
        self.img = img

        with open(config, 'r') as file:
            try:
                data = yaml.safe_load(file)
                print(data)
            except yaml.YAMLError as exc:
                print(exc)


if __name__ == '__main__':
    img_path = pathlib.Path('../data/IMG_1470.PNG')
    img = cv2.imread(img_path)
    config_path = pathlib.Path('../config/ui.yaml')
    print('--------------')
    print(config_path)
    screen = Screen(img, config_path)
