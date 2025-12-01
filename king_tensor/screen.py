import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import yaml
import pathlib

matplotlib.use('TkAgg')

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
            self.data = yaml.safe_load(file)

        self.images = self._parse_images(self.data['regions'])

    def _crop_image(self, img: np.ndarray, region: ROI):
        crop = img[
            region.y : region.y + region.h,
            region.x : region.x + region.w,
        ]
        return crop

    def _parse_images(self, dims: dict[str, dict[str, int]]):
        img_items = {}
        for key, dim in dims.items():
            region = ROI(
                dim['x'],
                dim['y'],
                dim['w'],
                dim['h'],
            )
            img_items[key] = self._crop_image(self.img, region)
        return img_items

    def plot_images(self):
        num_images = len(self.images) # + 1 # add one for full image
        img_cols = 4
        img_rows = num_images // 4 + int(num_images % 4 > 0)
        fig, axes = plt.subplots(img_rows, img_cols)
        for (label, img), ax in zip(self.images.items(), axes.flat):
            ax.imshow(img)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from king_tensor.globals import CONFIG_DIR, DATA_DIR
    img_path = DATA_DIR / 'IMG_1470.PNG'
    img = cv2.imread(img_path)
    config_path = CONFIG_DIR / 'ui.yaml'
    screen = Screen(img, config_path)
    plt.imshow(screen.images['elixir_bar'])
    plt.show()
    # screen.plot_images(PROJECT_DIR / 'data')
