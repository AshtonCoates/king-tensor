import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from PIL import Image
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

    def __init__(self, img: np.ndarray | Image.Image, config: dict[dict[str, int]]):
        """
        Args:
            img (np.ndarray | Image.Image): The image of the screen
            config (pathlib.Path): Path to UI element config file
        Returns:
            Screen: Object containing images for each cropped component
        """
        self.full_img = self._ensure_numpy(img)
        self.config = config
        self.screen_roi = ROI(
            self.config['screen']['x'],
            self.config['screen']['y'],
            self.config['screen']['w'],
            self.config['screen']['h'],
        )

        # crop the screen to just the game screen
        self.img = self._init_screen_img()
        self.images = self._parse_images()

    def _init_screen_img(self):
        game_img = self._crop_image(self.screen_roi, source_img=self.full_img)
        return game_img

    def _ensure_numpy(self, img: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            arr = np.array(img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
            return arr
        raise TypeError(f"Unsupported image type: {type(img)}")

    def _crop_image(self, region: ROI, source_img: np.ndarray | None = None):
        img = self.img if source_img is None else source_img
        crop = img[
            region.y : region.y + region.h,
            region.x : region.x + region.w,
        ]
        return crop

    def _parse_images(self):
        img_items = {'screen': self.img}
        for key, dim in self.config.items():
            if key == 'screen':
                continue
            region = self._roi_from_relative(dim)
            img_items[key] = self._crop_image(region)
        return img_items

    def _roi_from_relative(self, region_cfg: dict[str, int]) -> ROI:
        rel_x = max(region_cfg['x'], 0)
        rel_y = max(region_cfg['y'], 0)
        max_w = max(self.img.shape[1] - rel_x, 0)
        max_h = max(self.img.shape[0] - rel_y, 0)
        rel_w = min(region_cfg['w'], max_w)
        rel_h = min(region_cfg['h'], max_h)
        return ROI(rel_x, rel_y, rel_w, rel_h)

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
